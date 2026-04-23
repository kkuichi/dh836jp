# evaluate.py
# Evaluačný skript pre systém identifikácie autocitácií a errat.
# Spustením skriptu sa automaticky vykonajú všetky tri experimenty:
# Výstupné súbory sú uložené do aktuálneho adresára.
#
# Experiment A — porovnanie embedding modelov pre detekciu errat
# Experiment B — sweep prahovej hodnoty DL (0.30 – 0.95), určenie optimálneho prahu
# Experiment C — priame porovnanie metód Jaro-Winkler a Damerau-Levenshtein
#
# Damian Husár, 2026

import json
import csv
import sys
import time
from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer, util

from similarity import (
    check_authors_similarity,
    check_authors_similarity_dl,
    detect_errata,
    normalize_title,
    AUTHOR_DL_THRESHOLD,
    ERRATA_KEYWORDS,
    TITLE_SIMILARITY_THRESHOLD,
)


# Modely zahrnuté do porovnania v Experimente A. 
# Model all-MiniLM-L6-v2 predstavuje baseline nasadenú v produkčnom systéme, all-mpnet-base-v2 slúži
# ako väčší referenčný model s vyššou presnosťou.
MODELS_TO_COMPARE = [
    {
        "name":        "all-MiniLM-L6-v2",
        "model_id":    "sentence-transformers/all-MiniLM-L6-v2",
    },
    {
        "name":        "all-mpnet-base-v2",
        "model_id":    "sentence-transformers/all-mpnet-base-v2",
    },
    {
        "name":        "paraphrase-MiniLM-L6-v2",
        "model_id":    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    },
]

# Hraničné hodnoty a krok sweepov pre Experimenty B a C.
DL_SWEEP_MIN  = 0.30
DL_SWEEP_MAX  = 0.95
DL_SWEEP_STEP = 0.05

# ---- dátové triedy pre výsledky ----

# Reprezentuje jeden záznam z testovacieho datasetu vrátane metadát oboch článkov
# a anotovaného skutočného zaradenia (true_label).
@dataclass
class EvalRecord:
    id:                  str
    main_authors:        list
    main_title:          str
    citing_authors:      list
    citing_title:        str
    is_xml_autocitation: bool
    true_label:          str


# Uchováva výsledky jedného embedding modelu v Experimente A vrátane časov načítania
# a inferencie, ktoré sú relevantné pri porovnaní praktickej použiteľnosti modelov.
@dataclass
class ErrataModelResult:
    model_name:   str
    model_id:     str
    threshold:    float
    load_time_s:  float = 0.0
    infer_time_s: float = 0.0
    err_tp: int = 0
    err_fp: int = 0
    err_tn: int = 0
    err_fn: int = 0
    errors: list = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.err_tp + self.err_fp
        return self.err_tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.err_tp + self.err_fn
        return self.err_tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.err_tp + self.err_fp + self.err_tn + self.err_fn
        return (self.err_tp + self.err_tn) / total if total > 0 else 0.0


# Uchováva výsledky evaluácie autocitácií pre jeden konkrétny prah v rámci sweepového experimentu B.
@dataclass
class SweepResult:
    threshold: float
    auto_tp: int = 0
    auto_fp: int = 0
    auto_tn: int = 0
    auto_fn: int = 0

    @property
    def precision(self) -> float:
        d = self.auto_tp + self.auto_fp
        return self.auto_tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.auto_tp + self.auto_fn
        return self.auto_tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.auto_tp + self.auto_fp + self.auto_tn + self.auto_fn
        return (self.auto_tp + self.auto_tn) / total if total > 0 else 0.0


# Reprezentuje jeden riadok porovnávacej tabuľky Experimentu C.
# Pre každú prahovú hodnotu sú súčasne uložené metriky oboch porovnávaných metód.
# Polia s prefixom jw_ zodpovedajú metóde Jaro-Winkler, polia s prefixom dl_
# metóde Damerau-Levenshtein, čo umožňuje ich priame porovnanie v rámci jedného riadku.
@dataclass
class MethodSweepRow:
    threshold:    float
    jw_precision: float = 0.0
    jw_recall:    float = 0.0
    jw_f1:        float = 0.0
    jw_accuracy:  float = 0.0
    jw_tp: int = 0
    jw_fp: int = 0
    jw_tn: int = 0
    jw_fn: int = 0
    dl_precision: float = 0.0
    dl_recall:    float = 0.0
    dl_f1:        float = 0.0
    dl_accuracy:  float = 0.0
    dl_tp: int = 0
    dl_fp: int = 0
    dl_tn: int = 0
    dl_fn: int = 0


# ---- načítanie datasetu ----

# Konvertuje záznam autora z ľubovoľného vstupného formátu na jednotný slovníkso štruktúrou {"name": ..., "id": ...}, 
# aby bolo možné dataset načítaný z rôznych zdrojov spracovávať jednotným spôsobom.
def _to_author_dicts(authors_raw: list) -> list:
    result = []
    for a in authors_raw:
        if isinstance(a, dict):
            result.append({"name": a.get("name", ""), "id": a.get("id", "").strip()})
        else:
            result.append({"name": str(a), "id": ""})
    return result


# Načíta JSON dataset a prevedie záznamy na objekty EvalRecord. 
# Záznamy s hodnotou true_label = null sú vynechané, keďže vyžadujú ručnú anotáciu a nie sú spôsobilé na zahrnutie do evaluácie.
def load_dataset(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    records = []
    skipped = 0
    for item in raw:
        if item.get("true_label") is None:
            skipped += 1
            continue
        records.append(EvalRecord(
            id                  = str(item["id"]),
            main_authors        = _to_author_dicts(item["main_authors"]),
            main_title          = item["main_title"],
            citing_authors      = _to_author_dicts(item["citing_authors"]),
            citing_title        = item["citing_title"],
            is_xml_autocitation = item.get("is_xml_autocitation", False),
            true_label          = item["true_label"].upper()
        ))

    if skipped:
        print(f"  Preskočených {skipped} záznamov s true_label = null "
              f"(vyžadujú ručnú anotáciu).")
    return records


# ---- pomocné funkcie ----

# Zjednodušená verzia detekcie errat bez volania embedding modelu. 
# Využíva výhradne kontrolu kľúčových slov a presnú zhodu názvov. 
# Zámerom je optimalizovať výpočtovú náročnosť v Experimentoch B a C, kde by volanie modelu pre každý záznam a každú
# prahovú hodnotu predstavovalo zbytočné spomalenie.
def _errata_simple(citing_title: str, main_title: str) -> bool:
    c_norm = normalize_title(citing_title)
    m_norm = normalize_title(main_title)
    if set(c_norm.split()) & ERRATA_KEYWORDS:
        return True
    return c_norm == m_norm and c_norm != ""


# Na základe predpočítaných skóre a zadanej prahovej hodnoty vypočíta hodnoty
# matice zámen (TP/FP/TN/FN) spolu s metrikami presnosti, návratnosti, F1-miery a accuracy. 
# Errata záznamy sú zo sweepového hodnotenia vylúčené, keďže prahová hodnota sa vzťahuje výhradne na detekciu autocitácií.
def _count_for_threshold(precomputed: list, score_key: str, t: float) -> tuple:
    tp = fp = tn = fn = 0
    for p in precomputed:
        if p["is_errata"]:
            continue
        true_is_auto = (p["true_label"] == "AUTOCITATION")
        pred_auto    = (p[score_key] >= t)
        if true_is_auto:
            tp += 1 if pred_auto else 0
            fn += 0 if pred_auto else 1
        else:
            fp += 1 if pred_auto else 0
            tn += 0 if pred_auto else 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    tot  = tp + fp + tn + fn
    acc  = (tp + tn) / tot if tot > 0 else 0.0
    return (round(prec, 4), round(rec, 4), round(f1, 4), round(acc, 4),
            tp, fp, tn, fn)


# ---- Experiment A ----

# Detekuje erratum prostredníctvom konkrétneho embedding modelu a zadanej prahove hodnoty.
# Kontrola kľúčových slov a presná zhoda názvov sú vyhodnotené bez volania modelu.
# Embedding je vypočítaný len v prípade, že ani jedna z týchto podmienok nie je splnená. 
# Výstupom je dvojica (predikcia, skóre podobnosti).
def _detect_errata_embedding(citing_title: str, main_title: str,
                              model: SentenceTransformer,
                              threshold: float) -> tuple:
    c_norm = normalize_title(citing_title)
    m_norm = normalize_title(main_title)
    if set(c_norm.split()) & ERRATA_KEYWORDS:
        return True, 1.0
    if c_norm == m_norm:
        return True, 1.0
    emb1  = model.encode(c_norm, convert_to_tensor=True)
    emb2  = model.encode(m_norm, convert_to_tensor=True)
    score = float(util.cos_sim(emb1, emb2))
    return score >= threshold, score


# Experiment A: pre každý model v zozname MODELS_TO_COMPARE je zaznamenaný čas načítania a čas inferencie,
# po ktorom je vyhodnotená detekcia errat na príslušnej podmnožine datasetu. 
# Meranie časov je súčasťou experimentu, keďže praktická použiteľnosť modelu závisí nielen od jeho presnosti, ale aj od výpočtovej náročnosti.
def run_experiment_a(records: list, threshold: float) -> list:
    errata_records = [r for r in records
                      if r.true_label in ("ERRATUM", "NOT_AUTOCITATION")]
    results = []

    for i, cfg in enumerate(MODELS_TO_COMPARE, 1):
        print(f"\n  [{i}/{len(MODELS_TO_COMPARE)}] {cfg['name']}")
        print(f"  Načítavam model ...", end=" ", flush=True)

        t0        = time.time()
        model     = SentenceTransformer(cfg["model_id"])
        load_time = round(time.time() - t0, 3)
        print(f"hotovo ({load_time} s)")

        res = ErrataModelResult(
            model_name=cfg["name"], model_id=cfg["model_id"],
            threshold=threshold, load_time_s=load_time,
        )

        t0 = time.time()
        for rec in errata_records:
            true_is_err = (rec.true_label == "ERRATUM")
            pred, score = _detect_errata_embedding(
                rec.citing_title, rec.main_title, model, threshold
            )
            if true_is_err:
                if pred:
                    res.err_tp += 1
                else:
                    res.err_fn += 1
                    res.errors.append({"type": "FN_ERRATA", "id": rec.id,
                                       "score": round(score, 4),
                                       "citing_title": rec.citing_title,
                                       "main_title":   rec.main_title})
            else:
                if pred:
                    res.err_fp += 1
                    res.errors.append({"type": "FP_ERRATA", "id": rec.id,
                                       "score": round(score, 4),
                                       "citing_title": rec.citing_title,
                                       "main_title":   rec.main_title})
                else:
                    res.err_tn += 1

        res.infer_time_s = round(time.time() - t0, 3)
        print(f"  Čas inferencie: {res.infer_time_s} s  |  "
              f"TP={res.err_tp} FP={res.err_fp} TN={res.err_tn} FN={res.err_fn}  |  "
              f"F1={res.f1:.4f}")
        results.append(res)

    return results


# ---- Experiment B ----

# Pre každý prah DL z rozsahu 0.30 – 0.95 vyhodnotí detekciu autocitácií.
# Skóre autorov je predpočítané pred spustením sweepu
# Errata záznamy sú z evaluácie vylúčené, pretože sú vyhodnocované samostatne v Experimente A.
def run_experiment_b(records: list) -> list:
    autocit_records = [r for r in records
                       if r.true_label in ("AUTOCITATION", "NOT_AUTOCITATION")]

    print("\n  Predpočítavam DL skóre autorov ...", end=" ", flush=True)
    precomputed = []
    for rec in autocit_records:
        score, _ = check_authors_similarity_dl(rec.main_authors, rec.citing_authors)
        precomputed.append({
            "true_label": rec.true_label,
            "is_errata":  _errata_simple(rec.citing_title, rec.main_title),
            "dl_score":   score,
            "id":         rec.id,
        })
    print("hotovo")

    n_steps    = round((DL_SWEEP_MAX - DL_SWEEP_MIN) / DL_SWEEP_STEP) + 1
    thresholds = [round(DL_SWEEP_MIN + i * DL_SWEEP_STEP, 2) for i in range(n_steps)]
    results    = []

    for t in thresholds:
        res = SweepResult(threshold=t)
        for p in precomputed:
            if p["is_errata"]:
                continue
            true_is_auto = (p["true_label"] == "AUTOCITATION")
            pred_auto    = (p["dl_score"] >= t)
            if true_is_auto:
                res.auto_tp += 1 if pred_auto else 0
                res.auto_fn += 0 if pred_auto else 1
            else:
                res.auto_fp += 1 if pred_auto else 0
                res.auto_tn += 0 if pred_auto else 1
        results.append(res)
        print(f"  DL={t:.2f}  |  "
              f"TP={res.auto_tp} FP={res.auto_fp} TN={res.auto_tn} FN={res.auto_fn}  |  "
              f"Prec={res.precision:.4f}  Rec={res.recall:.4f}  F1={res.f1:.4f}")

    return results


# ---- Experiment C ----

# Implementácia je zhodná s Experimentom B, avšak namiesto samotného DL skóre sú pre každý záznam predpočítané skóre oboch metód (JW aj DL) súčasne.
# Sweep následne prebieha paralelne pre obe metódy, čo umožňuje ich priame porovnanie pri každej prahovej hodnote bez potreby opakovaného výpočtu skóre.
def run_experiment_c(records: list) -> list:
    autocit_records = [r for r in records
                       if r.true_label in ("AUTOCITATION", "NOT_AUTOCITATION")]

    print("\n  Predpočítavam JW a DL skóre ...", end=" ", flush=True)
    precomputed = []
    for rec in autocit_records:
        jw_score, _ = check_authors_similarity(rec.main_authors, rec.citing_authors)
        dl_score, _ = check_authors_similarity_dl(rec.main_authors, rec.citing_authors)
        precomputed.append({
            "true_label": rec.true_label,
            "is_errata":  _errata_simple(rec.citing_title, rec.main_title),
            "jw_score":   jw_score,
            "dl_score":   dl_score,
        })
    print("hotovo")

    n_steps    = round((DL_SWEEP_MAX - DL_SWEEP_MIN) / DL_SWEEP_STEP) + 1
    thresholds = [round(DL_SWEEP_MIN + i * DL_SWEEP_STEP, 2) for i in range(n_steps)]
    rows       = []

    for t in thresholds:
        jw = _count_for_threshold(precomputed, "jw_score", t)
        dl = _count_for_threshold(precomputed, "dl_score", t)
        rows.append(MethodSweepRow(
            threshold=t,
            jw_precision=jw[0], jw_recall=jw[1], jw_f1=jw[2], jw_accuracy=jw[3],
            jw_tp=jw[4], jw_fp=jw[5], jw_tn=jw[6], jw_fn=jw[7],
            dl_precision=dl[0], dl_recall=dl[1], dl_f1=dl[2], dl_accuracy=dl[3],
            dl_tp=dl[4], dl_fp=dl[5], dl_tn=dl[6], dl_fn=dl[7],
        ))

    return rows


# ---- výstupné funkcie ----

# Vypíše súhrnnú tabuľku Experimentu A do štandardného výstupu.
# Pre každý testovaný model zobrazí metriky (Precision, Recall, F1, Accuracy) spolu s časom
# inferencie a hodnotami matice zámen. Na záver označí model s najvyšším F1.
def print_experiment_a(results: list):
    sep  = "=" * 74
    line = "-" * 74
    print(f"\n\n{sep}")
    print("  EXPERIMENT A — Porovnanie embedding modelov pre detekciu errát")
    print(f"  Prah kosínusovej podobnosti: {results[0].threshold:.2f}")
    print(sep)
    print(f"\n  {'Model':<32} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Čas':>8}")
    print(f"  {line}")
    for r in results:
        print(f"  {r.model_name:<32} {r.precision:>8.4f} {r.recall:>8.4f} "
              f"{r.f1:>8.4f} {r.accuracy:>8.4f} {r.infer_time_s:>7.2f}s")
    print(f"\n  {'Model':<32} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    print(f"  {line}")
    for r in results:
        print(f"  {r.model_name:<32} {r.err_tp:>6} {r.err_fp:>6} "
              f"{r.err_tn:>6} {r.err_fn:>6}")
    best = max(results, key=lambda r: r.f1)
    print(f"\n  Najlepší model podľa F1: {best.model_name}  (F1 = {best.f1:.4f})")
    print(f"\n{sep}")


# Vypíše tabuľku Experimentu B — každý riadok zodpovedá jednej prahovej hodnote DL a obsahuje metriky detekcie autocitácií. 
# Na záver označí optimálny prah podľa F1.
def print_experiment_b(results: list):
    sep  = "=" * 74
    line = "-" * 74
    print(f"\n\n{sep}")
    print("  EXPERIMENT B — Sweep prahovej hodnoty Damerau-Levenshtein (autocitácie)")
    print(f"  Rozsah prahov: {DL_SWEEP_MIN:.2f} – {DL_SWEEP_MAX:.2f}, krok {DL_SWEEP_STEP:.2f}")
    print(sep)
    print(f"\n  {'DL prah':>8} {'Prec':>10} {'Recall':>10} {'F1':>10} "
          f"{'Acc':>10} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print(f"  {line}")
    for r in results:
        print(f"  {r.threshold:>8.2f} {r.precision:>10.4f} {r.recall:>10.4f} "
              f"{r.f1:>10.4f} {r.accuracy:>10.4f} "
              f"{r.auto_tp:>5} {r.auto_fp:>5} {r.auto_tn:>5} {r.auto_fn:>5}")
    best = max(results, key=lambda r: r.f1)
    print(f"\n  Optimálny prah podľa F1: {best.threshold:.2f}  "
          f"(F1={best.f1:.4f}  Precision={best.precision:.4f}  Recall={best.recall:.4f})")
    print(f"\n{sep}")


# Vypíše porovnávaciu tabuľku Experimentu C — každý riadok obsahuje metriky oboch metód (JW aj DL) pre danú prahovú hodnotu. 
# Lepšia metóda v každom riadku je označená hviezdičkou (*). Na záver je vyhodnotená víťazná metóda podľa F1.
def print_experiment_c(rows: list):
    sep  = "=" * 88
    line = "-" * 88
    print(f"\n\n{sep}")
    print("  EXPERIMENT C — Porovnanie metód: Jaro-Winkler (JW) vs Damerau-Levenshtein (DL)")
    print(f"  Sweep prahov {DL_SWEEP_MIN:.2f} – {DL_SWEEP_MAX:.2f}, krok {DL_SWEEP_STEP:.2f}")
    print(sep)
    print(f"\n  {'Prah':>6}  {'JW Prec':>9} {'JW Rec':>8} {'JW F1':>8} {'JW Acc':>8}"
          f"  {'DL Prec':>9} {'DL Rec':>8} {'DL F1':>8} {'DL Acc':>8}")
    print(f"  {line}")
    for r in rows:
        jw_mark = " *" if r.jw_f1 > r.dl_f1 else "  "
        dl_mark = " *" if r.dl_f1 > r.jw_f1 else "  "
        print(f"  {r.threshold:>6.2f}  "
              f"{r.jw_precision:>9.4f} {r.jw_recall:>8.4f} {r.jw_f1:>8.4f}{jw_mark}"
              f"{r.jw_accuracy:>8.4f}  "
              f"{r.dl_precision:>9.4f} {r.dl_recall:>8.4f} {r.dl_f1:>8.4f}{dl_mark}"
              f"{r.dl_accuracy:>8.4f}")
    best_jw = max(rows, key=lambda r: r.jw_f1)
    best_dl = max(rows, key=lambda r: r.dl_f1)
    print(f"\n  Najlepší JW: prah={best_jw.threshold:.2f}  F1={best_jw.jw_f1:.4f}")
    print(f"  Najlepší DL: prah={best_dl.threshold:.2f}  F1={best_dl.dl_f1:.4f}")
    winner = "Jaro-Winkler" if best_jw.jw_f1 >= best_dl.dl_f1 else "Damerau-Levenshtein"
    diff   = abs(best_jw.jw_f1 - best_dl.dl_f1)
    print(f"\n  Víťazná metóda podľa F1: {winner}  (rozdiel = {diff:.4f})")
    print(f"  (* = lepšia metóda pre daný prah)")
    print(f"\n{sep}")


# ---- ukladanie výsledkov ----

# Výsledky Experimentu A sú serializované do JSON formátu vrátane metrík, matice zámen a zoznamu chybových záznamov pre každý testovaný model.
def save_experiment_a_json(results: list, path: str):
    export = [
        {
            "model_name": r.model_name,
            "threshold": r.threshold,
            "load_time_s": r.load_time_s, "infer_time_s": r.infer_time_s,
            "metrics": {
                "precision": round(r.precision, 4), "recall": round(r.recall, 4),
                "f1": round(r.f1, 4), "accuracy": round(r.accuracy, 4),
            },
            "confusion_matrix": {
                "tp": r.err_tp, "fp": r.err_fp,
                "tn": r.err_tn, "fn": r.err_fn,
            },
            "errors": r.errors,
        }
        for r in results
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=4)
    print(f"\n  JSON Experiment A: {path}")


# Výsledky Experimentu B sú uložené do CSV formátu, ktorý umožňuje priame otvorenie v tabuľkovom procesore.
def save_experiment_b_csv(results: list, path: str):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["DL prah", "Precision", "Recall", "F1", "Accuracy",
                         "TP", "FP", "TN", "FN"])
        for r in results:
            writer.writerow([
                r.threshold,
                round(r.precision, 4), round(r.recall, 4),
                round(r.f1, 4),        round(r.accuracy, 4),
                r.auto_tp, r.auto_fp, r.auto_tn, r.auto_fn,
            ])
    print(f"  CSV Experiment B: {path}")


# Výsledky Experimentu C sú uložené do CSV formátu, kde každý riadok zodpovedá jednej prahovej hodnote a obsahuje metriky oboch metód 
# (JW aj DL) pre ich vzájomné porovnanie.
def save_experiment_c_csv(rows: list, path: str):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Prah",
            "JW_Precision", "JW_Recall", "JW_F1", "JW_Accuracy",
            "JW_TP", "JW_FP", "JW_TN", "JW_FN",
            "DL_Precision", "DL_Recall", "DL_F1", "DL_Accuracy",
            "DL_TP", "DL_FP", "DL_TN", "DL_FN",
        ])
        for r in rows:
            writer.writerow([
                r.threshold,
                r.jw_precision, r.jw_recall, r.jw_f1, r.jw_accuracy,
                r.jw_tp, r.jw_fp, r.jw_tn, r.jw_fn,
                r.dl_precision, r.dl_recall, r.dl_f1, r.dl_accuracy,
                r.dl_tp, r.dl_fp, r.dl_tn, r.dl_fn,
            ])
    print(f"  CSV Experiment C: {path}")


# Výsledky všetkých troch experimentov sú spojené do jedného JSON súboru.
def save_all_experiments_json(exp_a: list, exp_b: list, exp_c: list, path: str):
    export = {
        "experiment_A_errata_models": [
            {
                "model_name": r.model_name,
                "threshold": r.threshold,
                "load_time_s": r.load_time_s, "infer_time_s": r.infer_time_s,
                "metrics": {
                    "precision": round(r.precision, 4), "recall": round(r.recall, 4),
                    "f1": round(r.f1, 4), "accuracy": round(r.accuracy, 4),
                },
                "confusion_matrix": {
                    "tp": r.err_tp, "fp": r.err_fp,
                    "tn": r.err_tn, "fn": r.err_fn,
                },
                "errors": r.errors,
            }
            for r in exp_a
        ],
        "experiment_B_dl_sweep": [
            {
                "dl_threshold": r.threshold,
                "metrics": {
                    "precision": round(r.precision, 4), "recall": round(r.recall, 4),
                    "f1": round(r.f1, 4), "accuracy": round(r.accuracy, 4),
                },
                "confusion_matrix": {
                    "tp": r.auto_tp, "fp": r.auto_fp,
                    "tn": r.auto_tn, "fn": r.auto_fn,
                },
            }
            for r in exp_b
        ],
        "experiment_C_method_comparison": [
            {
                "threshold": r.threshold,
                "jaro_winkler": {
                    "precision": r.jw_precision, "recall": r.jw_recall,
                    "f1": r.jw_f1, "accuracy": r.jw_accuracy,
                    "confusion_matrix": {
                        "tp": r.jw_tp, "fp": r.jw_fp,
                        "tn": r.jw_tn, "fn": r.jw_fn,
                    },
                },
                "damerau_levenshtein": {
                    "precision": r.dl_precision, "recall": r.dl_recall,
                    "f1": r.dl_f1, "accuracy": r.dl_accuracy,
                    "confusion_matrix": {
                        "tp": r.dl_tp, "fp": r.dl_fp,
                        "tn": r.dl_tn, "fn": r.dl_fn,
                    },
                },
            }
            for r in exp_c
        ],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=4)
    print(f"  JSON (všetky experimenty): {path}")


# ---- main ----

def main():
    # Cestu k datasetu možno zadať ako prvý argument, inak sa použije predvolená hodnota.
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "test_dataset.json"

    print(f"Načítavam dataset: {dataset_path}")
    records = load_dataset(dataset_path)
    print(f"Načítaných záznamov: {len(records)}\n")

    if not records:
        print("CHYBA: Dataset je prázdny alebo všetky záznamy vyžadujú anotáciu.")
        return

    print("=" * 88)
    print("  EXPERIMENTY — Porovnanie metód a modelov")
    print("=" * 88)

    print(f"\n{'─' * 88}")
    print("  EXPERIMENT A — Porovnanie embedding modelov")
    print(f"{'─' * 88}")
    exp_a = run_experiment_a(records, TITLE_SIMILARITY_THRESHOLD)
    print_experiment_a(exp_a)
    save_experiment_a_json(exp_a, "errata_model_comparison.json")

    print(f"\n{'─' * 88}")
    print("  EXPERIMENT B — Sweep DL prahov")
    print(f"{'─' * 88}")
    exp_b = run_experiment_b(records)
    print_experiment_b(exp_b)
    save_experiment_b_csv(exp_b, "autocitation_threshold_sweep.csv")

    print(f"\n{'─' * 88}")
    print("  EXPERIMENT C — Porovnanie JW vs DL")
    print(f"{'─' * 88}")
    exp_c = run_experiment_c(records)
    print_experiment_c(exp_c)
    save_experiment_c_csv(exp_c, "method_comparison_jw_vs_dl.csv")

    save_all_experiments_json(exp_a, exp_b, exp_c, "experiment_results.json")
    print("\nExperimenty dokončené.\n")


if __name__ == "__main__":
    main()
