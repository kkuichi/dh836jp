# similarity.py
# Funkcie pre porovnávanie autorov a názvov článkov.
# Porovnávanie autorov je realizované prostredníctvom metriky Damerau-Levenshtein.
# Porovnávanie názvov článkov na báze embeddingov je využívané výlučne pri detekcii errat.
# Damian Husár, 2026

import re
import unicodedata
import logging
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Prahová hodnota podobnosti DL pre porovnávanie autorov.
# Hodnota 0.65 bola stanovená experimentálne, pri hodnotách nižších (napr. 0.60) dochádzalo k zachyteniu
# falošne pozitívnych zhôd medzi autormi s podobnými, avšak odlišnými menami.
AUTHOR_DL_THRESHOLD = 0.65

# Prahová hodnota podobnosti embeddingov pre porovnávanie názvov článkov.
# Uplatňuje sa výlučne v rámci detekcie errat.
TITLE_SIMILARITY_THRESHOLD = 0.90

EMBEDDING_THRESHOLD = AUTHOR_DL_THRESHOLD

# Množina kľúčových slov identifikujúcich erratum a corrigendum v anglickom aj slovenskom jazyku.
ERRATA_KEYWORDS = {"erratum", "errata", "corrigendum", "correction", "oprava", "korekcia"}

logger.info(f"Načítavam model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)


# ---- Normalizácia ----

# Odstraňuje diakritiku prostredníctvom NFKD dekompozície a prevedie text na malé písmená.
# NFKD rozloží viacdielne znaky (napr. 'š' na 's' + combining char). 
# Kombinačné znaky sú následne eliminované.
def normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize('NFKD', name)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


# Zjednocuje varianty pomlčiek na štandardný znak, odstraňuje nadbytočné biele znaky a prevedie text na malé písmená. 
# Výsledkom je tvar vhodný na porovnanie názvov článkov.
def normalize_title(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-')
    return re.sub(r'\s+', ' ', text).strip().lower()


# ---- Detekcia errat ----

# Určuje, či citujúci záznam predstavuje erratum. Overenie prebieha v troch krokoch:
# kontrola kľúčových slov, presná zhoda normalizovaných názvov a sémantická podobnosť
# prostredníctvom embeddingového modelu (len ak predchádzajúce kroky zhodu nepotvrdili).
def detect_errata(citing_title: str, main_title: str,
                  title_threshold: float = TITLE_SIMILARITY_THRESHOLD) -> bool:
    c_norm = normalize_title(citing_title)
    m_norm = normalize_title(main_title)
    if set(c_norm.split()) & ERRATA_KEYWORDS:
        return True
    if c_norm == m_norm:
        return True
    return check_text_similarity(c_norm, m_norm) >= title_threshold


# ---- Jaro-Winkler ----

def _jaro(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Okno zhody je definované štandardnou formulou pre Jarovu metriku.
    match_dist = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


# Rozširuje Jarovu metriku o bonus za zhodný prefix (maximálne 4 znaky).
# Parameter p=0.1 predstavuje štandardnú hodnotu váhy prefixu podľa Winklerovej modifikácie.
def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    jaro_score = _jaro(s1, s2)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro_score + prefix * p * (1 - jaro_score)


# Porovnáva dve mená autorov prostredníctvom metriky Jaro-Winkler po predchádzajúcej normalizácii.
def compare_author_names(name1: str, name2: str) -> float:
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if n1 == n2:
        return 1.0
    return jaro_winkler(n1, n2)


# ---- Damerau-Levenshtein ----

# Vypočíta editačnú vzdialenosť Damerau-Levenshtein medzi dvoma reťazcami.
# Oproti klasickej Levenshteinovej vzdialenosti je transpozícia susedných znakov považovaná za jednu editačnú operáciu.
# Táto vlastnosť je pri porovnávaní mien autorov obzvlášť prínosná,
# keďže vzájomná zámena susedných znakov predstavuje bežný typ preklep (napr. „Novák" vs. „Nvoák").
def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Dynamická matica je inicializovaná tak, že d[i][0]=i a d[0][j]=j predstavujú základné hodnoty pre porovnanie s prázdnym reťazcom.
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # delete
                d[i][j - 1] + 1,        # insert
                d[i - 1][j - 1] + cost  # replace
            )
            # Ak sú susedné znaky vzájomne prehodené, transpozícia je zahrnutá ako jedna operácia v súlade s definíciou DL vzdialenosti.
            if (i > 1 and j > 1
                    and s1[i - 1] == s2[j - 2]
                    and s1[i - 2] == s2[j - 1]):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)

    return d[len1][len2]


# Normalizuje editačnú vzdialenosť DL na interval [0.0, 1.0] vydelením dĺžkou dlhšieho reťazca.
# Hodnota 1.0 zodpovedá úplnej zhode, hodnota 0.0 predstavuje maximálnu odlišnosť.
def damerau_levenshtein_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = damerau_levenshtein_distance(s1, s2)
    return 1.0 - dist / max_len


# Produkčná verzia porovnávania mien autorov. 
# Mená sú pred porovnaním normalizované a podobnosť je vypočítaná prostredníctvom metriky Damerau-Levenshtein.
def compare_author_names_dl(name1: str, name2: str) -> float:
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if n1 == n2:
        return 1.0
    return damerau_levenshtein_similarity(n1, n2)


# ---- Porovnávanie zoznamov autorov ----

# Prechádza všetky kombinácie autorov medzi dvoma článkami a pre každý pár určuje mieru podobnosti.
# Pokiaľ je pre oboch autorov dostupný identifikátor osoby z databázy CREPČ, podobnosť je určená
# priamym porovnaním identifikátorov — tento prístup je spoľahlivejší ako textová podobnosť.
# V prípade absencie identifikátora je uplatnená metrika Damerau-Levenshtein.
# Funkcia vracia maximálnu dosiahnutú hodnotu podobnosti a zoznam dvojíc nad prahom.
def check_authors_similarity(authors_main: list, authors_other: list) -> tuple:
    if not authors_main or not authors_other:
        return 0.0, []

    max_score = 0.0
    matches = []

    for a_main in authors_main:
        if isinstance(a_main, dict):
            name_main = a_main.get("name", "")
            id_main   = a_main.get("id", "").strip()
        else:
            name_main = str(a_main)
            id_main   = ""

        for a_other in authors_other:
            if isinstance(a_other, dict):
                name_other = a_other.get("name", "")
                id_other   = a_other.get("id", "").strip()
            else:
                name_other = str(a_other)
                id_other   = ""

            # Pri dostupnosti identifikátora CREPČ pre oboch autorov je textová podobnosť nahradená priamym porovnaním identifikátorov.
            if id_main and id_other:
                score  = 1.0 if id_main == id_other else 0.0
                method = "ID"
            else:
                score  = compare_author_names_dl(name_main, name_other)
                method = "DL"

            if score > max_score:
                max_score = score

            if score >= AUTHOR_DL_THRESHOLD:
                matches.append(
                    f"{name_main} <-> {name_other} "
                    f"(score: {round(score, 3)}, method: {method})"
                )

    return max_score, matches


# Implementácia je zhodná s check_authors_similarity, avšak využíva výhradne metriku Damerau-Levenshtein
# Táto funkcia je určená pre Experiment C, kde evaluate.py explicitne vyžaduje porovnanie oboch metód bez zásahu do produkčnej verzie.
def check_authors_similarity_dl(authors_main: list, authors_other: list) -> tuple:
    if not authors_main or not authors_other:
        return 0.0, []

    max_score = 0.0
    matches = []

    for a_main in authors_main:
        if isinstance(a_main, dict):
            name_main = a_main.get("name", "")
            id_main   = a_main.get("id", "").strip()
        else:
            name_main = str(a_main)
            id_main   = ""

        for a_other in authors_other:
            if isinstance(a_other, dict):
                name_other = a_other.get("name", "")
                id_other   = a_other.get("id", "").strip()
            else:
                name_other = str(a_other)
                id_other   = ""

            if id_main and id_other:
                score  = 1.0 if id_main == id_other else 0.0
                method = "ID"
            else:
                score  = compare_author_names_dl(name_main, name_other)
                method = "DL"

            if score > max_score:
                max_score = score

            if score >= AUTHOR_DL_THRESHOLD:
                matches.append(
                    f"{name_main} <-> {name_other} "
                    f"(score: {round(score, 3)}, method: {method})"
                )

    return max_score, matches


# ---- Embedding podobnosť názvov (výlučne pre detekciu errat) ----

# Vypočíta kosínusovú podobnosť dvoch názvov článkov prostredníctvom vektorových reprezentácií generovaných modelom sentence-transformers. 
# Táto funkcia je využívaná výlučne pri detekcii errat. 
def check_text_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0

    t1 = text1.lower().strip()
    t2 = text2.lower().strip()

    if t1 == t2:
        return 1.0

    emb1 = model.encode(t1, convert_to_tensor=True)
    emb2 = model.encode(t2, convert_to_tensor=True)

    return float(util.cos_sim(emb1, emb2))
