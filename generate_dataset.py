# generate_dataset.py
# Bc. Damián Husár, damian.husar999@gmail.com, 2026
#
# Skript zabezpečuje sťahovanie bibliografických záznamov z informačného
# systému CREPČ prostredníctvom protokolu OAI-PMH a ich prípravu vo formáte
# JSON pre následnú anotáciu a evaluáciu.
#
# Dôležité: v tejto fáze sa žiadna analýza podobnosti nevykonáva.
# Použitie metód JW alebo DL pri priraďovaní klasifikačných označení
# by zaviedlo skreslenie do referenčného datasetu (ground truth).
#
# Klasifikačné označenie (true_label) sa priradí automaticky len
# pri výskyte vysokospoľahlivých signalov:
#   ERRATUM      <- kľúčové slovo errata/corrigendum v názve citujúceho článku
#   AUTOCITATION <- XML príznak autocitácie priamo z CREPČ
#   null         <- záznam bez spoľahlivého signalu, vyžaduje ručnú anotáciu
#
# Spustenie:
#   python generate_dataset.py --ids 123456 789012 345678
#   python generate_dataset.py --ids-file moje_ids.txt --from-date 2020-01-01

import json
import argparse
import requests
import xml.etree.ElementTree as ET
import re
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CREPC_GET_RECORD_URL   = "https://app.crepc.sk/oai"
CREPC_LIST_RECORDS_URL = "https://app.crepc.sk/oai/biblioCitations"

# Rovnaká sada kľúčových slov ako v module similarity.py — konzistentnosť klasifikácie
ERRATA_KEYWORDS = {"erratum", "errata", "corrigendum", "correction", "oprava", "korekcia"}


# ---- sieťové volania ----

def _safe_get(url: str, params: dict = None, timeout: int = 20):
    """
    HTTP GET s opakovaným pokusom pri zlyhaní — maximálne 2 pokusy s pauzou 1 s.
    Server CREPČ príležitostne neodpovedá pri prvej požiadavke, preto je
    mechanizmus opakovania nevyhnutný pre spoľahlivé sťahovanie väčšieho
    množstva záznamov. Pri neúspechu oboch pokusov vracia None.
    """
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            logger.warning(f"HTTP {response.status_code} (pokus {attempt + 1})")
        except requests.RequestException as e:
            logger.warning(f"Sieťová chyba (pokus {attempt + 1}): {e}")
        time.sleep(1)
    return None


# ---- parsovanie XML ----

def _strip_ns(tag: str) -> str:
    # OAI-PMH odpovede obsahujú tagy v plne kvalifikovanom tvare {http://...}nazov,
    # táto funkcia extrahuje iba lokálny názov tagu za znakom }
    return tag.split('}')[-1] if '}' in tag else tag


def _get_element_text(element, local_tag: str) -> str:
    # Vráti textový obsah prvého priameho potomka so zadaným lokálnym tagom,
    # alebo prázdny reťazec ak takýto potomok neexistuje
    for child in element:
        if _strip_ns(child.tag) == local_tag:
            return (child.text or "").strip()
    return ""


def _build_person_index(root) -> dict:
    """
    Zostrojí vyhľadávací index osôb vo formáte {person_id: {"name": ..., "id": ...}}.

    V niektorých záznamoch CREPČ nie je meno autora dostupné priamo v elemente
    cross_biblio_person, ale iba v samostatnom elemente rec_person inde v dokumente.
    Index umožňuje toto meno dohľadať podľa identifikátora osoby.
    Pri viacerých výskytoch rovnakého person_id sa uprednostní záznam s dlhším menom,
    pretože ten spravidla obsahuje aj krstné meno a je kompletnejší.
    """
    index = {}
    for elem in root.iter():
        if _strip_ns(elem.tag) != 'rec_person':
            continue
        person_id = elem.get('id', '').strip()
        if not person_id:
            continue
        lname = fname = labelname = ""
        for child in elem:
            local = _strip_ns(child.tag)
            if local == 'lastname':
                lname = (child.text or "").strip()
            elif local == 'firstname':
                fname = (child.text or "").strip()
            elif local == 'labelname':
                labelname = (child.text or "").strip()
        full_name = f"{lname} {fname}".strip() if lname else labelname
        if not full_name:
            continue
        if person_id not in index or len(full_name) > len(index[person_id]["name"]):
            index[person_id] = {"name": full_name, "id": person_id}
    return index


def _parse_authors(root) -> list:
    """
    Extrahuje zoznam autorov z XML záznamu vo formáte [{"name": ..., "id": ...}].

    Identifikátor osoby (person_id) je pri porovnávaní zásadný — keď je dostupný,
    modul similarity.py ho využíva na priamu zhodu namiesto výpočtu podobnosti mien
    metódou Damerau-Levenshtein. Staršie záznamy v CREPČ identifikátor nemusia
    obsahovať; v takom prípade sa porovnávanie opiera o reťazce mien.
    Duplicitné záznamy sú eliminované — prednostne podľa person_id,
    pri jeho absencii podľa celého mena.
    """
    person_index    = _build_person_index(root)
    main_rec_biblio = None
    for elem in root.iter():
        if _strip_ns(elem.tag) == 'rec_biblio':
            main_rec_biblio = elem
            break
    if main_rec_biblio is None:
        return []

    authors_list = []
    seen_ids     = set()
    seen_names   = set()

    for child in main_rec_biblio:
        if _strip_ns(child.tag) != 'cross_biblio_person':
            continue
        role = child.get('role', '')
        if role not in ('author', 'author_corporation'):
            continue

        lname = fname = labelname = person_id = ""
        for subchild in child:
            local = _strip_ns(subchild.tag)
            if local == 'rec_person':
                person_id = subchild.get('id', '').strip()
                for gc in subchild:
                    gl = _strip_ns(gc.tag)
                    if gl == 'lastname':
                        lname = (gc.text or "").strip()
                    elif gl == 'firstname':
                        fname = (gc.text or "").strip()
                    elif gl == 'labelname':
                        labelname = (gc.text or "").strip()

        if lname:
            full_name = f"{lname} {fname}".strip()
        elif labelname:
            full_name = labelname
        elif person_id and person_id in person_index:
            # záložný zdroj — meno dohľadané cez index osôb
            full_name = person_index[person_id]["name"]
        else:
            full_name = ""

        if not full_name:
            continue

        if person_id:
            if person_id in seen_ids:
                continue
            seen_ids.add(person_id)
        else:
            if full_name in seen_names:
                continue
            seen_names.add(full_name)

        authors_list.append({"name": full_name, "id": person_id})

    return authors_list


def _check_xml_autocitation(record, target_id_str: str) -> bool:
    """
    Overí prítomnosť XML príznaku autocitácie voči zadanému hlavnému článku.

    Systém CREPČ zaznamenáva autocitácie priamo do XML metadát záznamu
    prostredníctvom elementu cross_biblio_biblio s hodnotou autocitation=true.
    Ide o najpresnejší dostupný signal, keďže pochádza priamo z databázy
    a nevyžaduje výpočet podobnosti. Funkcia overí súčasné splnenie oboch
    podmienok: príznak autocitation=true a zhoda identifikátora s cieľovým článkom.
    """
    for bond in record.findall('.//{*}cross_biblio_biblio'):
        autocit_val = _get_element_text(bond, 'autocitation')
        if autocit_val.lower() != 'true':
            continue
        for child in bond:
            if _strip_ns(child.tag) == 'rec_biblio':
                if child.get('id', '').strip() == target_id_str:
                    return True
    return False


def _normalize_title(text: str) -> str:
    # Normalizácia názvu: zjednotenie typov pomlčiek, prevod na malé písmená
    # a zlúčenie viacnásobných medzier na jednu
    if not text:
        return ""
    text = text.replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-')
    return re.sub(r'\s+', ' ', text).strip().lower()


# ---- priradenie klasifikačného označenia ----

def _assign_label(citing_title: str, is_xml_autocitation: bool) -> tuple:
    """
    Priradí klasifikačné označenie záznamu výlučne na základe vysokospoľahlivých signalov.

    Metódy výpočtu podobnosti (Jaro-Winkler, Damerau-Levenshtein) sú zámerne
    vynechané — sú predmetom evaluácie v skripte evaluate.py a ich zahrnutie
    do priraďovania označení by invalidovalo referenčný dataset (ground truth).
    V záujme zachovania objektívnosti datasetu sa uprednostňuje väčší počet
    záznamov určených na ručnú anotáciu pred zavedením skreslenia.

    Vracia trojicu (label, confidence, reason).
    """
    c_norm = _normalize_title(citing_title)

    if set(c_norm.split()) & ERRATA_KEYWORDS:
        return "ERRATUM", "HIGH", "kľúčové slovo erraty v názve"

    if is_xml_autocitation:
        return "AUTOCITATION", "HIGH", "XML príznak autocitácie v CREPČ"

    return None, "NEEDS_REVIEW", "žiadny spoľahlivý signál — vyžaduje ručnú anotáciu"


# ---- sťahovanie dát z CREPČ ----

def fetch_main_article(record_id: str) -> dict | None:
    """
    Stiahne bibliografický záznam hlavného článku cez OAI-PMH GetRecord.
    Extrahuje názov článku a zoznam autorov. V prípade sieťovej chyby
    alebo neplatného XML vracia None a zaznamenáva problém do logu.
    """
    url = (
        f"{CREPC_GET_RECORD_URL}?verb=GetRecord"
        f"&metadataPrefix=xml-crepc2-flat4"
        f"&identifier=oai:crepc.sk:biblio/{record_id}"
    )
    response = _safe_get(url, timeout=15)
    if response is None:
        return None
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        logger.error(f"Chyba XML pre ID {record_id}: {e}")
        return None

    title_elem = root.find('.//{*}title')
    title      = (title_elem.text or "").strip() if title_elem is not None else ""
    authors    = _parse_authors(root)
    return {"authors": authors, "title": title}


def fetch_citations(target_id: str, from_date: str) -> list:
    """
    Stiahne citácie daného článku prostredníctvom OAI-PMH ListRecords.

    Využíva endpoint biblioCitations, ktorý bol experimentálne overený
    ako vhodnejší pre sťahovanie citácií ako štandardný OAI-PMH endpoint /oai.
    Pre každý nájdený záznam sa extrahujú autori, názov a overí sa prítomnosť
    XML príznaku autocitácie. Záznamy s príznakom status="deleted" sú vynechané.
    Žiadna analýza podobnosti sa v tejto funkcii nevykonáva.
    """
    params = {
        'verb':           'ListRecords',
        'metadataPrefix': 'xml-crepc2-flat4',
        'set':            target_id,
        'from':           from_date,
    }
    response = _safe_get(CREPC_LIST_RECORDS_URL, params=params, timeout=30)
    if response is None:
        return []
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        logger.error(f"Chyba XML pre set {target_id}: {e}")
        return []

    target_id_str   = str(target_id).strip()
    found_citations = []

    for record in root.findall('.//{*}record'):
        header = record.find('.//{*}header')
        if header is None or header.get('status') == 'deleted':
            continue
        identifier_elem = header.find('.//{*}identifier')
        if identifier_elem is None or not identifier_elem.text:
            continue
        # Identifikátor má formát "oai:crepc.sk:biblio/123456" —
        # pre ďalšie spracovanie sa extrahuje iba číselná časť za lomkou
        citing_id = identifier_elem.text.split('/')[-1]

        citing_title = ""
        rec_biblio   = record.find('.//{*}rec_biblio')
        if rec_biblio is not None:
            citing_title = _get_element_text(rec_biblio, 'title')

        is_xml_autocitation = _check_xml_autocitation(record, target_id_str)
        authors_list        = _parse_authors(record)

        found_citations.append({
            "id":                  citing_id,
            "authors":             authors_list,
            "citing_title":        citing_title,
            "is_xml_autocitation": is_xml_autocitation,
        })

    return found_citations


# ---- generovanie datasetu ----

def generate_dataset(main_ids: list, from_date: str, output: str):
    """
    Hlavná funkcia skriptu — pre každý zadaný identifikátor stiahne hlavný článok
    a jeho citácie, priradí klasifikačné označenia a uloží dataset do JSON súboru.

    Záznamy s vysokospoľahlivými signálmi sú označené automaticky, ostatné
    ostávajú s hodnotou null a vyžadujú ručnú anotáciu pred použitím v experimentoch.
    Pomocné polia s prefixom _ sú určené pre anotátora a nie sú súčasťou
    schémy datasetu pre evaluate.py — po dokončení anotácie je potrebné ich odstrániť.
    """
    dataset = []
    stats   = {
        "total":            0,
        "erratum":          0,
        "autocitation_xml": 0,
        "needs_review":     0,
        "skipped":          0,
    }

    for main_id in main_ids:
        print(f"\n{'─' * 62}")
        print(f"  Hlavný článok ID: {main_id}")

        main_doc = fetch_main_article(main_id)
        if not main_doc:
            logger.warning(f"Článok {main_id} sa nepodarilo stiahnuť — preskakujem.")
            stats["skipped"] += 1
            continue

        main_title   = main_doc["title"]
        main_authors = main_doc["authors"]

        print(f"  Názov:  {main_title[:70]}")
        print(f"  Autori: {', '.join(a['name'] for a in main_authors)}")

        citations = fetch_citations(main_id, from_date)
        print(f"  Citácií nájdených: {len(citations)}")
        print()

        for cit in citations:
            label, confidence, reason = _assign_label(
                citing_title        = cit["citing_title"],
                is_xml_autocitation = cit["is_xml_autocitation"],
            )

            record = {
                "id":                  cit["id"],
                "main_title":          main_title,
                "main_authors":        main_authors,
                "citing_title":        cit["citing_title"],
                "citing_authors":      cit["authors"],
                "is_xml_autocitation": cit["is_xml_autocitation"],
                "true_label":          label,
                # Pomocné polia pre anotátora — po dokončení anotácie odstrániť
                "_confidence":         confidence,
                "_reason":             reason,
                "_main_article_id":    main_id,
            }

            dataset.append(record)
            stats["total"] += 1

            if label == "ERRATUM":
                stats["erratum"] += 1
            elif label == "AUTOCITATION":
                stats["autocitation_xml"] += 1
            else:
                stats["needs_review"] += 1

            flag  = " ✓ HIGH" if confidence == "HIGH" else " <- ANOTUJ"
            short = cit["citing_title"][:52] if cit["citing_title"] else "(bez názvu)"
            label_str = label if label else "null"
            print(f"    [{label_str:<17}]{flag}")
            print(f"      Názov:  {short}")
            print(f"      Dôvod:  {reason}")
            print()

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"\n{'=' * 62}")
    print(f"  Dataset uložený: {output}")
    print(f"  Celkový počet záznamov : {stats['total']}")
    print(f"    -> ERRATUM (HIGH)      : {stats['erratum']}")
    print(f"    -> AUTOCITATION (HIGH) : {stats['autocitation_xml']}  (XML príznak CREPČ)")
    print(f"    -> null (NEEDS_REVIEW) : {stats['needs_review']}  <- vyžadujú ručnú anotáciu")
    print(f"    -> Preskočené          : {stats['skipped']}")
    print(f"\n  ĎALŠÍ KROK:")
    print(f"  Otvor {output} a pre každý záznam s true_label = null")
    print(f"  nastav hodnotu AUTOCITATION alebo NOT_AUTOCITATION.")
    print(f"  Po anotácii zmaž pomocné polia začínajúce '_'.")
    print(f"{'=' * 62}\n")


# ---- main ----

def main():
    parser = argparse.ArgumentParser(
        description="Generovanie testovacieho datasetu z CREPČ API"
    )

    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument(
        '--ids', nargs='+', metavar='ID',
        help='Zoznam CREPČ ID hlavných článkov (napr. --ids 123456 789012)'
    )
    id_group.add_argument(
        '--ids-file', metavar='SUBOR',
        help='Textový súbor s CREPČ ID — jedno ID na riadok'
    )
    parser.add_argument(
        '--from-date', default='2000-01-01',
        help='Stiahni citácie od tohto dátumu (YYYY-MM-DD, predvolene: 2000-01-01)'
    )
    parser.add_argument(
        '--output', default='test_dataset.json',
        help='Názov výstupného súboru (predvolene: test_dataset.json)'
    )
    args = parser.parse_args()

    if args.ids:
        main_ids = [str(i) for i in args.ids]
    else:
        with open(args.ids_file, 'r', encoding='utf-8') as f:
            main_ids = [line.strip() for line in f if line.strip()]

    print("=" * 62)
    print("  Generovanie testovacieho datasetu z CREPČ")
    print("=" * 62)
    print(f"  Počet hlavných článkov: {len(main_ids)}")
    print(f"  Citácie od:             {args.from_date}")
    print(f"  Výstupný súbor:         {args.output}")
    print(f"\n  Poznámka: true_label sa priradí automaticky iba pre")
    print(f"  záznamy s HIGH spoľahlivosťou (XML príznak, kľúčové")
    print(f"  slová errát). Ostatné vyžadujú ručnú anotáciu.")

    generate_dataset(
        main_ids  = main_ids,
        from_date = args.from_date,
        output    = args.output,
    )


if __name__ == "__main__":
    main()
