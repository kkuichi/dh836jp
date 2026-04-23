# crepc_api.py
# Komunikácia s databázou CREPC prostredníctvom protokolu OAI-PMH — sťahovanie záznamov a citácií.
# Endpoint GetRecord využíva rozhranie https://app.crepc.sk/oai.
# Endpoint ListRecords (citácie) využíva https://app.crepc.sk/oai/biblioCitations — experimentálne overený.
# Damián Husár, 2026

import requests
import xml.etree.ElementTree as ET
import logging
import re
from similarity import check_text_similarity, TITLE_SIMILARITY_THRESHOLD, ERRATA_KEYWORDS

CREPC_GET_RECORD_URL    = "https://app.crepc.sk/oai"
CREPC_LIST_RECORDS_URL  = "https://app.crepc.sk/oai/biblioCitations"

logger = logging.getLogger(__name__)


# ---- Pomocné XML funkcie ----

# Odstraňuje namespace z názvu XML tagu, napr. {http://...}title -> title.
def strip_ns(tag: str) -> str:
    return tag.split('}')[-1] if '}' in tag else tag

# Vyhľadá prvý priamy potomok elementu s daným lokálnym názvom tagu a vráti jeho textový obsah.
def _get_element_text(element, local_tag: str) -> str:
    for child in element:
        if strip_ns(child.tag) == local_tag:
            return (child.text or "").strip()
    return ""

# Realizuje HTTP GET požiadavku s opakovaním — v prípade neúspešnej odpovede sú vykonané
# najviac dva pokusy. Pri zlyhaní je zaznamenané varovanie do logu; funkcia vracia None.
def _safe_get(url: str, params: dict = None, timeout: int = 20) -> requests.Response | None:
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            logger.warning(f"HTTP {response.status_code} pre {url} (pokus {attempt + 1})")
        except requests.RequestException as e:
            logger.warning(f"Sieťová chyba pre {url} (pokus {attempt + 1}): {e}")
    return None

# Zostavuje index osôb z XML stromu vo forme slovníka {person_id: {"name": ..., "id": ...}}.
# V prípade viacerých záznamov pre tú istú osobu je uprednostnený záznam s dlhším menom,
# čím sa minimalizuje strata informácií spôsobená neúplnými zápismi.
def _build_person_index(root) -> dict:

    index = {}
    for elem in root.iter():
        if strip_ns(elem.tag) != 'rec_person':
            continue
        person_id = elem.get('id', '').strip()
        if not person_id:
            continue

        lname     = ""
        fname     = ""
        labelname = ""

        for child in elem:
            local = strip_ns(child.tag)
            if local == 'lastname':
                lname  = (child.text or "").strip()
            elif local == 'firstname':
                fname = (child.text or "").strip()
            elif local == 'labelname':
                labelname = (child.text or "").strip()

        if lname:
            full_name = f"{lname} {fname}".strip()
        elif labelname:
            full_name = labelname
        else:
            continue

        if person_id not in index or len(full_name) > len(index[person_id]["name"]):
            index[person_id] = {"name": full_name, "id": person_id}

    return index

# Extrahuje zoznam autorov z XML záznamu CREPC vo forme [{"name": ..., "id": ...}, ...].
# Identifikátor osoby (person ID) je kľúčový pre modul similarity.py, kde umožňuje
# priame porovnanie totožnosti autora bez nutnosti výpočtu podobnosti embeddingovým modelom.
def parse_authors_from_xml(root) -> list:
    person_index  = _build_person_index(root)

    main_rec_biblio = None
    for elem in root.iter():
        if strip_ns(elem.tag) == 'rec_biblio':
            main_rec_biblio = elem
            break

    if main_rec_biblio is None:
        return []

    authors_list = []
    seen_ids     = set()
    seen_names   = set()

    # Prechádza elementy cross_biblio_person; spracované sú výhradne záznamy
    # s rolou author alebo author_corporation.
    for child in main_rec_biblio:
        if strip_ns(child.tag) != 'cross_biblio_person':
            continue
        role = child.get('role', '')
        if role not in ('author', 'author_corporation'):
            continue

        lname     = ""
        fname     = ""
        labelname = ""
        person_id = ""

        for subchild in child:
            local = strip_ns(subchild.tag)
            if local == 'rec_person':
                person_id = subchild.get('id', '').strip()
                for grandchild in subchild:
                    gl = strip_ns(grandchild.tag)
                    if gl == 'lastname':
                        lname     = (grandchild.text or "").strip()
                    elif gl == 'firstname':
                        fname     = (grandchild.text or "").strip()
                    elif gl == 'labelname':
                        labelname = (grandchild.text or "").strip()

        # Priamo vložený záznam má prednosť pred hodnotou získanou z indexu osôb.
        if lname:
            full_name = f"{lname} {fname}".strip()
        elif labelname:
            full_name = labelname
        elif person_id and person_id in person_index:
            full_name = person_index[person_id]["name"]
        else:
            full_name = ""

        if not full_name:
            continue

        # Deduplikácia prebieha primárne podľa identifikátora osoby;
        # v prípade jeho absencie je ako náhradné kritérium použité meno autora.
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


# ---- Sťahovanie hlavného článku ----

# Stiahne hlavný článok identifikovaný zadaným ID prostredníctvom OAI-PMH požiadavky GetRecord
# a extrahuje názov článku spolu so zoznamom autorov. Pri sieťovej chybe alebo chybe
# XML spracovania je vrátená hodnota None.
def fetch_main_article(record_id: str) -> dict | None:
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
        logger.error(f"Chyba parsovanie XML pre ID {record_id}: {e}")
        return None

    title_elem = root.find('.//{*}title')
    title      = (title_elem.text or "").strip() if title_elem is not None else ""
    authors    = parse_authors_from_xml(root)

    return {"authors": authors, "title": title}


# ---- Detekcia erraty ----

# Normalizuje názov záznamu za účelom porovnania — zjednocuje varianty pomlčiek
# na štandardný znak, odstraňuje nadbytočné biele znaky a prevedie text na malé písmená.
def normalize_title(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-')
    return re.sub(r'\s+', ' ', text).strip().lower()

# Určuje, či citujúci článok predstavuje erratu — overenie prebieha v troch krokoch
# zoradených podľa výpočtovej náročnosti:
# 1. Kontrola kľúčových slov (erratum, corrigendum, oprava...) — najrýchlejšia metóda.
# 2. Presná zhoda normalizovaných názvov.
# 3. Sémantická podobnosť prostredníctvom embeddingového modelu (check_text_similarity)
#    — najnáročnejšia metóda, vyvolávaná len ak predchádzajúce kroky zhodu nepotvrdili.
def is_errata_by_title_ai(citing_title: str, main_title: str) -> bool:

    if not citing_title or not main_title:
        return False

    c_norm = normalize_title(citing_title)
    m_norm = normalize_title(main_title)

    # 1. Kontrola kľúčových slov charakteristických pre erratu.
    if set(c_norm.split()) & ERRATA_KEYWORDS:
        logger.debug(f"Errata kľúčové slovo: '{citing_title}'")
        return True

    # 2. Presná zhoda normalizovaných názvov.
    if c_norm == m_norm:
        return True

    # 3. Sémantická podobnosť prostredníctvom embeddingového modelu.
    score = check_text_similarity(c_norm, m_norm)
    logger.debug(f"Errata embedding skóre: {score:.3f} | '{citing_title}' vs '{main_title}'")
    return score >= TITLE_SIMILARITY_THRESHOLD


# ---- XML príznak autocitácie ----

# Overuje, či záznam obsahuje XML príznak autocitácie voči zadanému cieľovému ID.
# Vyhľadávané sú elementy cross_biblio_biblio s hodnotou autocitation=true
# a zhodným identifikátorom rec_biblio.
def _check_xml_autocitation(record, target_id_str: str) -> bool:
    main_rec_biblio = record.find('.//{*}rec_biblio')
    if main_rec_biblio is None:
        return False

    for bond in main_rec_biblio:
        if strip_ns(bond.tag) != 'cross_biblio_biblio':
            continue
        autocit_val = _get_element_text(bond, 'autocitation')
        if autocit_val.lower() != 'true':
            continue
        for child in bond:
            if strip_ns(child.tag) == 'rec_biblio':
                if child.get('id', '').strip() == target_id_str:
                    return True

    return False


# ---- Sťahovanie citácií zo setu ----

# Stiahne všetky citácie zo setu identifikovaného zadaným ID prostredníctvom OAI-PMH
# požiadavky ListRecords. Pre každý nájdený záznam je určené, či ide o erratu,
# či je prítomný XML príznak autocitácie, a extrahovaný je zoznam autorov.
# Výstupom je zoznam slovníkov s uvedenými atribútmi.
def fetch_citations_from_set(
    target_id: str,
    main_title: str = "",
    from_date: str  = '2025-01-01'
) -> list:

    params = {
        'verb'           : 'ListRecords',
        'metadataPrefix' : 'xml-crepc2-flat4',
        'set'            : target_id,
        'from'           : from_date
    }

    response = _safe_get(CREPC_LIST_RECORDS_URL, params=params, timeout=30)
    if response is None:
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        logger.error(f"Chyba parsovanie XML pre set {target_id}: {e}")
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
        citing_id = identifier_elem.text.split('/')[-1]

        citing_title_text = ""
        rec_biblio = record.find('.//{*}rec_biblio')
        if rec_biblio is not None:
            citing_title_text = _get_element_text(rec_biblio, 'title')

        is_errata           = is_errata_by_title_ai(citing_title_text, main_title)
        is_xml_autocitation = _check_xml_autocitation(record, target_id_str)
        authors_list        = parse_authors_from_xml(record)

        found_citations.append({
            "id"                  : citing_id,
            "autori"              : authors_list,
            "citing_title"        : citing_title_text,
            "is_errata"           : is_errata,
            "is_xml_autocitation" : is_xml_autocitation,
        })

    return found_citations


# ---- Surový export záznamu ----

# Rekurzívne prevedie XML element na slovník, kde kľúčom je názov tagu
# a hodnotou sú textový obsah, atribúty alebo potomkovia elementu.
# V prípade viacerých potomkov s rovnakým tagom sú tieto združené do zoznamu.
def xml_to_dict(element) -> dict:
    tag    = strip_ns(element.tag)
    result = {tag: {}}

    if element.attrib:
        result[tag].update(element.attrib)

    text = (element.text or "").strip()
    if text:
        if not element.attrib:
            result[tag] = text
        else:
            result[tag]['value'] = text

    for child in list(element):
        child_dict = xml_to_dict(child)
        child_tag  = list(child_dict.keys())[0]

        if child_tag not in result[tag]:
            result[tag][child_tag] = child_dict[child_tag]
        else:
            if not isinstance(result[tag][child_tag], list):
                result[tag][child_tag] = [result[tag][child_tag]]
            result[tag][child_tag].append(child_dict[child_tag])

    return result

# Stiahne úplný XML záznam z CREPC a prevedie ho na slovník prostredníctvom xml_to_dict.
# Funkcia slúži primárne pri ladení štruktúry dát v rámci endpointu /process_crepc_raw_export.
def fetch_raw_xml_record(record_id: str) -> dict:
    url = (
        f"{CREPC_GET_RECORD_URL}?verb=GetRecord"
        f"&metadataPrefix=xml-crepc2-flat4"
        f"&identifier=oai:crepc.sk:biblio/{record_id}"
    )
    response = _safe_get(url, timeout=15)
    if response is None:
        return {"error": "Záznam nedostupný"}

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        return {"error": f"Chyba XML: {e}"}

    rec_biblio = root.find('.//{*}rec_biblio')
    if rec_biblio is not None:
        return xml_to_dict(rec_biblio).get('rec_biblio', {})
    return {"error": "Záznam nenájdený"}
