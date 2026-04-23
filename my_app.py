"""
    Main function responsible for orchestrating the entire program, run application and handle the input.

    This function serves as the entry point for the program and
    calls other functions to carry out different tasks with input string.

    Author: Michal Michnak and Damian Husár
    Contact: michalmichnak@gmail.com nad damian.husar999@gmail.com

    LAST UPDATE: 23.4.2026
    U

"""

from flask import Flask, request
from exceptions import ConfigFileNotFoundError, JSONDecodeError, UnhandledInput

import os
import time
import re
import json
import csv
import logging

# Importing custom modules for various extraction functions
from most_used_patterns import most_used_patterns
from load_config import load_config
from convert_month import convert_month
from extract_month_year import extract_month_year
from extract_special_issue import remove_special_issue
from extract_art_no import extract_art_no
from extract_format import extract_format
from extract_volume import extract_volume
from extract_issue import extract_issue
from extract_year import extract_year
from extract_pages import extract_pages
from extract_pages import extract_pages_leftover
from extract_chapter import extract_chapter
from extract_chapter import extract_chapter_leftover
from extract_date import extract_date
from extract_supplement import extract_supplement
from extract_doi import extract_and_remove_doi
from extract_series import extract_and_remove_series
from extract_http_link import extract_http_link
from functions import is_empty, remove_commas, remove_non_alnum, clear_row, replace_roman_numerals, clean_str, find_and_save, replace_patterns

from crepc_api import fetch_main_article, fetch_citations_from_set, fetch_raw_xml_record
from similarity import check_authors_similarity, EMBEDDING_THRESHOLD

logger = logging.getLogger(__name__)

# Creating a Flask web application instance
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.json.sort_keys = False

# Home route to check if the Flask app is running
@app.route('/')
def index():
    return 'Flask app is running, use "POST" method on /process_string route. '

# Route to handle data extraction from an input string
@app.route('/process_string', methods=['POST'])
def parse_row():
    # Get input JSON data from the request
    data = request.get_json()
    row = data.get('inputString', '')
    error = None

    # Preprocess the input string to handle specific characters and symbols
    row = row.replace("–", "-")
    row = row.replace('‐', '-')
    row = row.replace(' a ', ',')
    row = row.replace(" l ", " 1 ")
    row = row.replace(" l,", " 1,")
    row = row.replace(" l.", " 1.")
    row = row.replace(" l]", " 1]")
    row = row.replace("? ", "")
    row = row.replace("#","")
    backup = row  # Create a backup of the original input string

    # Handling most used patterns in a separate function
    result, error, returned = most_used_patterns(row, error)

    # Check if most_used_patterns function found a match
    if returned == "returned":
        # If match found, create a response with extracted data
        response = {
            "data": result,
            "inputString": backup,  # Original input string backup
            "errorCode": error,
            "unhandledInput": None,  # Placeholder for unhandled input (not used here)
        }

        return response, {'Content-Type': 'application/json'}

    # Additional preprocessing steps for specific strings and symbols
    row = row.replace("Číslo článku", "art.no.")
    row = row.replace("neuvedebý", "")
    row = row.replace("neuvedený", "")
    row = row.replace("neuvedené", "")
    row = row.replace("(dvojčíslo)","")
    row = row.replace(u'\u202f'," ")
    row = row.replace("]-", "-")
    row = row.replace("na CD", "[CD-ROM]")
    row = row.replace("a.n.", "art.no.")
    row = row.replace("Dostupné na internete", "(online)")
    row = row.replace("dostupné na internete", "(online)")
    row = row.replace("Nno", "No")
    if row.startswith("ol."):
        row = row.replace("ol.", "vol.", 1)


    # Extracting various elements step by step
    row, link = extract_http_link(row)
    row, chapter = extract_chapter(row)
    row, format, error = extract_format(row, error)
    row, supplement  = extract_supplement(row)
    row, doi = extract_and_remove_doi(row)
    row, series = extract_and_remove_series(row)



    row = row.replace("iv-v","IV-V")

    # Replace Roman numerals with corresponding Arabic numerals
    row = replace_roman_numerals(row)

    # Additional string replacements and formatting
    row = row.replace(" a "," , ")
    row = row.replace("iss.","no.")
    row = row.replace("s.", "p.")
    row = row.replace("p. [", "p.[")
    row = row.replace("Roč ", "Roč. ")
    row = row.replace("Č ", "č. ")

    # CHANGE TO LOWERCASE
    row = row.lower()

    # REMOVAL OF SPACES
    row = row.replace(" ", "")

    # Remove special issues
    row, Is_special_issue, error = remove_special_issue(row, error)

    # Replace specific patterns in the row
    row = replace_patterns(row)

    # Extract date information from the row
    row, date, error = extract_date(row, error)

    # Remove "[cit.]" from the row
    row = row.replace("[cit.]", "")

    # Initialize variables for year and month
    year = None
    month = None

    # Extract year information from the row
    row, year = extract_year(row, year)

    # Extract month and year information from the row
    row, month, year, error = extract_month_year(row, year, error)

    # Clear unnecessary characters from the row
    row = clear_row(row)

    # Extract article number from the row
    row, Articel_number, error = extract_art_no(row, error)

    # Convert month names to numeric values
    row, error = convert_month(row, error)

    # Extract volume information from the row
    row, volume, error = extract_volume(row, error)

    # Additional replacements and formatting
    row = row.replace("no.no.", "no.")
    row = row.replace("no.number", "no.")
    row = row.replace(".:", ".")

    # Extract issue information from the row
    row, issue, error = extract_issue(row, error)

    # Extract page information from the row
    row, page, start_page, end_page, error = extract_pages(row, error)

    # Clear unnecessary characters from the row
    row = clear_row(row)

    # Extract additional page information if no data was extracted in the previous step
    if not page and not start_page and not end_page:
        row, page, start_page, end_page = extract_pages_leftover(row, page, start_page, end_page)

    # Remove non-alphanumeric characters from the row
    row = remove_non_alnum(row)

    # Remove commas from the row
    row = remove_commas(row)

    # Extract leftover chapter information from the row
    row, chapter = extract_chapter_leftover(row, chapter)

    # Find and save remaining information in the row in input format without removed space
    row, info = find_and_save(row, backup)

    if info == "" or info == "no":
        info = None

    # Construct a dictionary containing extracted data
    data = {
        "volume": volume,
        "issue": issue,
        "month": month,
        "year": year,
        "isSpecialIssue": Is_special_issue,
        "articelNumber": Articel_number,
        "series": series,
        "format": format,
        "link": link,
        "page": page,
        "startPage": start_page,
        "endPage": end_page,
        "chapter": chapter,
        "doi": doi,
        "date": date,
        "supplement": supplement,
        "info": info
    }

    # Create a filtered data dictionary containing only non-null values
    filtered_data = {key: value for key, value in data.items() if value is not None}

    # Load excluded values from the configuration file
    config_filename = "configuration\excluded_values.json"
    config, error = load_config(config_filename, error, error_number=1100)

    excluded_values = []
    if config is not None:
        excluded_values = config.get('excluded_values', [])

    row = None if row == "" else row

    if error is not None:
        error_number_returned = error.error_number
    else:
        error_number_returned = None

    try:
        # Check if row is not empty and not in the list of excluded values
        if not is_empty(row) and row not in excluded_values:
            # Raise UnhandledInput exception
            raise UnhandledInput(backup, row, error_number=10001)
        elif row in excluded_values:
            row = None

    except UnhandledInput as e:
        # Handle UnhandledInput exception
        if error_number_returned is None:
            error_number_returned = e.error_number

    # Construct the response dictionary
    response = {
        "data": filtered_data,
        "inputString": backup,
        "errorCode": error_number_returned,
        "unhandledInput": row,
    }

    # Return the response along with the specified content type
    return response, {'Content-Type': 'application/json'}


###------------------------------------###
###                                    ###
### CREPC API related code starts here ###
###                                    ###
###------------------------------------###

# Výsledky behu sú uložené do centrálneho JSON logu prostredníctvom priradenia
# nového záznamu s časovou pečiatkou k existujúcemu súboru. V prípade, že súbor
# neexistuje alebo je poškodený, je inicializované prázdne pole.
def append_to_master_json(new_run_data, filename="start_identification.json"):

    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    run_entry = {
        "api_call_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processed_articles": new_run_data
    }

    data.append(run_entry)

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Výsledky úspešne zapísané do centrálneho logu: {filename}")
    except Exception as e:
        logger.error(f"Nepodarilo sa zapísať do {filename}: {e}")

# Endpoint pre detekciu autocitácií.
# Na vstupe prijíma jedno alebo viac identifikátorov CREPČ (oddelených ľubovoľným oddeľovačom) a voliteľný filter dátumu.
# Pre každé ID je stiahnutý hlavný článok a jeho citácie.
# Errata záznamy sú filtrované už v rámci fetch_citations_from_set. 
# Každá citácia je následne klasifikovaná ako ERRATUM, AUTOCITATION alebo NOT AUTOCITATION.
# Výsledky sú zapísané do logu a vrátené ako kompletná JSON odpoveď.
@app.route('/process_crepc', methods=['POST'])
def process_crepc():
    # 1. Získanie dát a parametrov zo žiadosti.
    data = request.get_json() if request.is_json else {}
    raw_input = str(data.get("crepccids", ""))
    search_from = data.get("fromDate", "2025-01-01")

    ids_found = re.findall(r"(\d{4,})", raw_input)
    if not ids_found:
        return {"errorCode": "Nezadali ste platné ID."}, 400

    all_analyses = []

    for user_id in ids_found:
        # 2. Získanie metadát hlavného článku z CREPČ API.
        main_doc = fetch_main_article(user_id)
        if not main_doc or not main_doc.get("authors"):
            all_analyses.append({"mainArticleId": user_id, "errorCode": "Záznam nenájdený alebo bez autorov"})
            continue

        main_authors = main_doc["authors"]
        main_title = main_doc.get("title", "")

        # 3. Získanie zoznamu citácií; detekcia errát prostredníctvom sentence embeddings
        # prebieha interne v rámci funkcie fetch_citations_from_set.
        citations_found = fetch_citations_from_set(user_id, main_title=main_title, from_date=search_from)

        results = []
        autocitation_count = 0

        for citation in citations_found:
            # Záznamy identifikované ako errata sú vyčlenené so špeciálnym statusom a nie sú podrobené analýze autorskej zhody.
            if citation.get("is_errata"):
                results.append({
                    "id": citation["id"],
                    "status": "ERRATUM",
                    "citing_title": citation.get("citing_title", ""),
                    "note": "Identifikované ako oprava na základe embedding podobnosti názvu s originálom"
                })
                continue

            # 4. Porovnanie autorov algoritmom DL/ID — výstupom je skóre a zoznam všetkých identifikovaných zhôd.
            score, matches_list = check_authors_similarity(main_authors, citation['autori'])

            # Autocitácia je detekovaná pri dosiahnutí prahového skóre DL/ID alebo na základe príznaku autocitácie priamo v XML zázname CREPČ.
            is_match = (score >= EMBEDDING_THRESHOLD) or citation.get("is_xml_autocitation", False)

            if is_match:
                autocitation_count += 1
                status = "AUTOCITATION"
            else:
                status = "NOT AUTOCITATION"

            results.append({
                "id": citation['id'],
                "status": status,
                "citing_title": citation.get("citing_title", ""),
                "score": round(score, 4),
                "matched_authors": matches_list,
                "all_citation_authors": [a["name"] for a in citation["autori"]]
            })

        all_analyses.append({
            "mainArticleId": user_id,
            "mainArticleTitle": main_title,
            "totalProcessed": len(citations_found),
            "autocitationsFound": autocitation_count,
            "details": results
        })

    append_to_master_json(all_analyses)
    return {"results": all_analyses}, 200

# Verzia endpointu, ktorá je využitá pri RPA procese /process_crepc určená na zobrazenie relevantných výsledkov. 
# JSON odpoveď obsahuje výhradne záznamy klasifikované ako AUTOCITATION a ERRATUM — záznamy NOT AUTOCITATION sú z nej vylúčené.
# CSV export naproti tomu zahŕňa všetky záznamy vrátane NOT AUTOCITATION, čo umožňuje úplný prehľad spracovania. 
# Zápis do centrálneho logu nie je realizovaný, pretože endpoint je určený na exploratívne použitie.
@app.route('/process_crepc_only_autocitation', methods=['POST'])
def process_crepc_only_autocitation():
    data = request.get_json() if request.is_json else {"crepccids": request.get_data(as_text=True)}
    raw_input = str(data.get("crepccids", ""))
    search_from = data.get("fromDate", "2025-01-01")

    ids_found = re.findall(r"(\d{4,})", raw_input)
    if not ids_found:
        return {"errorCode": "Nezadali ste platné ID."}, 400

    all_results = []

    csv_rows = []

    for user_id in ids_found:
        main_doc = fetch_main_article(user_id)
        if not main_doc or not main_doc.get("authors"):
            continue

        main_authors = main_doc["authors"]
        main_title   = main_doc.get("title", "")
        # Mená autorov hlavného článku sú zlúčené do jedného reťazca pre stĺpec CSV exportu.
        main_authors_str = "; ".join(a["name"] for a in main_authors)

        # Získanie všetkých potenciálnych citácií pre daný záznam.
        citations_to_check = fetch_citations_from_set(user_id, main_title=main_title, from_date=search_from)

        detected_entries = []

        for citation in citations_to_check:
            citing_title   = citation.get("citing_title", "")
            citing_authors = citation.get("autori", [])
            citing_auth_str = "; ".join(a["name"] for a in citing_authors)

            # Záznamy identifikované ako errata sú zaradené s príslušnou poznámkou.
            if citation.get("is_errata"):
                detected_entries.append({
                    "id":           citation['id'],
                    "type":         "ERRATUM",
                    "citing_title": citing_title
                })
                csv_rows.append({
                    "mainArticleId":      user_id,
                    "mainArticleTitle":   main_title,
                    "mainArticleAuthors": main_authors_str,
                    "citingId":           citation['id'],
                    "type":               "ERRATUM",
                    "citingTitle":        citing_title,
                    "citingAuthors":      citing_auth_str,
                    "matches":            "",
                    "score":              ""
                })
                continue

            if citation.get("is_xml_autocitation", False):
                continue

            # Porovnanie autorov algoritmom DL/ID.
            score, matches_list = check_authors_similarity(main_authors, citing_authors)
            # matches_list je zoznam reťazcov vo formáte „Meno1 <-> Meno2 (score: 0.9, method: DL)".
            is_xml_autocit = citation.get("is_xml_autocitation", False)
            if matches_list:
                matches_str = "; ".join(matches_list)
            elif is_xml_autocit:
                # Príznak autocitácie je prevzatý priamo z XML záznamu CREPČ — zhoda autorov
                # algoritmom nebola identifikovaná, avšak systém CREPČ záznam takto označil.
                matches_str = "XML príznak autocitácie z CREPČ"
            else:
                matches_str = ""

            if score >= EMBEDDING_THRESHOLD or is_xml_autocit:
                entry_type = "AUTOCITATION"
                detected_entries.append({
                    "id":           citation['id'],
                    "type":         entry_type,
                    "citing_title": citing_title,
                    "matches":      matches_list
                })
            else:
                entry_type = "NOT AUTOCITATION"

            csv_rows.append({
                "mainArticleId":      user_id,
                "mainArticleTitle":   main_title,
                "mainArticleAuthors": main_authors_str,
                "citingId":           citation['id'],
                "type":               entry_type,
                "citingTitle":        citing_title,
                "citingAuthors":      citing_auth_str,
                "matches":            matches_str,
                "score":              round(score, 4)
            })

        all_results.append({
            "mainArticleId":      user_id,
            "mainArticleTitle":   main_title,
            "mainArticleAuthors": [a["name"] for a in main_authors],
            "totalProcessedByAI": len(citations_to_check),
            "autocitationsCount": len(detected_entries),
            "details":            detected_entries
        })

    # Pôvodný CSV export pre Blue Prism obsahuje výhradne záznamy typu AUTOCITATION so stĺpcami mainArticleId a citingId. 
    # Nástroj Blue Prism tieto identifikátory využíva na navigáciu na hlavný článok a identifikáciu riadku v tabuľke citácií,
    # kde je potrebné potvrdiť autocitáciu.
    bp_filename = "autocitation_blueprism.csv"
    bp_rows = [
        {
            "mainArticleId":   r["mainArticleId"],
            "mainArticleTitle": r["mainArticleTitle"],
            "citingId":        r["citingId"],
            "citingTitle":     r["citingTitle"],
            "matches":         r["matches"]
        }
        for r in csv_rows if r["type"] == "AUTOCITATION"
    ]
    with open(bp_filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mainArticleId", "mainArticleTitle", "citingId", "citingTitle", "matches"
        ])
        writer.writeheader()
        if bp_rows:
            writer.writerows(bp_rows)

    logger.info(f"Blue Prism CSV: {bp_filename} ({len(bp_rows)} autocitácií na spracovanie)")

    response = {
        "total_requests":              len(ids_found),
        "blueprism_export":            bp_filename,
        "autocitations_for_blueprism": len(bp_rows),
        "results":                     all_results
    }

    return response, {'Content-Type': 'application/json'}

# Endpoint určený na stiahnutie a export surových XML záznamov vo formáte JSON.
# Zámerom je poskytnúť nástroj na ladenie a exploráciu štruktúry dát z CREPČ — záznamy nie sú triedené ani klasifikované, len stiahnuté a uložené do JSON súboru.
@app.route('/process_crepc_raw_export', methods=['POST'])
def process_crepc_raw_export():
    data = request.get_json() if request.is_json else {"crepccids": request.get_data(as_text=True)}
    raw_input = str(data.get("crepccids", ""))
    ids_found = re.findall(r"(\d{4,})", raw_input)

    if not ids_found:
        return {"errorCode": "Nezadali ste platné ID."}, 400

    export_data = []

    for user_id in ids_found:
        # Stiahnutie celého záznamu hlavného článku.
        main_record = fetch_raw_xml_record(user_id)

        # Získanie zoznamu identifikátorov citácií pre daný set prostredníctvom štandardnej logiky načítania citácií.
        citations_to_fetch = fetch_citations_from_set(user_id)

        citations_raw = []
        for cit in citations_to_fetch:
            # Stiahnutie celého záznamu pre každú citáciu.
            raw_cit = fetch_raw_xml_record(cit['id'])
            citations_raw.append({
                "citation_id": cit['id'],
                "full_record": raw_cit
            })

        # Zostavenie exportnej štruktúry pre daný článok.
        article_export = {
            "main_article": {
                "id": user_id,
                "full_record": main_record
            },
            "citations": citations_raw
        }
        export_data.append(article_export)

    # Zápis exportovaných dát do JSON súboru.
    filename = "crepc_raw_export.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=4)

    return {
        "status": "Export dokončený",
        "file_saved": filename,
        "data": export_data
    }, 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
