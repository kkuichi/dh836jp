# Identifikácia autocitácií v databáze CREPC
Tento repozitár obsahuje zdrojový kód k práci zameranej na automatickú identifikáciu autocitácií vo vedeckých publikáciách evidovaných v systéme CREPC (Centrálny register publikačnej činnosti).
Súčasťou riešenia je webová aplikácia, ktorá umožňuje vyhľadávať publikácie autora a zobrazovať detegované autocitácie prostredníctvom prehľadného používateľského rozhrania.

# Charakteristika riešenia
Systém kombinuje prístup k CREPC API na získavanie publikačných záznamov s výpočtom sémantickej podobnosti medzi citáciami. Na základe podobnosti textov sa určuje, či ide o autocitáciu, teda prípad, keď autor cituje svoju vlastnú predchádzajúcu prácu.

# Popis súborov
- crepc_api.py – zabezpečuje komunikáciu s CREPC API. Umožňuje vyhľadávať publikácie podľa autora a sťahovať ich metadáta vrátane zoznamu citácií.
- generate_dataset.py – spracúva získané dáta a vytvára dataset pre testovanie a vyhodnotenie. Zahŕňa čistenie textu, párovanie citácií a rozdelenie na trénovaciu a testovaciu množinu.
- similarity.py – počíta sémantickú podobnosť medzi pármi textov pomocou jazykových embeddingov a kosinusovej podobnosti. Na základe zvoleného prahu rozhoduje, či ide o autocitáciu.
- evaluate.py – vyhodnocuje výkon systému pomocou metrík presnosti, úplnosti a F1-skóre na označenej testovacej množine.
- my_app.py – webová aplikácia umožňujúca zadať meno autora, prehľadávať jeho publikácie a zobraziť detegované autocitácie.
- crepc v1.3.bprelease – release balík Blue Prism obsahujúci RPA proces Crepc Autocitation Process a všetky súvisiace objekty. Importuje sa priamo do platformy Blue Prism a zabezpečuje automatizovaný zápis autocitácií do portálu CREPC.

# Systémové požiadavky
Na spustenie je potrebné mať nainštalovaný Python 3.9 alebo novší a všetky knižnice uvedené v súbore requirements.txt.

# Spustenie aplikácie
1. Naklonovanie repozitára `git clone <repo-url>`
`cd <repo>`
3. Inštalácia závislostí
`pip install -r requirements.txt`
4. Spustenie aplikácie
`my_app.py`
Po spustení je aplikácia dostupná na adrese `http://127.0.0.1:5000`

# Dáta
Použité dáta pochádzajú z verejne dostupného rozhrania CREPC. Z dôvodu objemu nie sú surové dáta súčasťou repozitára. Dataset je možné vygenerovať spustením skriptu generate_dataset.py.

# RPA proces - automatizovaný zápis autocitácií do CREPC
RPA proces je implementovaný v platforme Blue Prism. Súbor crepc v1.3.bprelease obsahuje kompletný release balík, ktorý je potrebné importovať do Blue Prism pred prvým spustením.

# Požiadavky
- Blue Prism (nainštalovaný a nakonfigurovaný)
- Spustená REST API aplikácia (my_app.py) dostupná na http://127.0.0.1:5000
- Platné prihlasovacie údaje do portálu CREPC uložené v Blue Prism Credentials Store

# Import
1. Otvor Blue Prism
2. V menu vyber File → Import
3. Vyber súbor crepc v1.3.bprelease
4. Potvrď import – do systému sa načítajú všetky objekty: Crepc Autocitation Process, Crepc Autocitation, Crepc Epc, Crepc Epc - Basic Actions, Utility - http
   
# Spustenie procesu
Proces Crepc Autocitation Process je plne automatizovaný – na spustenie stačí zadať vstupné parametre crepcids a fromDate

# Postup spustenia
1. V Blue Prism otvor Control Room
2. Vyber proces Crepc Autocitation Process
3. Zadaj hodnotu parametra crepccids (CREPC ID záznamu, ktorý chceš spracovať)
4. Voliteľne zadaj fromDate pre obmedzenie rozsahu citácií
5. Spusti proces tlačidlom Run
