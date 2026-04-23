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
