# %% [markdown]
# # Importovanie knižnic

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,roc_curve,auc,ConfusionMatrixDisplay
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN,SMOTE
from sklearn.naive_bayes import GaussianNB  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


%matplotlib inline
sns.set()
pd.set_option ('display.max_columns', 50)

# %% [markdown]
# # Priprava datasetu

# %%
#Načítanie dát z csv súboru do datasetu data, následne premenovanie stĺpcov a vytvorenie nového csv súboru v ktorom budú už premenované stlpce atribútov.
data = pd.read_csv('bank-additional-full.csv')
data.columns = ['Columns']
split = data["Columns"].str.split(pat=';',expand=True)
split.to_csv('bank_4.csv')
bank = pd.read_csv('bank_4.csv')
bank.drop('Unnamed: 0',inplace=True, axis=1)
bank = bank.rename(columns = {'0':'vek',
                              '1':'povolanie',    
                              '2':'r_stav',
                              '3':'vzdelanie',
                              '4':'podlžnosti',
                              '5' : 'uver_byvanie', 
                              '6' : 'osobny_uver',
                              '7' : 'typ_kontaktu', 
                              '8' : 'mesiac',
                              '9' : 'den', 
                              '10' : 'dlžka_kontaktu',
                              '11' : 'počet_kontaktov', 
                              '12' : 'dni_posledny_kontakt',
                              '13' : 'predchadzajuce_kontakty', 
                              '14' : 'vysledok_pred_kampan', 
                              '15' : 'zmena_zamestnanosti', 
                              '16' : 'idx_cien_spotrebitelov',
                              '17' : 'idx_spotrebitel_dovery', 
                              '18' : 'euriborn3m', 
                              '19' : 'počet_zamestnancov', 
                              '20' : 'y'})


#Vyvorenie dvoch nových súborov, ktoré v akutalnom stave budú obsahovať rovnaké dáta. Tieto datasety budú slúžiť na vytvorenie dvoch rozdielnych datasetov a to jeden, ktorý bude mať podstránené určité dáta a druhý, ktorý tie dáta mať odstránené nebude.
bank.to_csv('final.csv')
bank.to_csv('final2.csv')
bank2 = pd.read_csv('final2.csv')
bank2.drop('Unnamed: 0',inplace=True, axis=1)
# pomocou funkcii .drop sme v datasete bank a bank2 odstránili stĺpec Unnamed: 9 ktorý nám vznikol po načpitaní datasetu.

# %%
#pomocou tohot príkazu sme si zobrazili časť dát potrebnú pre kontrolu či sú dáta správne rozdelené
bank.head()

# %% [markdown]
# # Zistovanie početnosti neznamych v datasete
# V tejto časti sme si pomocou print funkcie skontrolovali aké hodnoty nadobúdajú a zistili že atribút podlžnosti obsahuje veľke množstvo unknown hodnot, tento atribut neskor odstránime
# 

# %%
print(bank['povolanie'].value_counts())

# %%
print(bank['r_stav'].value_counts())


# %%
print(bank['vzdelanie'].value_counts())


# %%
print(bank['podlžnosti'].value_counts())


# %%
print(bank['uver_byvanie'].value_counts())


# %%
print(bank['osobny_uver'].value_counts())


# %%
print(bank['typ_kontaktu'].value_counts())


# %%
print(bank['dni_posledny_kontakt'].value_counts())


# %%
print(bank['y'].value_counts())


# %% [markdown]
# # Ploty atributov

# %%
#v tomto bloku sme si pomocou subplotu zobrazili grafy početnosti jednotlivých kategoriálnych atribútov v histogramoch.
fig, axs = plt.subplots(3, 3, constrained_layout=True, figsize=(10, 7))

sns.set_style("whitegrid")
sns.histplot(data=bank, y="vysledok_pred_kampan", ax=axs[0, 0])
axs[0, 0].set_title("Výsledok predchádzajúcej kampane")
axs[0,0].set_xlabel("Počet zákazníkov")

sns.histplot(data=bank, y="r_stav", ax=axs[0, 1])
axs[0, 1].set_title("Rodinný stav zákazníkov")
axs[0,1].set_xlabel("Počet zákazníkov")

sns.histplot(data=bank, y="den", ax=axs[0, 2])
axs[0, 2].set_title("Deň kontaktovania")
axs[0,2].set_xlabel("Počet zákazníkov")

sns.histplot(data=bank, y="podlžnosti", ax=axs[1, 0])
axs[1, 0].set_title("Splácanie iných záväzkov")
axs[1,0].set_xlabel("Počet zákazníkov")

sns.histplot(data=bank, y="uver_byvanie", ax=axs[1, 1])
axs[1, 1].set_title("Úver na bývanie")
axs[1,1].set_xlabel("Počet zákazníkov")

sns.histplot(data=bank, y="osobny_uver", ax=axs[1, 2])
axs[1, 2].set_title("Iné osobné pôžičky")
axs[1,2].set_xlabel("Počet zákazníkov")

sns.histplot(bank,y="povolanie",hue="y", ax=axs[2,0])
axs[2,0].set_title("Porovnanie povolania voči cielovému atrivútu")
axs[2,0].set_xlabel("Počet zákazníkov")

sns.histplot(bank,y="mesiac",ax=axs[2,1])
axs[2, 1].set_title("Mesiac kontaktu zákazníka")
axs[2,1].set_xlabel("Počet zákazníkov")

sns.histplot(bank,y="vzdelanie",ax=axs[2,2])
axs[2, 2].set_title("Vzdelanie zákazníka")
axs[2,2].set_xlabel("Počet zákazníkov")

plt.show()

# %% [markdown]
# # Korelacna tabulka

# %%
#v tomto bloku sme pomocou funkcie corr_table vytvorili korelačnú maticu pre numerické atribúty a následne ju pomocou heat map zobrazili graficky.
numerical_atributes = ['vek','dlžka_kontaktu','počet_kontaktov','dni_posledny_kontakt','predchadzajuce_kontakty','zmena_zamestnanosti','idx_cien_spotrebitelov','idx_spotrebitel_dovery','euriborn3m', 'počet_zamestnancov']
corr_table = bank[numerical_atributes].corr()
corr_table

# %%
p = sns.heatmap(corr_table, 
                xticklabels=corr_table.columns, yticklabels=corr_table.columns,
                vmin=-1,vmax=1,
                cmap='coolwarm',
                square=True)

# %% [markdown]
# # chi-test

# %%
# v následovnom bloku sme vykonali porovnanie kategoriálnych atribútov s cieĺovým atribútom pomocou chi-kvadrát testu a výsledky sme si vypísali.
contingency_table = pd.crosstab(bank['povolanie'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------povolanie-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)

contingency_table = pd.crosstab(bank['r_stav'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------r_stav-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)

contingency_table = pd.crosstab(bank['vzdelanie'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------vzdelanie-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)

contingency_table = pd.crosstab(bank['podlžnosti'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------podlžnosti-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)

contingency_table = pd.crosstab(bank['uver_byvanie'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------uver_byvanie-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)

contingency_table = pd.crosstab(bank['osobny_uver'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------osobny_uver-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)

contingency_table = pd.crosstab(bank['typ_kontaktu'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------typ_kontaktu-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)

contingency_table = pd.crosstab(bank['mesiac'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------mesiac-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)

contingency_table = pd.crosstab(bank['den'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------den-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)

contingency_table = pd.crosstab(bank['vysledok_pred_kampan'], bank['y'])
chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
print("-------------------vysledok_pred_kampan-------------------")
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_val)


# %% [markdown]
# # Odstránenie atributov

# %%
#v tomto bloku sme si definovali si najprv definovali atribúty ktoré chcem odstrániť z prislušnych datasetov tieto atributy su v premennych atributes_to_drop a atribute_to_drop
atributes_to_drop = ['osobny_uver','mesiac','vysledok_pred_kampan','podlžnosti']
atribute_to_drop = ['podlžnosti']
bank = bank.drop(atribute_to_drop,axis='columns')
bank2 = bank2.drop(atributes_to_drop,axis='columns')

#následne sme v datasete hodnoty ktoré boli unknown nahradili hodnotami NA a odstránili pomocou funkcie dropna následne sme vykonali kontrolu či daný dataset ešte obsahuje unknown hodnoty tento postup sme opakovali pre oba datasety
column_names = bank.columns
column_names2 = bank2.columns
print(column_names)
bank[column_names] = bank[column_names].replace('"unknown"',pd.NA)
bank = bank.dropna()
print(bank[column_names].isna().count())
bank[column_names] = bank[column_names].replace('"', '')

print(column_names2)
bank2[column_names2] = bank2[column_names2].replace('"unknown"',pd.NA)
bank2 = bank2.dropna()
print(bank2[column_names2].isna().count())
bank2[column_names2] = bank2[column_names2].replace('"', '')
print(bank2['y'])

# %% [markdown]
# # transformovanie kategorialnzch atributov

# %%
#pre transformáciu kategoriálnych atribútov na numerické sme používali funkciu LabelEncoder. Najprv sme si vytvorili samotný ekóder v premennej ecn a následne sme ho implementovali na naše atribúty v oboch datasetoch
ecn = LabelEncoder()

bank['povolanie'] = ecn.fit_transform(bank['povolanie'])
bank['r_stav'] = ecn.fit_transform(bank['r_stav'])
bank['vzdelanie'] = ecn.fit_transform(bank['vzdelanie'])
bank['typ_kontaktu'] = ecn.fit_transform(bank['typ_kontaktu'])
bank['mesiac'] = ecn.fit_transform(bank['mesiac'])
bank['vysledok_pred_kampan'] = ecn.fit_transform(bank['vysledok_pred_kampan'])
bank['y'] = ecn.fit_transform(bank['y'])
bank['uver_byvanie'] = ecn.fit_transform(bank['uver_byvanie'])
bank['osobny_uver'] = ecn.fit_transform(bank['osobny_uver'])
bank['den'] = ecn.fit_transform(bank['den'])

bank2['povolanie'] = ecn.fit_transform(bank2['povolanie'])
bank2['r_stav'] = ecn.fit_transform(bank2['r_stav'])
bank2['vzdelanie'] = ecn.fit_transform(bank2['vzdelanie'])
bank2['typ_kontaktu'] = ecn.fit_transform(bank2['typ_kontaktu'])
bank2['y'] = ecn.fit_transform(bank2['y'])
bank2['uver_byvanie'] = ecn.fit_transform(bank2['uver_byvanie'])
bank2['den'] = ecn.fit_transform(bank2['den'])

# %% [markdown]
# # Normalizácia atributov

# %%
attributes_to_normalize = ['vek', 'dlžka_kontaktu', 'počet_kontaktov', 'dni_posledny_kontakt', 'predchadzajuce_kontakty', 
                           'zmena_zamestnanosti', 'idx_cien_spotrebitelov', 'idx_spotrebitel_dovery', 'euriborn3m', 'počet_zamestnancov']

# v tomto bloku sme pomocou minmaxscaler funkcie normalizovlai dáta, ktoré sme si na základe toho že dané atribúty boli v rozdielnych jednotkách a a veľkých rozsahoch
scaler = MinMaxScaler()
bank[attributes_to_normalize] = scaler.fit_transform(bank[attributes_to_normalize])
bank2[attributes_to_normalize] = scaler.fit_transform(bank2[attributes_to_normalize])

# %% [markdown]
# # Vytvorenie Train test množiny

# %%
# v tomto bloku sme pomocou funkcie train_test_split vytvorili z našich dvoch datasetov trenovacie a testovacie množiny pre jednotlivé datasety.
x = bank2.drop("y",axis=1)
y = bank2["y"]
train_data1,test_data1,train_label1,test_label1 = train_test_split(x,y,test_size=0.3,stratify=y,random_state=1)

x2 = bank.drop("y",axis=1)
y2 = bank["y"]
train_data2,test_data2,train_label2,test_label2 = train_test_split(x,y,test_size=0.3,stratify=y,random_state=1)

# %% [markdown]
# # ADASYN oversampling

# %%
#v tomto bloku sme vytvárali pre oba naše datasety množiny ktoré boli doplnené o dovzorkované dáta a v tomto prípade a bloku bola použitá samplovacia metóda ADASYN.
#v tejto monžine sme samplovali minoritnú triedu nášho cieľového atribútu s počtom susedov 11, parameter random state sme použili pre stálosť jednotlivých dát. Dané množiny boli v pomere 70-30 v oboch prípadoch
# do premenných x_train_adasyn sme uložili dosamlpované trénovacie dáta, do x_test_adasyn sme uložili dosamplované testovacie dáta a v atribútoch y_train_adasyn a y_test_adasyn  sme uložili trénovacie a testovacei cieľové premenne postup pri vytvarani bol pri oboch datasetoch rovnaký rozidel je iba v názve premenných ku ktorý sme pre rýchlejšie písanie pridali 2 na koniec každého názvu premennej
adasyn = ADASYN(sampling_strategy='minority', random_state=1, n_neighbors=11)
new_data_adasyn, new_classes_adasyn = adasyn.fit_resample(x,y)

new_bank_adasyn = pd.DataFrame(new_data_adasyn, columns=bank2.columns)
new_bank_adasyn['y'] = new_classes_adasyn

x_adasyn = new_bank_adasyn.drop("y",axis=1)
y_adasyn = new_bank_adasyn["y"]
x_train_adasyn,x_test_adasyn,y_train_adasyn,y_test_adasyn = train_test_split(x_adasyn,y_adasyn,test_size=0.3,stratify=y_adasyn,random_state=1)
print("Class distribution after ADASYN:")
print(new_bank_adasyn['y'].value_counts())
print("Original class distribution:")
print(bank2['y'].value_counts())

adasyn2 = ADASYN(sampling_strategy='minority', random_state=1, n_neighbors=11)
new_data_adasyn2, new_classes_adasyn2 = adasyn2.fit_resample(x2,y2)

new_bank_adasyn2 = pd.DataFrame(new_data_adasyn2, columns=bank.columns)
new_bank_adasyn2['y'] = new_classes_adasyn2

x_adasyn2 = new_bank_adasyn2.drop("y",axis=1)
y_adasyn2 = new_bank_adasyn2["y"]
x_train_adasyn2,x_test_adasyn2,y_train_adasyn2,y_test_adasyn2 = train_test_split(x_adasyn2,y_adasyn2,test_size=0.3,stratify=y_adasyn2,random_state=1)
print("Class distribution after ADASYN:")
print(new_bank_adasyn2['y'].value_counts())
print("Original class distribution:")
print(bank['y'].value_counts())

# %% [markdown]
# # SMOTE oversampling

# %%
#v tomto bloku sme vytvárali pre oba naše datasety množiny ktoré boli doplnené o dovzorkované dáta a v tomto prípade a bloku bola použitá samplovacia metóda SMOTE.
#v tejto monžine sme samplovali minoritnú triedu nášho cieľového atribútu, parameter random state sme použili pre stálosť jednotlivých dát. Dané množiny boli v pomere 70-30 v oboch prípadoch
# do premenných x_train_smote sme uložili dosamlpované trénovacie dáta, do x_test_smote sme uložili dosamplované testovacie dáta a v atribútoch y_train_smote a y_test_smote  sme uložili trénovacie a testovacei cieľové premenne postup pri vytvarani bol pri oboch datasetoch rovnaký rozidel je iba v názve premenných ku ktorý sme pre rýchlejšie písanie pridali 2 na koniec každého názvu premennej

smote = SMOTE(sampling_strategy='minority', random_state=1)
new_bank_smote = smote.fit_resample(x, y)
new_data_smote, new_classes_smote = smote.fit_resample(x, y)

new_bank_smote = pd.DataFrame(new_data_smote, columns=bank2.columns)
new_bank_smote['y'] = new_classes_smote

x_smote = new_bank_smote.drop("y", axis=1)
y_smote = new_bank_smote["y"]
x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.3, stratify=y_smote, random_state=1)
print("Class distribution after SMOTE:")
print(new_bank_smote['y'].value_counts())
print("Original class distribution:")
print(bank2['y'].value_counts())




smote2 = SMOTE(sampling_strategy='minority', random_state=1)
new_bank_smote2 = smote.fit_resample(x, y)
new_data_smote2, new_classes_smote2 = smote2.fit_resample(x2, y2)

new_bank_smote2 = pd.DataFrame(new_data_smote2, columns=bank.columns)
new_bank_smote2['y'] = new_classes_smote2

x_smote2 = new_bank_smote2.drop("y", axis=1)
y_smote2 = new_bank_smote2["y"]
x_train_smote2, x_test_smote2, y_train_smote2, y_test_smote2 = train_test_split(x_smote2, y_smote2, test_size=0.3, stratify=y_smote2, random_state=1)
print("Class distribution after SMOTE:")
print(new_bank_smote2['y'].value_counts())
print("Original class distribution:")
print(bank['y'].value_counts())

# %%
x_train_smote2.isna().count()
x_train_smote2.count()
print(x_train_smote2)

# %% [markdown]
# # Bayes

# %%
# vytvorenie modelu a následne trenovanie a vytvorenie novej predikcie
modelBA = GaussianNB()
modelBA.fit(train_data2, train_label2)
resultB = modelBA.predict(test_data2)

# vyber najdoležitejšich 5 atributov jeho zobrazenie
selector = SelectKBest(chi2, k=5)  
selected_train_data = selector.fit_transform(train_data2, train_label2)
selected_indices = selector.get_support(indices=True)
selected_feature_names = [train_data2.columns[i] for i in selected_indices]

plt.figure(figsize=(10, 6))
plt.bar(selected_feature_names, selector.scores_[selected_indices])
plt.xlabel("Atribúty")
plt.ylabel("Chi-kvadrát hodnota")
plt.title("Dôležitosť atribútov (chi-kvadrát test)- Naive Bayes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#zobrazenie matice zmetenosti a reportu klasifikácie
ct_bayes = confusion_matrix(test_label2, resultB)
print(ct_bayes)
print(classification_report(test_label2, resultB,digits=4))

#vytvorenie ROC krivky a vypočítanie AUC hodnoty s následným zorbazením ROC krivky a vypísaním AUC hodnoty
fpraa, tpraa, _ = roc_curve(test_label2, resultB)
roc_aucaa = auc(fpraa, tpraa)

# Display AUC value on the graph
plt.plot(fpraa, tpraa, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucaa))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Naive Bayes - ADASYN')
plt.legend(loc='lower right')

# Show the plot
plt.show()
print(roc_aucaa)


# %%
##Bayes ADASYN                                                      
# Natrénovanie modelu Naive Bayes
modelBB = GaussianNB()
modelBB.fit(x_train_adasyn, y_train_adasyn)

# Výber najdôležitejších atribútov
selectorA = SelectKBest(chi2, k=5)  # Vyberte počet atribútov, ktoré chcete
selected_train_dataA = selectorA.fit_transform(x_train_adasyn, y_train_adasyn)

# Získanie indexov vybraných atribútov
selected_indicesA = selectorA.get_support(indices=True)

# Vypísanie názvov vybraných atribútov
selected_feature_namesA = [x_train_adasyn.columns[i] for i in selected_indicesA]

# Vytvorenie grafu
plt.figure(figsize=(10, 6))
plt.bar(selected_feature_namesA, selectorA.scores_[selected_indicesA])
plt.xlabel("Atribúty")
plt.ylabel("Chi-kvadrát hodnota")
plt.title("Dôležitosť atribútov (chi-kvadrát test)- Naive Bayes- ADASYN")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

resultB = modelBB.predict(x_test_adasyn)                            
                            
ct_bayes_adasyn = confusion_matrix(y_test_adasyn, resultB)
print(ct_bayes_adasyn)                        
print(classification_report(y_test_adasyn, resultB,digits=4)) 

fprab, tprab, _ = roc_curve(y_test_adasyn, resultB)
roc_aucab = auc(fprab, tprab)

# Zobrazenie hodnoty AUC na grafe
plt.plot(fprab, tprab, color='b', label='ROC krivka(AUC = {:.3f})'.format(roc_aucab))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Naive Bayes - ADASYN')
plt.legend(loc='lower right')

# Zobrazenie grafu
plt.show()
print('AUC' + str(roc_aucab))


# %%
##Bayes SMOTE             
# Natrénovanie modelu Naive Bayes
modelBC = GaussianNB()
modelBC.fit(x_train_smote, y_train_smote)

# Výber najdôležitejších atribútov
selectorB = SelectKBest(chi2, k=5)  # Vyberte počet atribútov, ktoré chcete
selected_train_dataB = selectorB.fit_transform(x_train_smote, y_train_smote)

# Získanie indexov vybraných atribútov
selected_indicesB = selectorB.get_support(indices=True)

# Vypísanie názvov vybraných atribútov
selected_feature_namesB = [x_train_smote.columns[i] for i in selected_indicesB]

# Vytvorenie grafu
plt.figure(figsize=(10, 6))
plt.bar(selected_feature_namesB, selectorB.scores_[selected_indicesB])
plt.xlabel("Atribúty")
plt.ylabel("Chi-kvadrát hodnota")
plt.title("Dôležitosť atribútov (chi-kvadrát test)- Naive Bayes- SMOTE")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

resultB = modelBC.predict(x_test_smote)                            
                            
ct_bayes_smote = confusion_matrix(y_test_smote, resultB)
print(ct_bayes_smote)                        
print(classification_report(y_test_smote, resultB,digits=4))

fprac, tprac, _ = roc_curve(y_test_smote, resultB)
roc_aucac = auc(fprac, tprac)

# Zobrazenie hodnoty AUC na grafe
plt.plot(fprac, tprac, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucac))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Naive Bayes - Smote')
plt.legend(loc='lower right')

# Zobrazenie grafu
plt.show()
print('AUC' + str(roc_aucac))



# %%
# Vytvorenie grafu pre model Naive Bayes
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(fpraa, tpraa, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucaa))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Naive Bayes - bez vzorkovania')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model KNN
plt.subplot(132)
plt.plot(fprab, tprab, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucab))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Naive Bayes - ADASYN')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model SVM
plt.subplot(133)
plt.plot(fprac, tprac, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucac))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Naive Bayes - Smote')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# %%
fig,axes = plt.subplots(1,3,figsize=(15,5))

disp_1 = ConfusionMatrixDisplay(ct_bayes)
disp_1.plot(ax=axes[0], cmap="viridis")

disp_2 = ConfusionMatrixDisplay(ct_bayes_adasyn)
disp_2.plot(ax=axes[1], cmap="viridis")

disp_3 = ConfusionMatrixDisplay(ct_bayes_smote)
disp_3.plot(ax=axes[2], cmap="viridis")

fig.suptitle("Matice zámen Naive Bayes")
plt.tight_layout()
plt.show()



# %% [markdown]
# # kNN

# %%
##kNN pred                                            

model1a = KNeighborsClassifier(n_neighbors = 31)                          
fitaa = model1a.fit(train_data2, train_label2)                          
res1 = model1a.predict(test_data2)

ct_knn = confusion_matrix(test_label2, res1) 
print(ct_knn)
print(classification_report(test_label2, res1,digits=4))

# Výpočet ROC krivky
fprba, tprba, threshold = roc_curve(test_label2, res1)
roc_aucba = auc(fprba, tprba)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprba, tprba, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucba))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - kNN - bez vzorkovania')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucba)

# %%
#KNN ADASYN
model1b = KNeighborsClassifier(n_neighbors = 31)
fitab = model1b.fit(x_train_adasyn, y_train_adasyn)                          
res2 = model1b.predict(x_test_adasyn)

ct_knn_adasyn = confusion_matrix(y_test_adasyn, res2)
print(ct_knn_adasyn)  
print(classification_report(y_test_adasyn, res2,digits=4))

# Výpočet ROC krivky
fprbb, tprbb, threshold = roc_curve(y_test_adasyn, res2)
roc_aucbb = auc(fprbb, tprbb)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprbb, tprbb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucbb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - kNN - ADASYN')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucbb)

# %%


# %%
##kNN SMOTE

model1c = KNeighborsClassifier(n_neighbors = 31)
fitac = model1c.fit(x_train_smote, y_train_smote)
res3 = model1c.predict(x_test_smote)

ct_knn_smote = confusion_matrix(y_test_smote, res3)
print(ct_knn_smote)  
print(classification_report(y_test_smote, res3,digits=4))

# Výpočet ROC krivky
fprbc, tprbc, threshold = roc_curve(y_test_smote, res3)
roc_aucbc = auc(fprbc, tprbc)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprbc, tprbc, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucbc))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - kNN - Smote')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucbc)

# %%
# Vytvorenie grafu pre model RF
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(fprba, tprba, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucba))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - kNN - bez vzorkovania')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model KNN
plt.subplot(132)
plt.plot(fprbb, tprbb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucbb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - kNN - ADASYN')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model SVM
plt.subplot(133)
plt.plot(fprbc, tprbc, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucbc))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - kNN - Smote')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# %%
fig,axes = plt.subplots(1,3,figsize=(15,5))

disp_1 = ConfusionMatrixDisplay(ct_knn)
disp_1.plot(ax=axes[0], cmap="viridis", values_format="d")

disp_2 = ConfusionMatrixDisplay(ct_knn_adasyn)
disp_2.plot(ax=axes[1], cmap="viridis")

disp_3 = ConfusionMatrixDisplay(ct_knn_smote)
disp_3.plot(ax=axes[2], cmap="viridis")

fig.suptitle("Matice zámen kNN")
plt.tight_layout()
plt.show()

# %% [markdown]
# # RF

# %%
##Random Forrest pred
# Natrénovanie modelu Random Forest
model_rfA = RandomForestClassifier(random_state=0)
model_rfA.fit(train_data2, train_label2)

# Získanie dôležitosti atribútov
importances_rfA = model_rfA.feature_importances_

# Zoradenie atribútov podľa dôležitosti
sorted_indicesA = importances_rfA.argsort()[::-1]
top_feature_indicesA = sorted_indicesA[:5]
top_feature_namesA = [train_data2.columns[i] for i in top_feature_indicesA]

# Vytvorenie grafu
plt.figure(figsize=(10, 6))
plt.bar(top_feature_namesA, importances_rfA[top_feature_indicesA])
plt.xlabel("Atribúty")
plt.ylabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Random Forest - bez vzorkovania")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

pred = model_rfA.predict(test_data2)

ct_rf = confusion_matrix(test_label2, pred)
print(ct_rf)  
print(classification_report(test_label2, pred,digits=4))

# Výpočet ROC krivky
fprca, tprca, threshold = roc_curve(test_label2, pred)
roc_aucca = auc(fprca, tprca)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprca, tprca, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucca))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Random Forrest - bez nadvzorkovania')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucca)

# %%
##Random Forrest adasyn
# Natrénuj model Random Forest na trénovacích dátach
model_rfB = RandomForestClassifier(random_state=0)
model_rfB.fit(x_train_adasyn, y_train_adasyn)

# Získaj dôležitosť atribútov
importances_rfB = model_rfB.feature_importances_

# Zorad atribúty podľa dôležitosti
sorted_indicesB = importances_rfB.argsort()[::-1]
top_feature_indicesB = sorted_indicesB[:5]
top_feature_namesB = [x_train_adasyn.columns[i] for i in top_feature_indicesB]

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.bar(top_feature_namesB, importances_rfB[top_feature_indicesB])
plt.xlabel("Atribúty")
plt.ylabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Random Forest - ADASYN")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Predikuj na testovacích dátach
pred2 = model_rfB.predict(x_test_adasyn)

# Vytvor kontingenčnú tabuľku
ct_rf_adasyn = confusion_matrix(y_test_adasyn, pred2)
print(ct_rf_adasyn)

# Vypíš klasifikačný report
print(classification_report(y_test_adasyn, pred2,digits=4))

# Výpočet ROC krivky
fprcb, tprcb, threshold = roc_curve(y_test_adasyn, pred2)
roc_auccb = auc(fprcb, tprcb)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprcb, tprcb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_auccb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Random Forrest - ADASYN')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_auccb)

# %%
##Random Forrest smote
# Trenovanie modelu RF na trenovacich datach
model_rfC = RandomForestClassifier(random_state=0)
model_rfC.fit(x_train_smote, y_train_smote)

# Získabnie doležitych atributov
importances_rfC = model_rfC.feature_importances_

# Zorad atribúty podľa dôležitosti
sorted_indicesC = importances_rfC.argsort()[::-1]
top_feature_indicesC = sorted_indicesC[:5]
top_feature_namesC = [x_train_smote.columns[i] for i in top_feature_indicesC]

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.bar(top_feature_namesC, importances_rfC[top_feature_indicesC])
plt.xlabel("Atribúty")
plt.ylabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Random Forest - SMOTE")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Predikuj na testovacích dátach
pred3 = model_rfC.predict(x_test_smote)

# Vytvor kontingenčnú tabuľku
ct_rf_smote = confusion_matrix(y_test_smote, pred3)
print(ct_rf_smote)
print(classification_report(y_test_smote, pred3,digits=4))

# Výpočet ROC krivky
fprcc, tprcc, threshold = roc_curve(y_test_smote, pred3)
roc_auccc = auc(fprcc, tprcc)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprcc, tprcc, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_auccc))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Random Forrest - Smote')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_auccc)

# %%
# Vytvorenie grafu pre model RF
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(fprca, tprca, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucca))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Random Forrest - bez nadvzorkovania')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model KNN
plt.subplot(132)
plt.plot(fprcb, tprcb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_auccb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Random Forrest - ADASYN')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model SVM
plt.subplot(133)
plt.plot(fprcc, tprcc, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_auccc))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Random Forrest - Smote')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# %%
fig,axes = plt.subplots(1,3,figsize=(15,5))

disp_1 = ConfusionMatrixDisplay(ct_rf)
disp_1.plot(ax=axes[0], cmap="viridis", values_format="d")

disp_2 = ConfusionMatrixDisplay(ct_rf_adasyn)
disp_2.plot(ax=axes[1], cmap="viridis")

disp_3 = ConfusionMatrixDisplay(ct_rf_smote)
disp_3.plot(ax=axes[2], cmap="viridis")

fig.suptitle("Matice zámen Random Forest")
plt.tight_layout()
plt.show()

# %% [markdown]
# # DT

# %%
##Decision Tree pred
# Natrénuj model Decision Tree na trénovacích dátach
model_DTA = DecisionTreeClassifier(random_state=0)
model_DTA = model_DTA.fit(train_data2, train_label2)
result = model_DTA.predict(test_data2)

# Získaj dôležitosť atribútov
importances_dtAA = model_DTA.feature_importances_

# Zorad atribúty podľa dôležitosti
sorted_indicesAA = importances_dtAA.argsort()[::-1]
top_feature_indicesAA = sorted_indicesAA[:5]
top_feature_namesAA = [train_data2.columns[i] for i in top_feature_indicesAA]

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.bar(top_feature_namesAA, importances_dtAA[top_feature_indicesAA])
plt.xlabel("Atribúty")
plt.ylabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Decision Tree - bez vzorkovania")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vytvor kontingenčnú tabuľku
ct_dt = confusion_matrix(test_label2, result)
print(ct_dt)
print(classification_report(test_label2, result, digits=4))


# Výpočet ROC krivky
fprda, tprda, threshold = roc_curve(test_label2, result)
roc_aucda = auc(fprda, tprda)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprda, tprda, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucda))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Decision Tree - bez nadvzorkovania')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucda)

# %%
##Decision Tree adasyn
# Natrénuj model Decision Tree na trénovacích dátach
model_DTB = DecisionTreeClassifier(random_state=0)
model_DTB = model_DTB.fit(x_train_adasyn, y_train_adasyn)
result2 = model_DTB.predict(x_test_adasyn)

# Získaj dôležitosť atribútov
importances_dtBB = model_DTB.feature_importances_

# Zorad atribúty podľa dôležitosti
sorted_indicesBB = importances_dtBB.argsort()[::-1]
top_feature_indicesBB = sorted_indicesBB[:5]
top_feature_namesBB = [x_train_adasyn.columns[i] for i in top_feature_indicesBB]

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.bar(top_feature_namesBB, importances_dtBB[top_feature_indicesBB])
plt.xlabel("Atribúty")
plt.ylabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Decision Tree - ADASYN")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vytvor kontingenčnú tabuľku
ct_dt_adasyn = confusion_matrix(y_test_adasyn, result2)
print(ct_dt_adasyn)
print(classification_report(y_test_adasyn, result2, digits=4))

# Výpočet ROC krivky
fprdb, tprdb, threshold = roc_curve(y_test_adasyn, result2)
roc_aucdb = auc(fprdb, tprdb)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprdb, tprdb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucdb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Decision Tree - ADASYN')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucdb)


# %%
##Decision Tree smote

# Natrénuj model Decision Tree na trénovacích dátach
model_DTC = DecisionTreeClassifier(random_state=0)
model_DTC = model_DTC.fit(x_train_smote, y_train_smote)
result3 = model_DTC.predict(x_test_smote)

# Získaj dôležitosť atribútov
importances_dtCC = model_DTC.feature_importances_

# Zorad atribúty podľa dôležitosti
sorted_indicesCC = importances_dtCC.argsort()[::-1]
top_feature_indicesCC = sorted_indicesCC[:5]
top_feature_namesCC = [x_train_smote.columns[i] for i in top_feature_indicesCC]

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.bar(top_feature_namesCC, importances_dtCC[top_feature_indicesCC])
plt.xlabel("Atribúty")
plt.ylabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Decision Tree - SMOTE")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vytvor kontingenčnú tabuľku
ct_dt_smote = confusion_matrix(y_test_smote, result3)
print(ct_dt_smote)
print(classification_report(y_test_smote, result3,digits=4))

# Výpočet ROC krivky
fprdc, tprdc, threshold = roc_curve(y_test_smote, result3)
roc_aucdc = auc(fprdc, tprdc)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprdc, tprdc, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucdc))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Decision Tree - Smote')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucdc)

# %%
# Vytvorenie grafu pre model RF
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(fprda, tprda, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucda))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Decision Tree - bez nadvzorkovania')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model KNN
plt.subplot(132)
plt.plot(fprdb, tprdb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucdb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Decision Tree - ADASYN')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model SVM
plt.subplot(133)
plt.plot(fprdc, tprdc, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucdc))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Decision Tree - Smote')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# %%
fig,axes = plt.subplots(1,3,figsize=(15,5))

disp_1 = ConfusionMatrixDisplay(ct_dt)
disp_1.plot(ax=axes[0], cmap="viridis", values_format="d")

disp_2 = ConfusionMatrixDisplay(ct_dt_adasyn)
disp_2.plot(ax=axes[1], cmap="viridis")

disp_3 = ConfusionMatrixDisplay(ct_dt_smote)
disp_3.plot(ax=axes[2], cmap="viridis")

fig.suptitle("Matice zámen Decision Tree")
plt.tight_layout()
plt.show()

# %% [markdown]
# # SVM

# %%
#SVM pred
# Natrénuj model SVM s lineárnym jadrom na trénovacích dátach
model_SVMA = SVC(kernel='linear',random_state=0)
model_SVMA = model_SVMA.fit(train_data2, train_label2)
result = model_SVMA.predict(test_data2)

# Získaj dôležitosť atribútov
importances_svmAA = model_SVMA.coef_[0]
feature_namesAA = train_data2.columns

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.barh(feature_namesAA, importances_svmAA)
plt.xlabel("Dôležitosť")
plt.title("Dôležitosť atribútov - SVM - bez vzorkovania")
plt.show()

# Vytvor kontingenčnú tabuľku
ct_svm = confusion_matrix(test_label2, result)
print(ct_svm)
print(classification_report(test_label2, result,digits=4))

# Výpočet ROC krivky
fprea, tprea, threshold = roc_curve(test_label2, result)
roc_aucea = auc(fprea, tprea)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprea, tprea, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucea))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - SVM - bez nadvzorkovania')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucea)

# %%
# #SVM adasyn
# Natrénuj model SVM s lineárnym jadrom na trénovacích dátach
model_SVMB = SVC(kernel='linear',random_state=0)
model_SVMB = model_SVMB.fit(x_train_adasyn, y_train_adasyn)
result = model_SVMB.predict(x_test_adasyn)

# Získaj dôležitosť atribútov
importances_svmBB = model_SVMB.coef_[0]
feature_namesBB = x_train_adasyn.columns

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.barh(feature_namesBB, importances_svmBB)
plt.xlabel("Dôležitosť")
plt.title("Dôležitosť atribútov - SVM - ADASYN")
plt.show()

# Vytvor kontingenčnú tabuľku
ct_svm_adasyn = confusion_matrix(y_test_adasyn, result)
print(ct_svm_adasyn)
print(classification_report(y_test_adasyn, result,digits=4))

# Výpočet ROC krivky
fpreb, tpreb, threshold = roc_curve(y_test_adasyn, result)
roc_auceb = auc(fpreb, tpreb)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fpreb, tpreb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_auceb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - SVM - ADASYN')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_auceb)

# %%
#SVM smote
# Natrénuj model SVM s lineárnym jadrom na trénovacích dátach
model_SVMC = SVC(kernel='linear',random_state=0)
model_SVMC = model_SVMC.fit(x_train_smote, y_train_smote)
result = model_SVMC.predict(x_test_smote)

# Získaj dôležitosť atribútov
importances_svmCC = model_SVMC.coef_[0]
feature_namesCC = x_train_smote.columns

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.barh(feature_namesCC, importances_svmCC)
plt.xlabel("Dôležitosť")
plt.title("Dôležitosť atribútov - SVM - SMOTE")
plt.show()

# Vytvor kontingenčnú tabuľku
ct_svm_smote = confusion_matrix(y_test_smote, result)
print(ct_svm_smote)
print(classification_report(y_test_smote, result,digits=4))

# Výpočet ROC krivky
fprec, tprec, threshold = roc_curve(y_test_smote, result)
roc_aucec = auc(fprec, tprec)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprec, tprec, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucec))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - SVM - Smote')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucec)

# %%
import matplotlib.pyplot as plt

# Vytvorenie grafu pre model RF
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(fprea, tprea, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucea))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - SVM - bez nadvzorkovania')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model KNN
plt.subplot(132)
plt.plot(fpreb, tpreb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_auceb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - SVM - ADASYN')
plt.legend(loc='lower right')

# Vytvorenie grafu pre model SVM
plt.subplot(133)
plt.plot(fprec, tprec, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucec))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - SVM - Smote')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# %%
fig,axes = plt.subplots(1,3,figsize=(15,5))

disp_1 = ConfusionMatrixDisplay(ct_svm)
disp_1.plot(ax=axes[0], cmap="viridis", values_format="d")

disp_2 = ConfusionMatrixDisplay(ct_svm_adasyn)
disp_2.plot(ax=axes[1], cmap="viridis")

disp_3 = ConfusionMatrixDisplay(ct_svm_smote)
disp_3.plot(ax=axes[2], cmap="viridis")

fig.suptitle("Matice zámen SVM")
plt.tight_layout()
plt.show()

# %% [markdown]
# # LR

# %%
#LR bez samplingu

#Vytvorenie modelu
LM1 = LogisticRegression(random_state=0)
model_LMA = LM1.fit(train_data2, train_label2)
LRresulta = model_LMA.predict(test_data2)

importances_LRa = model_LMA.coef_[0]
feature_namesLRa = train_data2.columns
#Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.barh(feature_namesLRa, importances_LRa)
plt.xlabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Logistická Regresia - bez vzorkovania")
plt.show()

#Vytvorenie confusion metrix a vypísanie classification reportu
ct_lra = confusion_matrix(test_label2, LRresulta) 
print(ct_lra)
print(classification_report(test_label2, LRresulta,digits=4)) 

# Výpočet ROC krivky
fprfa, tprfa, threshold = roc_curve(test_label2, LRresulta)
roc_aucfa = auc(fprfa, tprfa)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprfa, tprfa, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucfa))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Logistická regresia - bez nadvzorkovania')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucfa)

# %%
#LR ADASYN

#Vytvorenie modelu
LM1 = LogisticRegression(random_state=0)
model_LMB = LM1.fit(x_train_adasyn, y_train_adasyn)
LRresultb = model_LMB.predict(x_test_adasyn)

importances_LRb = model_LMB.coef_[0]
feature_namesLRb = x_train_adasyn.columns

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.barh(feature_namesLRb, importances_LRb)
plt.xlabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Logistická Regresia - ADASYN")
plt.show()

ct_lr_adasyn = confusion_matrix(y_test_adasyn, LRresultb) 
print(ct_lr_adasyn)
print(classification_report(y_test_adasyn, LRresultb,digits=4))

# Výpočet ROC krivky
fprfb, tprfb, threshold = roc_curve(y_test_adasyn, LRresultb)
roc_aucfb = auc(fprfb, tprfb)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprfb, tprfb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucfb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - LR - ADASYN')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucfb)

# %%
#LR SMOTE
#Vytvorenie modelu
LM1 = LogisticRegression(random_state=0)
model_LMC = LM1.fit(x_train_smote, y_train_smote)
LRresultc = model_LMC.predict(x_test_smote)

importances_LRc = model_LMC.coef_[0]
feature_namesLRc = x_train_smote.columns
# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.barh(feature_namesLRc, importances_LRc)
plt.xlabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Logistická Regresia - SMOTE")
plt.show()

ct_lr_smote = confusion_matrix(y_test_smote, LRresultc) 
print(ct_lr_smote)
print(classification_report(y_test_smote, LRresultc,digits=4))

# Výpočet ROC krivky
fprfc, tprfc, threshold = roc_curve(y_test_smote, LRresultc)
roc_aucfc = auc(fprfc, tprfc)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprfc, tprfc, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucfc))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - LR - Smote')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucfc)

# %%
import matplotlib.pyplot as plt

# Vytvorenie grafu pre model LR
plt.figure(figsize=(15, 5))  # Zväčšenie výšky pre zobrazenie grafov pod sebou

# Subplot 1: LR bez nadvzorkovania
plt.subplot(131)
plt.plot(fprfa, tprfa, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucfa))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - LR - bez nadvzorkovania')
plt.legend(loc='lower right')

# Subplot 2: LR s ADASYN
plt.subplot(132)
plt.plot(fprfb, tprfb, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucfb))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - LR - ADASYN')
plt.legend(loc='lower right')

# Subplot 3: LR s Smote
plt.subplot(133)
plt.plot(fprfc, tprfc, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucfc))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - LR - Smote')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# %%
fig,axes = plt.subplots(1,3,figsize=(15,5))

disp_1 = ConfusionMatrixDisplay(ct_lra)
disp_1.plot(ax=axes[0], cmap="viridis", values_format="d")

disp_2 = ConfusionMatrixDisplay(ct_lr_adasyn)
disp_2.plot(ax=axes[1], cmap="viridis")

disp_3 = ConfusionMatrixDisplay(ct_lr_smote)
disp_3.plot(ax=axes[2], cmap="viridis")

fig.suptitle("Matice zámen Logistická Regresia")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Porovnanie s atributmi z chi-kvadrat testu

# %%
##Bayes SMOTE2             

# Natrénovanie modelu Naive Bayes
modelBCS = GaussianNB()
modelBCS.fit(x_train_smote2, y_train_smote2)

# Výber najdôležitejších atribútov
selectorBS = SelectKBest(chi2, k=5)  # Vyberte počet atribútov, ktoré chcete
selected_train_dataBS = selectorBS.fit_transform(x_train_smote2, y_train_smote2)

# Získanie indexov vybraných atribútov
selected_indicesBS = selectorBS.get_support(indices=True)

# Vypísanie názvov vybraných atribútov
selected_feature_namesBS = [x_train_smote2.columns[i] for i in selected_indicesBS]

# Vytvorenie grafu
plt.figure(figsize=(10, 6))
plt.bar(selected_feature_namesBS, selectorBS.scores_[selected_indicesBS])
plt.xlabel("Atribúty")
plt.ylabel("Chi-kvadrát hodnota")
plt.title("Dôležitosť atribútov (chi-kvadrát test)- Naive Bayes- SMOTE- bez chi-kvadrátu")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

resultBS = modelBCS.predict(x_test_smote2)                            

ct_bayes_smoteS = pd.crosstab(y_test_smote2, resultBS)
print(ct_bayes_smoteS)                        
print(classification_report(y_test_smote2, resultBS,digits=4))

fpracS, tpracS, _ = roc_curve(y_test_smote2, resultBS)
roc_aucacS = auc(fpracS, tpracS)

# Zobrazenie hodnoty AUC na grafe
plt.plot(fpracS, tpracS, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucacS))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Naive Bayes - Smote- bez chi-kvadrátu')
plt.legend(loc='lower right')

# Zobrazenie grafu
plt.show()
print('AUC' + str(roc_aucacS))



# %%
##Random Forrest smote
# Natrénuj model Random Forest na trénovacích dátach
model_rfCa = RandomForestClassifier(random_state=0)
model_rfCa.fit(x_train_smote2, y_train_smote2)

# Predikuj na testovacích dátach
pred3a = model_rfCa.predict(x_test_smote2)

# Získaj dôležitosť atribútov
importances_rfCa = model_rfCa.feature_importances_

# Zorad atribúty podľa dôležitosti
sorted_indicesCa = importances_rfCa.argsort()[::-1]
top_feature_indicesCa = sorted_indicesCa[:5]
top_feature_namesCa = [x_train_smote2.columns[i] for i in top_feature_indicesCa]

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.bar(top_feature_namesCa, importances_rfCa[top_feature_indicesCa])
plt.xlabel("Atribúty")
plt.ylabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Random Forest- SMOTE- bez chi-kvadrátu")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Vytvor kontingenčnú tabuľku
ct_rf_smotea = pd.crosstab(y_test_smote2, pred3a)
print(ct_rf_smotea)
print(classification_report(y_test_smote2, pred3a,digits=4))

# Výpočet ROC krivky
fprccs, tprccs, threshold = roc_curve(y_test_smote2, pred3a)
roc_aucccs = auc(fprccs, tprccs)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprccs, tprccs, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucccs))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Random Forrest - Smote')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucccs)

# %%
##Decision Tree smote

# Natrénuj model Decision Tree na trénovacích dátach
model_DTCS = DecisionTreeClassifier(random_state=0)
model_DTCS = model_DTCS.fit(x_train_smote2, y_train_smote2)
result2a = model_DTCS.predict(x_test_smote2)

# Získaj dôležitosť atribútov
importances_dtCCa = model_DTCS.feature_importances_

# Zorad atribúty podľa dôležitosti
sorted_indicesCCa = importances_dtCCa.argsort()[::-1]
top_feature_indicesCCa = sorted_indicesCCa[:5]
top_feature_namesCC = [x_train_smote2.columns[i] for i in top_feature_indicesCCa]

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.bar(top_feature_namesCC, importances_dtCC[top_feature_indicesCC])
plt.xlabel("Atribúty")
plt.ylabel("Dôležitosť")
plt.title("Dôležitosť atribútov - Decision Tree - SMOTE- bez chi-kvadrátu")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vytvor kontingenčnú tabuľku
ct_dt_smotea = pd.crosstab(y_test_smote2, result2a)
print(ct_dt_smotea)
print(classification_report(y_test_smote2, result2a,digits=4))

# Výpočet ROC krivky
fprdcs, tprdcs, threshold = roc_curve(y_test_smote2, result2a)
roc_aucdcs = auc(fprdcs, tprdcs)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprdcs, tprdcs, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucdcs))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - Decision Tree - Smote')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucdcs)

# %%
#LR SMOTE


#Vytvorenie modelu
LM1a = LogisticRegression(random_state=0)
LM1a = LM1a.fit(x_train_smote2, y_train_smote2)
LRresultca = LM1a.predict(x_test_smote2)

importances_LRca = LM1a.coef_[0]
feature_namesLRca = x_train_smote2.columns
# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.barh(feature_namesLRca, importances_LRca)
plt.xlabel("Dôležitosť")
plt.title("Dôležitosť atribútov - LR- SMOTE- bez chi-kvadrátu")
plt.show()

ct_lr_smoted = pd.crosstab(y_test_smote2, LRresultca) 
print(ct_lr_smoted)
print(classification_report(y_test_smote2, LRresultca,digits=4))

# Výpočet ROC krivky
fprfd, tprfd, threshold = roc_curve(y_test_smote2, LRresultca)
roc_aucfd = auc(fprfd, tprfd)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprfd, tprfd, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucfd))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - LR - Smote')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucfd)

# %%
# #SVM adasyn
# Natrénuj model SVM s lineárnym jadrom na trénovacích dátach
model_SVMD = SVC(kernel='linear',random_state=0)
model_SVMD = model_SVMD.fit(x_train_adasyn2, y_train_adasyn2)
result = model_SVMD.predict(x_test_adasyn2)

# Získaj dôležitosť atribútov
importances_svmBD = model_SVMD.coef_[0]
feature_namesBD = x_train_adasyn2.columns

# Vytvor graf dôležitosti atribútov
plt.figure(figsize=(10, 6))
plt.barh(feature_namesBD, importances_svmBD)
plt.xlabel("Dôležitosť")
plt.title("Dôležitosť atribútov - SVM - ADASYN")
plt.show()

# Vytvor kontingenčnú tabuľku
ct_svm_adasyn2 = pd.crosstab(y_test_adasyn2, result)
print(ct_svm_adasyn2)
print(classification_report(y_test_adasyn2, result,digits=4))

# Výpočet ROC krivky
fpred, tpred, threshold = roc_curve(y_test_adasyn2, result)
roc_auced = auc(fpred, tpred)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fpred, tpred, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_auced))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - SVM - ADASYN')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_auced)

# %%
##kNN SMOTE

model1d = KNeighborsClassifier(n_neighbors = 31)
fitad = model1d.fit(x_train_smote2, y_train_smote2)
res3d = model1d.predict(x_test_smote2)

ct_knn_smoted = pd.crosstab(y_test_smote2, res3d)
print(ct_knn_smoted)  
print(classification_report(y_test_smote2, res3d,digits=4))

# Výpočet ROC krivky
fprbd, tprbd, threshold = roc_curve(y_test_smote2, res3d)
roc_aucbd = auc(fprbd, tprbd)

# Vykreslenie ROC krivky
plt.figure()
plt.plot(fprbd, tprbd, color='b', label='ROC krivka(AUC = {:.4f})'.format(roc_aucbd))
plt.plot([0, 1], [0, 1], color='r',label='Náhoda')
plt.xlabel('Falošne pozitívne hodnoty')
plt.ylabel('Skutočne pozitívne hodnoty')
plt.title('ROC krivka - kNN - Smote')
plt.legend(loc='lower right')
plt.show()

# Výpis AUC hodnoty
print('AUC:', roc_aucbd)

# %%


# %%


# %%



