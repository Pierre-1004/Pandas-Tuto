import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_excel('Crime_Data.xlsx')

#print(data.head())

#Création du Dataframe
df = pd.DataFrame(data)
#print(df)

#Filter et extraire
data_baltimore = df[df['JURISDICTION'] == 'Baltimore City']
#print(data_baltimore)

#Trier
dataframe_trie = df.sort_values('JURISDICTION', ascending = True)

#Fusion
data_robbery =df[df['ROBBERY'] > 0]
data_wo_robery = df[df['ROBBERY'] == 0]

whole_df = pd.merge(data_robbery, data_wo_robery, on ='JURISDICTION')

#Transformation
data_paradise = pd.DataFrame(df)
data_paradise['ROBBERY'] = data_paradise['ROBBERY'].apply(lambda x: x*0)
#print(data_paradise['ROBBERY'])

##############################################################################"

#Statistiques avec Pandas
statistique = df.describe()

#Calcul d'aggrégats
test_agg = df.groupby(['JURISDICTION']).agg(count_year =('YEAR', 'count'),min_rob = ('ROBBERY',min), max_rob = ('ROBBERY', max), sum_rob = ('ROBBERY',sum))
#print(test_agg)

test_filter_agg = test_agg[test_agg['min_rob'] == 0]
#print(test_filter_agg)

#On affiche le nombre maximal par colonne
test_which_max_crime = df.groupby(['JURISDICTION']).agg(murder = ('MURDER',max), rape = ('RAPE', max), ROBBERY = ('ROBBERY',max), agg_assault = ('AGG. ASSAULT',max), b_and_e = ('B & E', max), larency_theft = ('LARCENY THEFT',max),mv_theft =('M/V THEFT',max))
print(test_which_max_crime)

#On sort pour chaque ville quelle est le crime le plus élevé
max_crime_by_city = test_which_max_crime.idxmax(axis =1)
print(max_crime_by_city)

####################################################""""""""""""""""""""""""""""""""

#Génération de visualisation
test_which_max_crime.plot(kind='hist', subplots =True)
#plt.show()

#########################################################################################
#Traitement des données manquantes

data_null = df.isnull()

#Supprimeles lignes contenant des valeurs manquantes
data_sans_na = df.dropna()

#Supprime les collonnes contenant des valeurs manquantes
data_sas_na = df.dropna(axis =1)

#remplace les NA par une valeur
data_clean = df.fillna(0)

#Gestion des données manquantes dans les calculs
moyenne = df['MURDER'].mean(skipna =True)

#########################################################################################
#Optimisation des performances

dataframe = pd.read_excel('Crime_Data.xlsx', dtype={'JURISDICTION' : str, 'MUERDER' : int})


#Utilisation de .loc pour un acccès par étiquette
valeur = df.loc[5,'YEAR']
print(valeur)

# 8.3  utilier .iloc pour l'accès par position

valeur = df.iloc[5,1]
print(valeur)
