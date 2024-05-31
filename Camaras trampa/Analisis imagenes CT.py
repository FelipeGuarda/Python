import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"G:\Shared drives\FMA-G\C-CONSERVACIÓN\C1_Bosque Pehuen\6 Bosque Pehuen\Conservación\Camaras trampa\2022 - febr. 2024\Fotos\Base de datos para analisis 2022 - feb. 2024.csv"
data = pd.read_csv(file_path)

print (data.head())

unique_paths = data['RelativePath'].unique()
print(unique_paths)
# Continue with your code using the 'data' variable

data['Estacion'] = np.where(data['RelativePath'].isin(["2022\\Araucarias\\CT10_06_12_22", "2022\\Araucarias\\CT9", "2023\\Araucarias\\CT 3 hermanas_16_02_23", "2023\\Araucarias\\CT 3 hermanas_enero 23", "2023\\Araucarias\\camtrap 3 hermanas_16_02_23", "2023\\Araucarias\\camtrap 8_02_23", "2024\\TC_10", "2024\\TC_9\\100EK113"]), "Araucaria",
                           np.where(data['RelativePath'].isin(["2022\\CoRaTe Quebrada\\CT13"]), "CoRaTe quebrada",
                                    np.where(data['RelativePath'].isin(["2022\\CoRaTe\\CT11_12_2022", "2022\\CoRaTe\\Puma", "2023\\TC11", "2023\\TC12"]), "CoRaTe",
                                             np.where(data['RelativePath'].isin(["2022\\Lenga - Pradera\\CT16"]), "Lenga pradera",
                                                      np.where(data['RelativePath'].isin(["2022\\Lenga\\CT1", "2022\\Lenga\\CT2", "2024\\TC_1", "2024\\TC_2"]), "Lenga",
                                                               np.where(data['RelativePath'].isin(["2022\\RoRaCo Renoval\\CT5", "2022\\RoRaCo Renoval\\CT6\\100EK113", "2022\\RoRaCo Renoval\\CT6\\103EK113", "2023\\RoRaCo Renoval\\TC5", "2023\\RoRaCo Renoval\\TC6", "2024\\TC_5", "2024\\TC_6"]), "RoRaCo renoval",
                                                                        np.where(data['RelativePath'].isin(["2022\\RoRaCo Rio\\CT3"]), "RoRaCo rio",
                                                                                 np.where(data['RelativePath'].isin(["2022\\Santuario Anfibios\\CT7", "2022\\Santuario Anfibios\\CT8", "2022\\Santuario Anfibios\\CT8_nov_22", "2023\\Santuario Anfibios\\TC7", "2023\\Santuario Anfibios\\TC8", "2024\\TC_7", "2024\\TC_8"]), "Santuario anfibios", ""))))))))


unique_estacion = data['Estacion'].unique()
print(unique_estacion)

missing_values = data['Estacion'].isnull().sum()
print("Number of missing values in Estacion:", missing_values)

data['ID_CamaraTrampa'] = np.where(data['RelativePath'].isin(["2022\\Araucarias\\CT10_06_12_22", "2023\\Araucarias\\CT 3 hermanas_16_02_23", "2023\\Araucarias\\CT 3 hermanas_enero 23"]), "CT 10",
                                  np.where(data['RelativePath'].isin(["2022\\Araucarias\\CT9", "2024\\TC_9\\100EK113"]), "CT 09",
                                           np.where(data['RelativePath'].isin(["2022\\CoRaTe Quebrada\\CT13"]), "CT 13",
                                                    np.where(data['RelativePath'].isin(["2022\\CoRaTe\\CT11_12_2022", "2022\\CoRaTe\\Puma", "2023\\TC11", "2023\\TC12"]), "CT 11 y 12",
                                                             np.where(data['RelativePath'].isin(["2022\\Lenga - Pradera\\CT16"]), "CT 16",
                                                                      np.where(data['RelativePath'].isin(["2022\\Lenga\\CT1", "2024\\TC_1"]), "CT 01",
                                                                               np.where(data['RelativePath'].isin(["2022\\Lenga\\CT2", "2024\\TC_2"]), "CT 02",
                                                                                        np.where(data['RelativePath'].isin(["2022\\RoRaCo Renoval\\CT5", "2023\\RoRaCo Renoval\\TC5", "2024\\TC_5"]), "CT 05",
                                                                                                 np.where(data['RelativePath'].isin(["2022\\RoRaCo Renoval\\CT6\\100EK113", "2022\\RoRaCo Renoval\\CT6\\103EK113", "2023\\RoRaCo Renoval\\TC6", "2024\\TC_6"]), "CT 06",
                                                                                                          np.where(data['RelativePath'].isin(["2022\\RoRaCo Rio\\CT3"]), "CT 03",
                                                                                                                   np.where(data['RelativePath'].isin(["2022\\Santuario Anfibios\\CT7", "2023\\Santuario Anfibios\\TC7", "2024\\TC_7"]), "CT 07",
                                                                                                                            np.where(data['RelativePath'].isin(["2022\\Santuario Anfibios\\CT8", "2022\\Santuario Anfibios\\CT8_nov_22", "2023\\Santuario Anfibios\\TC8", "2024\\TC_8"]), "CT 08",
                                                                                                                                     np.where(data['RelativePath'].isin(["2023\\Araucarias\\camtrap 3 hermanas_16_02_23", "2023\\Araucarias\\camtrap 8_02_23"]), "Camtraption", "")))))))))))))

unique_ID_CamaraTrampa = data['ID_CamaraTrampa'].unique()
print(unique_ID_CamaraTrampa)

missing_values = data['ID_CamaraTrampa'].isnull().sum()
print("Number of missing values in ID_CamaraTrampa:", missing_values)

data = data.drop('Unnamed: 9', axis=1)
print(data.head(100))

# Convert the column to datetime format
data['DateTime'] = pd.to_datetime(data['DateTime'])

# Extract year, month, day, and time
data['Year'] = data['DateTime'].dt.year
data['Month'] = data['DateTime'].dt.month
data['Day'] = data['DateTime'].dt.day
data['Time'] = data['DateTime'].dt.time

print (data)


unique_especie = data['Especie'].unique()
print(unique_especie)


species_by_year = data.groupby('Year')['Especie'].value_counts().unstack().fillna(0)
species_by_year = species_by_year.T
print(species_by_year)

species_by_camara = data.groupby('ID_CamaraTrampa')['Especie'].value_counts().unstack().fillna(0)
species_by_camara = species_by_camara.T
print(species_by_camara)