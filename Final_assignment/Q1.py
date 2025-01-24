import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np  # log transformations

# Load data
animal_protein_cons = pd.read_csv("finalproject/Co2capta/animal-protein-consumption.csv")
co2_percap = pd.read_csv("finalproject/Co2capta/consumption-co2-per-capita-vs-gdppc.csv")
dom_aviation_percap = pd.read_csv("finalproject/Co2capta/per-capita-co2-domestic-aviation.csv")
energy_use_percap = pd.read_csv("finalproject/Co2capta/per-capita-energy-use.csv")

# Preprocess protein consumption
animal_protein_cons.fillna(0.0, inplace=True)  # fill empty parts with 0

# sum everything with iloc from column 4 and on, save into new column and drop other cokumns except for last one
animal_protein_cons["Protein"] = animal_protein_cons.iloc[:, 3:].sum(axis=1)
animal_protein_cons.drop(animal_protein_cons.columns[3:-1], axis=1, inplace=True)

# Choose columns
co2_percap = co2_percap[["Entity", "Year", "Per capita consumption-based CO₂ emissions"]]
animal_protein_cons = animal_protein_cons[["Entity", "Year", "Protein"]]
dom_aviation_percap = dom_aviation_percap[["Entity", "Year", "Per capita domestic aviation CO2"]]
energy_use_percap = energy_use_percap[["Entity", "Year", "Primary energy consumption per capita (kWh/person)"]]

# Rename columns
co2_percap.rename(columns={"Entity": "Country", "Per capita consumption-based CO₂ emissions": "CO2"}, inplace=True)
animal_protein_cons.rename(columns={"Entity": "Country"}, inplace=True)
dom_aviation_percap.rename(columns={"Entity": "Country", "Per capita domestic aviation CO2": "Aviation"}, inplace=True)
energy_use_percap.rename(columns={"Entity": "Country", "Primary energy consumption per capita (kWh/person)": "Energy"}, inplace=True)

# Data Cleaning
animal_protein_cons.dropna(inplace=True)
co2_percap.dropna(inplace=True)
dom_aviation_percap.dropna(inplace=True)
energy_use_percap.dropna(inplace=True)

# only compare the data from datasets that have same data in years and input, when it appears in the both we use otherwise we drop
protein_vs_co2 = pd.merge(animal_protein_cons, co2_percap, on=["Country", "Year"], how="inner")
aviation_vs_co2 = pd.merge(dom_aviation_percap, co2_percap, on=["Country", "Year"], how="inner")
energy_vs_co2 = pd.merge(energy_use_percap, co2_percap, on=["Country", "Year"], how="inner")

protein_corr = protein_vs_co2["Protein"].corr(protein_vs_co2["CO2"])
aviation_corr = aviation_vs_co2["Aviation"].corr(aviation_vs_co2["CO2"])
energy_corr = energy_vs_co2["Energy"].corr(energy_vs_co2["CO2"])

print(f"Protein: {protein_corr}, Aviation: {aviation_corr}, Energy: {energy_corr}")

# Scatter plot of Protein vs CO2 per capita for each year and country in three subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Protein vs CO2
axes[0].scatter(protein_vs_co2["Protein"], protein_vs_co2["CO2"], c='blue', label='Protein')
axes[0].set_title("Protein vs CO2 per Capita")
axes[0].set_xlabel("Protein Consumption")
axes[0].set_ylabel("CO2 per Capita")
axes[0].legend()

# Aviation vs CO2
axes[1].scatter(aviation_vs_co2["Aviation"], aviation_vs_co2["CO2"], c='green', label='Aviation')
axes[1].set_title("Aviation vs CO2 per Capita")
axes[1].set_xlabel("Aviation CO2")
axes[1].set_ylabel("CO2 per Capita")
axes[1].legend()

# Energy vs CO2
axes[2].scatter(energy_vs_co2["Energy"], energy_vs_co2["CO2"], c='red', label='Energy')
axes[2].set_title("Energy vs CO2 per Capita")
axes[2].set_xlabel("Energy Consumption")
axes[2].set_ylabel("CO2 per Capita")
axes[2].legend()

plt.tight_layout()
plt.show()

