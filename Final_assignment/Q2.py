import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np  # log transformations

co2_emission_gdp = pd.read_csv ("finalproject/decreasing co2/co2-emissions-vs-gdp/co2-emissions-vs-gdp.csv")
renewable_share_energy = pd.read_csv ("finalproject/decreasing co2/renewable-share-energy/renewable-share-energy.csv")
share_elec_renew = pd.read_csv ("finalproject/decreasing co2/share-electricity-renewables/share-electricity-renewables.csv")
supp_policy_climate = pd.read_csv ("finalproject/decreasing co2/support-policies-climate/support-policies-climate.csv")

# choose the columns
co2_emission_gdp = co2_emission_gdp [["Entity", "Year", "Annual CO₂ emissions (per capita)"]]
renewable_share_energy = renewable_share_energy [["Entity", "Year", "Renewables (% equivalent primary energy)"]]
share_elec_renew = share_elec_renew [["Entity", "Year", "Renewables - % electricity"]]
supp_policy_climate = supp_policy_climate [["Entity", "Year", "Support political action on climate"]]

# Rename columns
co2_emission_gdp.rename(columns={"Entity": "Country", "Annual CO₂ emissions (per capita)": "Co2"}, inplace=True)
renewable_share_energy.rename(columns={"Entity": "Country", "Renewables (% equivalent primary energy)": "Renewables"}, inplace=True)
share_elec_renew.rename(columns={"Entity": "Country", "Renewables - % electricity": "Ren Electricity"}, inplace=True)
supp_policy_climate.rename(columns={"Entity": "Country", "Support political action on climate": "Policy"}, inplace=True)

# Clean data
co2_emission_gdp.dropna(inplace=True)
renewable_share_energy.dropna(inplace=True)
share_elec_renew.dropna(inplace=True)
supp_policy_climate.dropna(inplace=True)

# compare co2 emission per capita with the countries over time 

# Calculate the slope of CO2 emissions for each country
slopes = {}
co2_emission_gdp_recent = co2_emission_gdp[co2_emission_gdp['Year'] >= 2010]
for country in co2_emission_gdp_recent['Country'].unique():
    country_data = co2_emission_gdp_recent[co2_emission_gdp_recent['Country'] == country]
    if len(country_data) > 1:
        slope = np.polyfit(country_data['Year'], country_data['Co2'], 1)[0]
        slopes[country] = slope

# Filter countries with negative slope
negative_slope_countries = [country for country, slope in slopes.items() if slope < 0]


# Filter the dataframe to include only countries with negative slope and data from 2010 onwards
filtered_co2_emission_gdp = co2_emission_gdp[(co2_emission_gdp['Country'].isin(negative_slope_countries)) & (co2_emission_gdp['Year'] >= 2010)]

# Sort the countries by the magnitude of their slope (biggest decrease first) and select the top N countries
N = 20  # Number of countries to select
top_countries = sorted(slopes, key=slopes.get)[:N]
print(top_countries)

for country in top_countries:
    print(f"{country}: Slope = {slopes[country]:.2f}")
    
# Filter the dataframe to include only the top N countries
filtered_co2_emission_gdp = filtered_co2_emission_gdp[filtered_co2_emission_gdp['Country'].isin(top_countries)]

# Plot the filtered data
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_co2_emission_gdp, x="Year", y="Co2", hue="Country")
plt.title(f'CO2 Emissions Per Capita Over Time (Top {N} Countries with Biggest Decrease)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (per capita)')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()

# Filter the supp_policy_climate dataframe to include only the top N countries and data from 2010 onwards
filtered_supp_policy_climate = supp_policy_climate[
    (supp_policy_climate["Country"].isin(top_countries))
    & (supp_policy_climate["Year"] >= 2010)
]

# Filter out countries that do not have Policy data
countries_with_policy_data = filtered_supp_policy_climate["Country"].unique()
filtered_co2_emission_gdp = filtered_co2_emission_gdp[
    filtered_co2_emission_gdp["Country"].isin(countries_with_policy_data)
]

# Calculate the mean policy support for each country
mean_policy_support = (
    filtered_supp_policy_climate.groupby("Country")["Policy"]
    .mean()
    .reindex(top_countries)
)
mean_policy_support.dropna(inplace=True)

# Plot the mean policy support
plt.figure(figsize=(12, 6))
mean_policy_support.plot(kind="bar")
plt.title(
    "Mean Support for Political Action on Climate (Top N Countries with Biggest Decrease in CO2 Emissions)"
)
plt.xlabel("Country")
plt.ylabel("Mean Support for Political Action on Climate")
plt.xticks(rotation=45)
plt.tight_layout()

# Calculate the mean policy support for the world
world_policy_support = supp_policy_climate[
    (supp_policy_climate["Country"] == "World") & (supp_policy_climate["Year"] >= 2010)
]["Policy"].mean()

# Draw a horizontal line representing the world mean policy support
plt.axhline(
    world_policy_support, color="r", linestyle="--", label="World Mean Policy Support"
)
plt.legend()

plt.show()
