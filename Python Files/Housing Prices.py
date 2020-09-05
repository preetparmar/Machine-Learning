# Importing Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# Importing the Data
housing = pd.read_csv('./Data/housing.csv')

# Initial description about the data
housing.info()
housing.describe()

# Visualizing the data
housing.hist(figsize=(50, 60), bins=50)
plt.show()

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# Let's add some new features and look at their coorelation
data = housing.copy()
data.columns

data['household_per_population'] = data['households']/data['population']
data['bedrooms_per_room'] = data['total_bedrooms']/data['total_rooms']
data['rooms_per_household'] = data['total_rooms']/data['households']
data['population_per_household'] = data['population']/data['households']

data.corr()['median_house_value'].sort_values(ascending=False)
