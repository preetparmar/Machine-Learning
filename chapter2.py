# Imorting Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Importing Data
housing = pd.read_csv('Data\\housing.csv')

""" Initial Exploration """
housing.head()
housing.info()
housing.describe()

### Visualizing the Data

# Histogram
housing.hist(figsize=(50, 60), bins=50)
plt.show()

### Scatter Plot

# Importing Image
img = mpimg.imread('images\\end_to_end_project\\california.png')

# Main Plot
ax = housing.plot(kind='scatter', x='longitude', y='latitude', figsize=(10,7),
                    c=housing['median_house_value'], cmap='jet', alpha=0.4, colorbar=False,
                    s=housing['population']/100, label='Population')

# Plotting the image
# print(plt.xlim(), plt.ylim())  # To get extent values
plt.imshow(img, extent=[-124.85, -113.80, 32.06, 42.42], alpha=0.5)

# Plotting the colorbar
prices = housing['median_house_value']
tick_values = np.linspace(min(prices), max(prices), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(['%dk'%round(x/1000) for x in tick_values], font_size=10)
cbar.set_label('Median House Value')

# Labeling
plt.title('Median House Value Distribution in California')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(fontsize=10)

# Show the plot
plt.show()


### Correlation
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ['median_house_value','median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

housing.plot(kind='scatter', x='median_house_value', y='median_income')
plt.title('House Value VS Median Income')
plt.xlabel('House Value')
plt.ylabel('Median Income')
plt.show()

# Adding in more columns

""" Splitting the Data """
housing['median_income'].hist(bins=5, c='blue')

housing['income_category'] = pd.cut(housing['median_income'], 
                                    bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                    labels=[1, 2, 3, 4, 5])

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_ix, test_ix in split.split(housing, housing['income_category']):
    housing_train = housing.iloc[train_ix]
    housing_test = housing.iloc[test_ix]

for data_ in (housing_test, housing_train):
    data_.drop('income_category', inplace=True, axis=1)


# Preparing the data for Machine Learning

from sklearn.base import BaseEstimator, TransformerMixin
room_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6

class combinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedroom_ix]/X[:, room_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

housing = housing_train.drop('median_house_value', axis=1)
housing_labels = housing_train['median_house_value'].copy()

num_attributes = list(housing.columns)[:-1]
cat_attributes = ['ocean_proximity']

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('add_attributes', combinedAttributesAdder()),
    ('scale', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder()),
])

full_pipeline = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes),
])


housing_prepared = full_pipeline.fit_transform(housing)

# Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_predictions = lin_reg.predict(housing_prepared)

# Evaluating
from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(housing_labels, lin_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, lin_predictions)
print(lin_mae)

from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(estimator=lin_reg, X=housing_prepared, y=housing_labels,
        scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
pd.Series(lin_rmse_scores).describe()

# Fine Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=forest_reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribution = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8)
}
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distribution,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
rnd_search.best_params_