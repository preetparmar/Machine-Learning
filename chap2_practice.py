# Imorting Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit

# Importing Data
housing = pd.read_csv('Data\\housing.csv')

# Initial Exploration
housing.info()
housing.describe()

# Visualizing the data
housing.hist(figsize=(50, 60), bins=50)
plt.show()

# Checking correlation
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# Adding more features and checking it's correlation
data = housing.copy()
corr_matrix = data.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

data['bedrooms_per_room'] = data['total_bedrooms']/data['total_rooms']
data['rooms_per_househould'] = data['total_rooms']/data['households']
data['population_per_household'] = data['population']/data['households']

# Creating a class to add this functionality, so this can be added into the pipeline

class combinedAttributeAdder(BaseEstimator, TransformerMixin):
    # self.room_ix, self.bedroom_ix, self.population_ix, self.household_ix = 3, 4, 5, 6

    def __init__(self, add_bedrooms_per_rooms=True):
        self.add_bedrooms_per_rooms = add_bedrooms_per_rooms
        self.room_ix, self.bedroom_ix, self.population_ix, self.household_ix = 3, 4, 5, 6
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X[:,self.room_ix]/X[:,self.household_ix]
        population_per_household = X[:,self.population_ix]/X[:,self.household_ix]
        if self.add_bedrooms_per_rooms:
            bedrooms_per_room = X[:, self.bedroom_ix]/X[:, self.room_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

housing['median_income'].hist()
plt.show()

housing['median_income_category'] = pd.cut(housing['median_income'],
                                            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                            labels=[1, 2, 3, 4, 5])

housing['median_income_category'].hist()
plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_ix, test_ix in split.split(housing, housing['median_income_category']):
    housing_train = housing.iloc[train_ix, :]
    housing_test = housing.iloc[test_ix, :]


for data_ in (housing_train, housing_test):
    data_.drop('median_income_category', inplace=True, axis=1)


housing = housing_train.drop('median_house_value', axis=1)
housing_labels = housing_train['median_house_value'].copy()

num_attributes = list(housing)[:-1]
cat_attributes = ['ocean_proximity']

num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('add_attributes', combinedAttributeAdder()),
    ('scale', StandardScaler()),
])
cat_pipeline = Pipeline(steps=[
    ('one_hot', OneHotEncoder())

])

full_pipeline = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes),
])

housing_prepared = full_pipeline.fit_transform(housing)

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_predictions = lin_reg.predict(housing_prepared)

# Evaluating
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(housing_labels, lin_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


# grid search - random forest regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(estimator=forest_reg, 
                            param_grid=param_grid,
                            cv=5, 
                            scoring='neg_mean_squared_error',
                            return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
extra_attributes = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attributes = list(cat_encoder.named_steps['one_hot'].categories_[0])

attributes = num_attributes + extra_attributes + cat_one_hot_attributes
sorted(zip(feature_importances, attributes), reverse=True)

