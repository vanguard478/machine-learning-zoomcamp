import os
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from IPython.display import display

os.chdir(Path(__file__).parent)

# !wget https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv

df = pd.read_csv('data.csv')

df.head()

# print(f'the columns in the dataframe are {df.columns}')

print(f'the datatype are \n{df.dtypes}')

required_cols = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle Style', 'highway MPG', 'city mpg', 'MSRP'
]

df = df[required_cols]

# Renaming columns to lower case and replacing space by underscore
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.rename(columns={'msrp': 'price'})
for i in df.columns:
  if df[i].dtype == 'object':
    df[i] = df[i].str.lower().str.replace(' ', '_')

df.fillna(0, inplace=True)
# for i in df.columns:
#     if df[i].dtype == 'object':
#         print(f'the unique elements of {i} objects are {df[i].unique()}')
"""
### Question 1

What is the most frequent observation (mode) for the column `transmission_type`?

- `AUTOMATIC`
- `MANUAL`
- `AUTOMATED_MANUAL`
- `DIRECT_DRIVE`
    
"""

mode = df.transmission_type.mode()[0]
print(f'the most frequent observation (mode) for the column \
  transmission_type is {mode}')
"""
Question 2
Create the correlation matrix for the numerical features of your dataset. 
In a correlation matrix, you compute the correlation coefficient between 
every pair of features in the dataset.

What are the two features that have the biggest correlation in this dataset?

- `engine_hp` and `year`
- `engine_hp` and `engine_cylinders`
- `highway_mpg` and `engine_cylinders`
- `highway_mpg` and `city_mpg`


Answer 

the correlation between engine_hp and year is 0.3387141847624468
the correlation between engine_hp and engine_cylinders is 0.7748509807813194
the correlation between highway_mpg and engine_cylinders is -0.6145414173953326
the correlation between highway_mpg and city_mpg is 0.8868294962591357

"""

pairs = [('engine_hp', 'year'), ('engine_hp', 'engine_cylinders'),
         ('highway_mpg', 'engine_cylinders'), ('highway_mpg', 'city_mpg')]

for i in pairs:
  print(
      f'the correlation between {i[0]} and {i[1]} is {df[i[0]].corr(df[i[1]])}')

### Answer highway_mpg and city_mpg have the highest correlation
"""
### Make `price` binary

* Now we need to turn the `price` variable from numeric into a binary format.
* Let's create a variable `above_average` which is `1` if the `price` is above its mean value and `0` otherwise.
    
"""
mean_price = df.price.mean()
print(f'the mean price is {mean_price}')
df.price = (df.price > mean_price).astype(int)

# Create a scaler object
# scaler = MinMaxScaler()
# # scaler = StandardScaler()

# # Use the scaler to normalize the 'A' column and assign back to the DataFrame
# df['year'] = scaler.fit_transform(df[['year']])
# df['engine_hp'] = scaler.fit_transform(df[['engine_hp']])
# df['engine_cylinders'] = scaler.fit_transform(df[['engine_cylinders']])
# df['highway_mpg'] = scaler.fit_transform(df[['highway_mpg']])
"""
### Split the data

* Split your data in train/val/test sets with 60%/20%/20% distribution.
* Use Scikit-Learn for that (the `train_test_split` function) and set the seed 
to `42`.
* Make sure that the target value (`price`) is not in your dataframe.
        
"""

required_split = [60, 20, 20]    # train, val, test

df_train_full, df_test = train_test_split(df,
                                          test_size=required_split[2] / 100,
                                          random_state=42)

df_train, df_val = train_test_split(df_train_full,
                                    test_size=required_split[1] /
                                    (required_split[0] + required_split[1]),
                                    random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(
    f'the length of train set is {len(df_train)} , val set is {len(df_val)} and test set is {len(df_test)}'
)

df_train.head()
"""
### Question 3

* Calculate the mutual information score between `above_average` and other 
categorical variables in our dataset. 
  Use the training set only.
* Round the scores to 2 decimals using `round(score, 2)`.

Which of these variables has the lowest mutual information score?
  
- `make`
- `model`
- `transmission_type`
- `vehicle_style`

model                0.460994
make                 0.238724
vehicle_style        0.083390
transmission_type    0.020884

"""

print(f'the columns in df_train are {df_train.columns}')
categorical = [
    'make',
    'model',
    'transmission_type',
    'vehicle_style',
]
df_train_columns = df_train.columns.tolist()
numerical = [col for col in df_train_columns if col not in categorical]

print(f'the numerical cols are {numerical}')


## is eninge_cylinder also categorical ,only 9 unique value and the data is not continuous??
def cal_mutual_info(series: pd.Series):
  return mutual_info_score(series, df_train_full.price)


# notice that we are passing the function name in apply method without any arguments, the arguments are all the filtered columns from df[cols]
mutual_info = df_train_full[categorical].apply(cal_mutual_info).round(2)

mutual_info.sort_values(ascending=False)
print(mutual_info)
"""
### Question 4

* Now let's train a logistic regression.
* Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
* Fit the model on the training dataset.
    - To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
    - `model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)`
* Calculate the accuracy on the validation dataset and round it to 2 decimal digits.

"""
### One hot encoding


def prepare_data(df_in: pd.DataFrame, required_features_in: list,
                 dv_args: DictVectorizer, data_type: str):
  data_df = df_in.copy()
  req_feature = required_features_in.copy()
  y = data_df[req_feature].price
  del data_df['price']
  req_feature.remove('price')
  df_dict = data_df[req_feature].to_dict(orient='records')
  if data_type == 'train':
    return dv_args.fit_transform(df_dict), y
  else:
    return dv_args.transform(df_dict), y


# categorical1 = categorical.copy()
# categorical1.remove('model')
# train_dict = df_train[categorical1 + numerical].to_dict(orient='records')
# X_train =dv.fit_transform(train_dict)
# # print(f'the feature names are {dv.get_feature_names_out()} ')
# val_dict = df_val[categorical1 + numerical].to_dict(orient='records')
# X_val = dv.transform(val_dict)
# print(f'the shape of X_train is {X_train.shape} and X_val is {X_val.shape}')

model = LogisticRegression(solver='liblinear',
                           C=10,
                           max_iter=1000,
                           random_state=42)

dv = DictVectorizer(sparse=False)

X_train, y_train = prepare_data(df_train, categorical + numerical, dv, 'train')
model.fit(X_train, y_train)

X_val, y_val = prepare_data(df_val, categorical + numerical, dv, 'val')

y_pred = model.predict_proba(X_val)[:, 1]
df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['pred_label'] = (y_pred >= 0.5).astype(int)
df_pred['actual'] = y_val.astype(int)
# print(y_val.isna().sum())
display(df_pred.head())

# # Calculate accuracy using sklearn
global_accuracy = accuracy_score(y_val, df_pred['pred_label'])

# Calculate accuracy manually
# correct_predictions = sum(true_label == pred_label for true_label, pred_label\
#  in zip(y_val.to_numpy(), df_pred['pred_label'].to_numpy()))
# total_predictions = len(y_val)
# accuracy = correct_predictions / total_predictions

# Print the rounded accuracy
print(round(global_accuracy, 2))

# from sklearn.feature_extraction import DictVectorizer
# dv = DictVectorizer(sparse=False)

# train_dict = df_train.to_dict(orient='records')
# X_train =dv.fit_transform(train_dict)
# all_features = dv.get_feature_names_out()

# categorical_features = [i for i in all_features if '=' in i]
# print(categorical_features)

# categorical_features_dict = {}

# for item in categorical_features:
#     key, value = item.split('=')
#     if key in categorical_features_dict:
#         categorical_features_dict[key].append(value)
#     else:
#         categorical_features_dict[key] = [value]

# for key,value in categorical_features_dict.items():
#     print(f'the number of values in {key} is {len(value)}')
# # print(result)
#
# import seaborn as sns
# import matplotlib.pyplot as plt

# df1 = df_train.groupby(['model']).size().reset_index(name = 'count_individual_models')
# display(df1)

# sns.histplot(df1.count_individual_models,)
# plt.xlabel('Total Count of Individual Models')
# plt.ylabel('Number of datapoints')
"""
### Question 5 

* Let's find the least useful feature using the *feature elimination* technique.
* Train a model with all these features (using the same parameters as in Q4).
* Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
* For each feature, calculate the difference between the original accuracy and the accuracy without the feature. 

Which of following feature has the smallest difference?

- `year`
- `engine_hp`
- `transmission_type`
- `city_mpg`

the accuracy difference without year is -1.0277%
the accuracy difference without engine_hp is 1.7426%
the accuracy difference without transmission_type is -0.6702% 
the accuracy difference without city_mpg is 0.7149%

"""
from collections import defaultdict

evaluation_features = ['year', 'engine_hp', 'transmission_type', 'city_mpg']
accuracy_diff = defaultdict(float)

superset = df_train.columns.tolist()

categorical = [
    'make',
    'model',
    'transmission_type',
    'vehicle_style',
]
numerical = [col for col in superset if col not in categorical]
for feature in evaluation_features:
  required_features = superset.copy()
  required_features.remove(feature)
  dv_selective = DictVectorizer(sparse=False)

  X_train, y_train = prepare_data(df_train, required_features, dv_selective,
                                  'train')
  X_val, y_val = prepare_data(df_val, required_features, dv_selective, 'val')
  model = LogisticRegression(solver='liblinear',
                             C=10,
                             max_iter=1000,
                             random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict_proba(X_val)[:, 1]
  y_pred_label = (y_pred >= 0.5).astype(int)
  accuracy = accuracy_score(y_val, y_pred_label)
  accuracy_diff[feature] = 100 * (global_accuracy - accuracy) / global_accuracy

for key, value in accuracy_diff.items():
  print(f'the accuracy difference without {key} is {round(value,4)}%')
"""
### Question 6

* For this question, we'll see how to use a linear regression model from Scikit-Learn.
* We'll need to use the original column `price`. Apply the logarithmic transformation to this column.
* Fit the Ridge regression model on the training data with a solver `'sag'`. Set the seed to `42`.
* This model also has a parameter `alpha`. Let's try the following values: `[0, 0.01, 0.1, 1, 10]`.
* Round your RMSE scores to 3 decimal digits.

Which of these alphas leads to the best RMSE on the validation set?

- 0
- 0.01
- 0.1
- 1
- 10

wo sparse martix, no convergence and intial results are as below    
for alpha 0 the rmse is 0.207
for alpha 0.01 the rmse is 0.207
for alpha 0.1 the rmse is 0.207
for alpha 1 the rmse is 0.207
for alpha 10 the rmse is 0.207

with sparse matrix and standard scaler 
for alpha 0 the rmse is 0.229
for alpha 0.01 the rmse is 0.222
for alpha 0.1 the rmse is 0.214 ***** ANSWER
for alpha 1 the rmse is 0.23
for alpha 10 the rmse is 0.32

with sparse matrix and no scaling 
for alpha 0 the rmse is 0.255
for alpha 0.01 the rmse is 0.255
for alpha 0.1 the rmse is 0.255
for alpha 1 the rmse is 0.258
for alpha 10 the rmse is 0.336

"""

df = pd.read_csv('data.csv')
required_cols = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle Style', 'highway MPG', 'city mpg', 'MSRP'
]
df = df[required_cols]
# Renaming columns to lower case and replacing space by underscore
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.rename(columns={'msrp': 'price'})
for i in df.columns:
  if df[i].dtype == 'object':
    df[i] = df[i].str.lower().str.replace(' ', '_')

df.fillna(0, inplace=True)
required_split = [60, 20, 20]    # train, val, test

df_train_full, df_test = train_test_split(df,
                                          test_size=required_split[2] / 100,
                                          random_state=42)

df_train, df_val = train_test_split(df_train_full,
                                    test_size=required_split[1] /
                                    (required_split[0] + required_split[1]),
                                    random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


def prepare_regression_data(df_train_arg, df_val_arg, df_test_arg,
                            selected_features):
  df_train1, df_val1, df_test1 = df_train_arg.copy(), df_val_arg.copy(
  ), df_test_arg.copy()
  required_features1 = selected_features.copy()

  y_train1 = np.log1p(df_train1[required_features1].price)
  y_val1 = np.log1p(df_val1[required_features1].price)
  y_test1 = np.log1p(df_test1[required_features1].price)

  del df_train1['price']
  del df_val1['price']
  del df_test1['price']
  required_features1.remove('price')
  numerical_cols_df = [
      col for col in df_train1.columns if df_train1[col].dtype != 'object'
  ]
  scaler = StandardScaler()
  for col in numerical_cols_df:
    df_train1[col] = scaler.fit_transform(df_train1[col].values.reshape(-1, 1))
    df_val1[col] = scaler.transform(df_val1[col].values.reshape(-1, 1))
    df_test1[col] = scaler.transform(df_test1[col].values.reshape(-1, 1))

  dv_reg = DictVectorizer(sparse=True)
  train_dict = df_train1[required_features1].to_dict(orient='records')
  val_dict = df_val1[required_features1].to_dict(orient='records')
  test_dict = df_test1[required_features1].to_dict(orient='records')
  x_train = dv_reg.fit_transform(train_dict)
  x_val = dv_reg.transform(val_dict)
  x_test = dv_reg.transform(test_dict)
  return x_train, y_train1, x_val, y_val1, x_test, y_test1


alphas = [0, 0.01, 0.1, 1, 10]
for alpha in alphas:

  X_train, y_train, X_val, y_val, X_test, y_test = prepare_regression_data(
      df_train, df_val, df_test, categorical + numerical)
  model = Ridge(alpha=alpha, solver='sag', random_state=42)

  model.fit(X_train, y_train)
  y_pred = model.predict(X_val)
  rmse = mean_squared_error(y_val, y_pred, squared=False)
  print(f'for alpha {alpha} the rmse is {round(rmse,3)}')
