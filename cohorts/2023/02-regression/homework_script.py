import pandas as pd
import seaborn as sns
import numpy as np
import os
os.chdir('cohorts/2023/02-regression/')
df = pd.read_csv('housing.csv')
df.head()
#list the column headers
print(f'the headers are {df.columns.values}')


#plot the histogram of the target variable
sns.histplot(df['median_house_value'])
## Tailed distribution is observed in the target variable

### Drop the values
df = df[(df['ocean_proximity'] == '<1H OCEAN') | (df['ocean_proximity'] == 'INLAND')]


selected_cols = [
    'latitude',
    'longitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'median_house_value',
]
df =df[selected_cols]
df.head()



 
# ### Question 1
# 
# There's one feature with missing values. What is it?
# 
# * `total_rooms`
# * `total_bedrooms`
# * `population`
# * `households`
# 


df.isnull().sum()

#Answer to Q1 : total_bedrooms has null values

 
# ### Question 2
# 
# What's the median (50% percentile) for variable `'population'`?

required_median = df['population'].median()
print(f'the median of population is {required_median}')

def create_split(seed,df):
    np.random.seed(seed)
    idx = np.arange(df.shape[0])
    np.random.shuffle(idx)
    ntrain = int(len(idx) * 0.6)
    nval = int(len(idx) * 0.2)
    ntest = len(idx) - ntrain - nval

    # print(f'IDX: the number of training samples is {ntrain}', f'the number of validation samples is {nval}', f'the number of test samples is {ntest}')


    df_train = df.iloc[idx[:ntrain]]
    df_val = df.iloc[idx[ntrain:ntrain+nval]]
    df_test = df.iloc[idx[ntrain+nval:]]

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    # print(f'DF: the number of training samples is {df_train.shape[0]}', f'the number of validation samples is {df_val.shape[0]}', f'the number of test samples is {df_test.shape[0]}')
    y_train = np.log1p(df_train['median_house_value'].values)
    y_val = np.log1p(df_val['median_house_value'].values)
    y_test = np.log1p(df_test['median_house_value'].values)


    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']

    return df_train,df_val,df_test,y_train,y_val,y_test


df_train,df_val,df_test,y_train,y_val,y_test = create_split(42,df)


def prepare_X_fill_zero(df):
    df = df.copy()
    features = df.columns.tolist()
    # for name, values in categorical.items():
    #     for value in values:
    #         df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
    #         features.append('%s_%s' % (name, value))
    #     df.drop(name, axis=1, inplace=True)
    #     features.remove(name)
    df_num = df[features]
    df_num = df_num.fillna(0)
    # print(df_num.isnull().sum())
    X = df_num.values
    return X
prepare_X_fill_zero(df_train)


def prepare_X_fill_mean(df,column_name):
    df = df.copy()
    features = df.columns.tolist()
    # for name, values in categorical.items():
    #     for value in values:
    #         df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
    #         features.append('%s_%s' % (name, value))
    #     df.drop(name, axis=1, inplace=True)
    #     features.remove(name)
    df_num = df[features]
    
    # Fill NaN values with the mean of the specified column
    df_num[column_name] = df_num[column_name].fillna(df_num[column_name].mean())
    
    # print(df_num.isnull().sum())
    X = df_num.values
    return X
prepare_X_fill_mean(df_train,'total_bedrooms')

 

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    # XTX = X.T.dot(X)
    XTX = np.dot(X.T, X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)



#### Question 3 : Train without regularization and (fill zero and fill mean) 

#### Using `fillna(0)` for the missing value
X_train = prepare_X_fill_zero(df_train)

w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X_fill_zero(df_val)
y_pred = w0 + X_val.dot(w)
rmse_with_zero_wo_reg = np.round(rmse(y_val, y_pred),2)
print(f'the value of rmse without reg and fill zero is {rmse_with_zero_wo_reg}')
#the value of rmse without reg and fill zero is 0.34
 
#### Using `fillna() with mean` for the missing value


X_train = prepare_X_fill_mean(df_train,'total_bedrooms')

w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X_fill_mean(df_val,'total_bedrooms')
y_pred = w0 + X_val.dot(w)
rmse_with_mean_wo_reg = np.round(rmse(y_val, y_pred),2)
print(f'the value of rmse without reg and fill mean is {rmse_with_mean_wo_reg}')
#the value of rmse without reg and fill mean is 0.34 



#### Question 4 : Train with regularization and fill zero


regularization = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]


def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = np.dot(X.T, X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


rmse_with_regularization = {}
for r in regularization:
    X_train = prepare_X_fill_zero(df_train)

    w0, w = train_linear_regression_reg(X_train, y_train,r)

    X_val = prepare_X_fill_zero(df_val)
    y_pred = w0 + X_val.dot(w)
    
    rmse_val = np.round(rmse(y_val, y_pred),2)
    rmse_with_regularization[r] = rmse_val

for key, value in rmse_with_regularization.items():
    print(f'for regularization {key} the rmse is {value}')

''' Question 4
for regularization 0 the rmse is 0.34
for regularization 1e-06 the rmse is 0.34
for regularization 0.0001 the rmse is 0.34
for regularization 0.001 the rmse is 0.34
for regularization 0.01 the rmse is 0.34
for regularization 0.1 the rmse is 0.34
for regularization 1 the rmse is 0.34
for regularization 5 the rmse is 0.35
for regularization 10 the rmse is 0.35
'''


### Question 5 : Try with different seeds ,fill zero and without regularization
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_with_variable_seed = {}
for seed in seeds:
    df_train,df_val,df_test,y_train,y_val,y_test = create_split(seed,df)
    X_train = prepare_X_fill_zero(df_train)

    w0, w = train_linear_regression(X_train, y_train)

    X_val = prepare_X_fill_zero(df_val)
    y_pred = w0 + X_val.dot(w)
    
    rmse_val = rmse(y_val, y_pred)
    rmse_with_variable_seed[seed] = rmse_val
for key, value in rmse_with_variable_seed.items():
    print(f'for seed {key} the rmse is {value}')

std_dev = np.std(list(rmse_with_variable_seed.values()))
print(f'the standard deviation is {np.round(std_dev,3)}')


### Question 6 : Try with seed 9  and regularization 0.001 with both train and val merged

df_train,df_val,df_test,y_train,y_val,y_test = create_split(seed,df)
df_full_train = pd.concat([df_train, df_val])
X_train = prepare_X_fill_zero(df_full_train)
y_train = np.concatenate([y_train, y_val])
w0, w = train_linear_regression_reg(X_train, y_train,r=0.001)

X_val = prepare_X_fill_zero(df_test)
y_pred = w0 + X_val.dot(w)

rmse_final = rmse(y_test, y_pred)

print(f'the final rmse value with seed 9 and regularization 0.001 is {np.round(rmse_final,2)} ')