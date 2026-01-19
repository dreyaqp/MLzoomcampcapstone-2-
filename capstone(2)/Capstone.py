#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor


# In[2]:


df= pd.read_csv(r"C:\Users\HP\Documents\capstone(2)\depression_data.csv")


# In[3]:


df.isnull().sum()


# In[4]:


df.head().T


# In[5]:


df.columns = df.columns.str.lower().str.replace(' ', '_')


# In[6]:


df.dtypes


# In[7]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)


# In[8]:


categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
numerical_columns = list(df.dtypes[df.dtypes == 'number'].index)


# In[9]:


len(df_test),len(df_train),len(df_val)


# In[10]:


y_train = df_train['family_history_of_depression']
y_test = df_test['family_history_of_depression']
y_val = df_val['family_history_of_depression']


# In[11]:


X_train = df_train.drop(['family_history_of_depression'], axis=1)
X_test = df_test.drop(['family_history_of_depression'], axis=1)
X_val = df_val.drop(['family_history_of_depression'], axis=1)



# In[12]:


X_train = X_train.select_dtypes(exclude=['object'])
X_test  = X_test.select_dtypes(exclude=['object'])
X_val   = X_val.select_dtypes(exclude=['object'])


# ## Statistics summary

# In[13]:


df.describe()


# In[14]:


df.info()


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


for col in numerical_columns:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[17]:


for col in numerical_columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[18]:


target = 'family_history_of_depression'  
for col in numerical_columns:
    if col != target:
        plt.figure()
        sns.scatterplot(x=df[col], y=df[target])
        plt.title(f'{col} vs {target}')
        plt.show()


# In[19]:


X_train_dicts = X_train.to_dict(orient='records')
X_val_dicts = X_val.to_dict(orient='records')
X_test_dicts = X_test.to_dict(orient='records')


# ## Linear Regression

# In[20]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)
y_val   = le.transform(y_val)


# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

baseline_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

baseline_model.fit(X_train, y_train)


# In[22]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = baseline_model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5              
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

rmse, mae, r2


# ## Decision Tree Regressor

# In[23]:


from sklearn.tree import DecisionTreeRegressor

dt_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(random_state=42))
])

dt_model.fit(X_train, y_train)


# In[36]:


y_pred = dt_model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5               
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

rmse, mae, r2


# ## Random Forest Regressor

# In[25]:


from sklearn.ensemble import RandomForestRegressor

rf_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)


# In[35]:


y_pred = rf_model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5               
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

rmse, mae, r2


# ## Gradient Boosting Code

# In[31]:


from sklearn.ensemble import GradientBoostingRegressor

gb_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

gb_model.fit(X_train, y_train)


# In[34]:


y_pred = gb_model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5             
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

rmse, mae, r2


# ## KNN Regressor

# In[29]:


from sklearn.neighbors import KNeighborsRegressor

knn_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
])

knn_model.fit(X_train, y_train)


# In[32]:


y_pred = knn_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5               
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

rmse, mae, r2


# In[ ]:




