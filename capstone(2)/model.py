import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.model_selection import train_test_split

# Load dataset
df= pd.read_csv(r"C:\Users\HP\Documents\capstone(2)\depression_data.csv")

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
numerical_columns = list(df.dtypes[df.dtypes == 'number'].index)

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)

# Only numeric columns for LinearRegression
y_train = df_train['family_history_of_depression']
y_test = df_test['family_history_of_depression']
y_val = df_val['family_history_of_depression']

X_train = df_train.drop(['family_history_of_depression'], axis=1)
X_test = df_test.drop(['family_history_of_depression'], axis=1)
X_val = df_val.drop(['family_history_of_depression'], axis=1)




X_train = X_train.select_dtypes(exclude=['object'])
X_test  = X_test.select_dtypes(exclude=['object'])
X_val   = X_val.select_dtypes(exclude=['object'])





# Train pipeline
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
# Save model
joblib.dump(baseline_model, "depression_model.pkl")
print("Model saved at: depression_model.pkl")
