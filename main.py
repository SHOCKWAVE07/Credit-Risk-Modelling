import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from scipy.stats import f_oneway
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
import os

print("Initialising...")

# Load the data

# case_study1.xlsx is Banks internal Dataset and case_study2.xlsx is External Dataset from Credit Bureau

a1 = pd.read_excel('case_study1.xlsx')
a2 = pd.read_excel('case_study2.xlsx')

print("Data Loaded")

# Data Preprocessing

df1 = a1.copy()
df2 = a2.copy()

# -99999.00 are nan values

df1 = df1.replace(-99999.000,np.nan)

# Drop the rows with nan values

df1 = df1.dropna()

df2 = df2.replace(-99999.000,np.nan)

# Drop columns with more than 10000 nan values

for col in df2.columns:
    if df2[col].isnull().sum() > 10000:
        df2 = df2.drop(col, axis=1)

# Drop the rows with nan values

df2 = df2.dropna()


# Merge the two dataframes

df = pd.merge(df1,df2,on='PROSPECTID')

# Seperate Categorical and Numerical Columns

cat_cols = df.select_dtypes(include='object').columns   
num_cols = df.select_dtypes(include=['int64','float64']).columns 

# Drop Target Column from Categorical Columns

cat_cols = cat_cols.drop('Approved_Flag')

# Chi-Square Test for Categorical Columns

columns_to_be_kept_categorical = []
for i in cat_cols:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    if pval <= 0.05:
        columns_to_be_kept_categorical.append(i)

num_cols = num_cols.drop('PROSPECTID')


# Multi-Collinearity Check for Numerical Columns using VIF(Variance Inflation Factor)

vif_data = df[num_cols]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range (0,total_columns):
    
    vif_value = variance_inflation_factor(vif_data, column_index)
    
    
    if vif_value <= 6:
        columns_to_be_kept.append( num_cols[i] )
        column_index = column_index+1
    
    else:
        vif_data = vif_data.drop([ num_cols[i] ] , axis=1)

columns_to_be_kept_numerical = []

# ANOVA Test for Numerical Columns

for i in columns_to_be_kept:
    a = list(df[i])  
    b = list(df['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']


    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)

# Final Features using Chi-Square Test and ANOVA Test   

features = columns_to_be_kept_numerical + columns_to_be_kept_categorical    
df = df[features + ['Approved_Flag']]

# Label Encoding Categorical Columns

df.loc[df['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
df.loc[df['EDUCATION'] == '12TH',['EDUCATION']]             = 2
df.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
df.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3

df['EDUCATION'] = df['EDUCATION'].astype(int)

# One Hot Encoding Categorical Columns

df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])

# Splitting the data into train and test

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis = 1 )

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# XGBoost Classifier with Hyperparameter Tuning

'''
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}
'''



xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4,learning_rate= 0.2, max_depth = 3, n_estimators = 200)

xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

# Model Evaluation

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy:.2f}')
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

