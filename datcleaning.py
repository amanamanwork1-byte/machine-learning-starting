import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("insurance.csv")

df_cleaned = df.copy()
#print(df_cleaned.drop_duplicates(inplace=True))
#print(df_cleaned['sex'].value_counts())
df_cleaned['sex'] = df_cleaned['sex'].map({"male": 0 , "female": 1})

df_cleaned['smoker'] = df_cleaned['smoker'].map({"no" : 0,"yes" : 1})
df_cleaned.rename(columns={
    'sex' :'is_female',
    'smoker': 'is_smoker'
                          },inplace = True)
df_cleaned = pd.get_dummies(df_cleaned,columns = ['region'],drop_first=True)
df_cleaned = df_cleaned.astype(int)
print(df_cleaned.head())
