import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings 

warnings.filterwarnings('ignore')

df = pd.read_csv("insurance.csv")

# ------ EDA ------( DATA ANYALYSIS)
#print(df.describe())
#print(df.info())
#print(df.shape)
#print(df.isnull().sum())
numeric_columns = df.select_dtypes(include=np.number).columns.tolist() #.colums gives name of the coulm and .todlist converts it into list
print(numeric_columns)
for col in numeric_columns:
   # plt.figure(figsize=(10,5))
    #sns.histplot(df[col], kde=True, bins=30)
    #plt.title(f'Distribution of {col}')
    #plt.xlabel(col)
    #plt.ylabel('Frequency')
    #plt.show()
    #plt.figure(figsize=(10,5))
    #sns.boxplot(x=df[col])
    #plt.show()
    plt.figure(figsize=(10,5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.show()
