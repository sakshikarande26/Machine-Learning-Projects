import pandas as pd
from sklearn.datasets import load_iris

irs = load_iris()
print(irs)
print(irs.keys())
print(irs.data)
print(irs.feature_names)
print(irs.target)
print(irs.target_names)
print(irs.DESCR)

df=pd.read_csv("C:/Users/admin/Desktop/sakshi_internship_dataset/IRIS.csv")

print(df)
print(df.head(10))
print(df.tail())
print(df.columns.values)
print(df.describe())

print(df.isnull().sum())
print(df.notnull().sum())

df['sepal_width'].fillna((df['sepal_width'].median()), inplace=True)
df['sepal_length'].fillna((df['sepal_length'].median()), inplace=True)
df['petal_width'].fillna((df['petal_width'].median()), inplace=True)
print(df.isnull().sum())

a = df['species'] == 'Iris-setosa'.sum()
print(a)
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X,Y)
