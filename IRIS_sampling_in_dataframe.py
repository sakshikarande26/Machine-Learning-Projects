import pandas as pd
from sklearn.datasets import load_iris
from collections import Counter

df=pd.read_csv("C:/Users/admin/Desktop/sakshi_internship_dataset/IRIS.csv")


X = df.drop('species', axis=1)
Y = df['species']
print(X)
print(Y)

irs = load_iris()
print(irs)
print(irs.keys())
print(irs.data)
print(irs.feature_names)
print(irs.target)
print(irs.target_names)
print(irs.DESCR)

print(df)
print(df.head(10))
print(df.tail())
print(df.columns.values)
print(df.describe())

print(Counter(Y))
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=1)
X, Y = ros.fit_resample(X,Y)
print(Counter(Y))


'''from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X,Y)'''


