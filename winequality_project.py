import sklearn as skl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/sakshikarande/Desktop/wine quality/winequalityN.csv")
'''
print(df)
'''

#Defining X and Y
X = df.drop("quality", axis=1)
Y = df['quality']
print(X)
print(Y)

#Null Values
#print(df.isnull().sum())

#filling null values
X['fixed acidity'].fillna((X['fixed acidity']).mean(), inplace=True)
X['volatile acidity'].fillna((X['volatile acidity']).mean(), inplace=True)
X['citric acid'].fillna((X['citric acid']).mean(), inplace=True)
X['residual sugar'].fillna((X['residual sugar']).mean(), inplace=True)
X['chlorides'].fillna((X['chlorides']).mean(), inplace=True)
X['pH'].fillna((X['pH']).mean(), inplace=True)
X['sulphates'].fillna((X['sulphates']).mean(), inplace=True)
print(df.isnull().sum())

#categorical to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(X['type'])
X['type'] = le.transform(X['type'])
print(X)

#identifying outliers
sns.boxplot(df['chlorides'])
plt.show()

#imbalance in dataset
from collections import Counter
print(Counter(Y))
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
print(Counter(Y))

#dealing with outliers

print(X['chlorides'])
Q1 = X['chlorides'].quantile(0.25)
Q3 = X['chlorides'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print(upper)
print(lower)

out1 = X[X['chlorides'] < lower].values
out2 = X[X['chlorides'] > upper].values

X['chlorides'].replace(out1, lower, inplace=True)
X['chlorides'].replace(out2, upper, inplace=True)

print(X['chlorides'])

#feature selection
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X, Y)
model.feature_importances_
feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(13).plot(kind='barh')
plt.show()
#training model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import r2_score

ran = RandomForestClassifier()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.3)
ran.fit(X_train, Y_train)
y_pred = ran.predict(X_test)

print(r2_score(Y_test,y_pred))
