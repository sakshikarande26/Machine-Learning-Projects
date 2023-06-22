import pandas as pd
df = pd.read_csv("/Users/sakshikarande/Desktop/boaton_housing_internship/HousingData.csv")
#print(df)

#Defining X and Y
X = df.drop('ZN', axis=1)
X = X.drop('MEDV', axis=1)
X = X.drop('NOX', axis=1)
X = X.drop('B', axis=1)
X = X.drop('LSTAT', axis=1)
Y = df['MEDV']

#print(X)
#print(Y)

#checking for missing data
#print(X.isnull().sum())

X['CRIM'].fillna((X['CRIM'].mean()), inplace=True)
X['INDUS'].fillna((X['INDUS'].mean()), inplace=True)
X['CHAS'].fillna((X['CHAS'].mean()), inplace=True)
X['AGE'].fillna((X['AGE'].mean()), inplace=True)
#print(X.isnull().sum())

#Feature selection
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

model = ExtraTreesRegressor()
model.fit(X, Y)
model.feature_importances_
feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(9).plot(kind='bar')
plt.show()

#train model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

gbr = GradientBoostingRegressor()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.3)
gbr.fit(X_train, Y_train)

y_pred = gbr.predict(X_test)
print(r2_score(Y_test, y_pred))
