import pandas as pd
df=pd.read_csv("C:/Users/admin/Desktop/walmartsales_sakshi/train.csv")
#print(df)

#defining X and Y

X = df.drop('Date', axis=1)
X = X.drop('Weekly_Sales', axis=1)
Y = df['Weekly_Sales']
#print(X)
#print(Y)

#chceking null values
#print(X.isnull().sum())

#Categorical to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(X['IsHoliday'])
X['IsHoliday'] = le.transform(X['IsHoliday'])
#print(X)

#feature selection
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model=ExtraTreesRegressor()
model.fit(X, Y)
model.feature_importances_
feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(3).plot(kind='barh')
plt.show()

#training model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

gbr = GradientBoostingRegressor()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
gbr.fit(X_train,Y_train)

y_pred = gbr.predict(X_test)
print(r2_score(Y_test,y_pred))
