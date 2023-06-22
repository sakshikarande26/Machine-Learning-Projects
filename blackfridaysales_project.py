import pandas as pd
df=pd.read_csv("/Users/sakshikarande/Desktop/blackfridaysale_internship/train.csv")
#print(df)

#defining X and Y
X = df.drop('User_ID', axis=1)
X = X.drop('Product_ID', axis=1)
X = X.drop('Occupation', axis=1)
X = X.drop('Stay_In_Current_City_Years', axis=1)
X = X.drop('Age', axis=1)
X = X.drop('Marital_Status', axis=1)
X = X.drop('Purchase', axis=1)
Y = df['Purchase']
#print(X)
#print(Y)

#checking for null values
X['Product_Category_2'].fillna((X['Product_Category_2'].mean()), inplace=True )
X['Product_Category_3'].fillna((X['Product_Category_3'].mean()), inplace=True )
#print(X.isnull().sum())

#categoricAL TO numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(X['Gender'])
X['Gender'] = le.transform(X['Gender'])

le = LabelEncoder()
le.fit(X['City_Category'])
X['City_Category'] = le.transform(X['City_Category'])

#print(X)

#feature selection
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

model=ExtraTreesRegressor()
model.fit(X, Y)
model.feature_importances_

feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(6).plot(kind='barh')
plt.show()

#train model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

gbr = GradientBoostingRegressor()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
gbr.fit(X_train,Y_train)

y_pred = gbr.predict(X_test)
print(r2_score(Y_test,y_pred))

