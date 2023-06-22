import pandas as pd
df=pd.read_csv("/Users/sakshikarande/Desktop/titanic_project_internship/titanic.csv")
print(df)

#defining X and Y
X = df.drop('Name',axis=1)
X = X.drop('Survived',axis=1)
X = X.drop('Ticket',axis=1)
X = X.drop('Parch',axis=1)
X = X.drop('Cabin',axis=1)
Y = df['Survived']

print(X)
print(Y)

#Categorical to numerical
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df['Sex'])
X['Sex'] = le.transform(X['Sex'])

le = LabelEncoder()
le.fit(df['Embarked'])
X['Embarked'] = le.transform(df['Embarked'])
print(X)


#checking for missing data and refilling it
#print(df.isnull().sum())

X['Age'].fillna((X['Age'].mean()), inplace=True)
X['Fare'].fillna((X['Fare'].mean()), inplace=True)

print(X.isnull().sum())

#checking imbalance in data

from collections import Counter
#print(Counter(Y))
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=1)
X, Y = ros.fit_resample(X,Y)
print(Counter(Y))

#Identifying outliers by plotting

import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(X['Fare'])
plt.show()

#dealing with outliers using inquantile range

print(X['Fare'])
Q1 = X['Fare'].quantile(0.25)
Q3 = X['Fare'].quantile(0.75)

IQR = Q3-Q1
print(IQR)

upper = Q1 + 1.5*(IQR)
lower = Q3 - 1.5*(IQR)

print(upper)
print(lower)

out1 = X[X['Fare']<lower].values
out2 = X[X['Fare']>upper].values

X['Fare'].replace(out1,lower,inplace=True)
X['Fare'].replace(out2,upper,inplace=True)

print(X['Fare'])

#Train data / feature selection
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(4).plot(kind='bar')
plt.show()

#Prediction using Logistic regression

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

ran=RandomForestClassifier()
logr = LogisticRegression()
pca = PCA(n_components=5)

pca.fit(X)
X = pca.transform(X)

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size=0.3)
logr.fit(X_train,Y_train)

y_pred = logr.predict(X_test)
print(accuracy_score(Y_test,y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size=0.3)
ran.fit(X_train,Y_train)

y_pred = ran.predict(X_test)
print(accuracy_score(Y_test,y_pred))
