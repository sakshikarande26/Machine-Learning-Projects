import pandas as pd
df=pd.read_csv("/Users/sakshikarande/Desktop/census_income_internship/adult.csv")
#print(df)

#identifying X and Y
X = df.drop('age', axis=1)
X = X.drop('fnlwgt', axis=1)
X = X.drop('education.num', axis=1)
X = X.drop('marital.status', axis=1)
X = X.drop('relationship', axis=1)
X = X.drop('capital.gain', axis=1)
X = X.drop('capital.loss', axis=1)
X = X.drop('income', axis=1)
Y = df['income']

#print(X)
#print(Y)

#replace ? with entry
X['workclass'] = X['workclass'].replace('?','Private')
X['occupation'] = X['occupation'].replace('?','Prof-specialty')
X['native.country'] = X['native.country'].replace('?','Mexico')
#print(X)

#categorical to numerical
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(X['race'])
X['race'] = le.transform(X['race'])

le.fit(X['sex'])
X['sex'] = le.transform(X['sex'])

le.fit(X['workclass'])
X['workclass'] = le.transform(X['workclass'])

le.fit(X['education'])
X['education'] = le.transform(X['education'])

le.fit(X['occupation'])
X['occupation'] = le.transform(X['occupation'])

le.fit(X['native.country'])
X['native.country'] = le.transform(X['native.country'])

#print(X)

#check for null values
#print(X.isnull().sum())

#check imbalance
from collections import Counter
#print(Counter(Y))

from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=1)
X, Y = ros.fit_resample(X,Y)
#print(Counter(Y))

#identifying outliers
import matplotlib.pyplot as plt
import seaborn as sns
#sns.boxplot(X['hours.per.week'])
#plt.show()

#dealing with outliers using interquantile range
print(X['hours.per.week'])
Q1 = X['hours.per.week'].quantile(0.25)
Q3 = X['hours.per.week'].quantile(0.75)

IQR = Q3-Q1
print(IQR)

upper = Q1 + 1.5*(IQR)
lower = Q3 - 1.5*(IQR)

print(upper)
print(lower)

out1 = X[X['hours.per.week']<lower].values
out2 = X[X['hours.per.week']>upper].values

X['hours.per.week'].replace(out1,lower,inplace=True)
X['hours.per.week'].replace(out2,upper,inplace=True)

print(X['hours.per.week'])
sns.boxplot(X['hours.per.week'])
plt.show()

#feature selection
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(7).plot(kind='barh')
plt.show()

#train model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ran = RandomForestClassifier()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.3)
ran.fit(X_train, Y_train)

y_pred = ran.predict(X_test)
print(accuracy_score(Y_test, y_pred))
