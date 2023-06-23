import pandas as pd
df=pd.read_csv("C:/Users/admin/Desktop/sakshi_internship_dataset/IRIS.csv")

#X and Y

X = df.drop('species', axis=1)
Y = df['species']
print(X)
print(Y)

'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['specs','score']

print(featureScores)

#feature engineerng
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(4).plot(kind='bar')
plt.show()



#RandomForestClassifier for outliers


from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

rf = RandomForestClassifier()
df['sepa_length'] = pd.cut(df['sepal_length'],3,labels=['0','1','2'])
df['sepal_width'] = pd.cut(df['sepal_width'],3,labels=['0','1','2'])
df['petal_length'] = pd.cut(df['petal_length'],3,labels=['0','1','2'])
df['petal_length'] = pd.cut(df['petal_length'],3,labels=['0','1','2'])

print(df)

#Label Decoder

print(Y)
le=LabelEncoder()
'''

#plotting outliers to identify

from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['sepal_length'])
plt.show()

#quantile range outliers dealing

print(df['sepal_length'])
Q1 = df['sepal_length'].quantile(0.25)
Q3 = df['sepal_width'].quantile(0.75)

IQR = Q3-Q1
print(IQR)

upper = Q1 + 1.5*(IQR)
lower = Q3 -  1.5*(IQR)

print(upper)
print(lower)

out1 = df[df['sepal_length']<lower].values
out2 = df[df['sepal_width']>upper].values

df['sepal_length'].replace(out1,lower,inplace=True)
df['sepal_width'].replace(out2,upper,inplace=True)

print(df['sepal_length'])

#PCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr = LogisticRegression()
pca = PCA(n_components=2)

X = df.drop('species',axis=1)
Y = df['species']

pca.fit(X)
X = pca.transform(X)

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size=0.3)
logr.fit(X_train,Y_train)
y_pred = logr.predict(X_test)
print(accuracy_score(Y_test,y_pred))


