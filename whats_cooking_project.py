import pandas as pd
df=pd.read_csv("/Users/sakshikarande/Desktop/whats_cooking_internship/sample_submission.csv")
#print(df)

#defining X and Y
X = df.drop('cuisine', axis=1)
Y = df['cuisine']
#print(X)
#print(Y)

#identifying outliers
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(X['id'])
plt.show()

#feature selection
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(4).plot(kind='bar')
plt.show()

#train data
'''