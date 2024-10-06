import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

dataset=pd.read_csv("iris.csv")
dataset.describe()

X=dataset.iloc[:,[0,1,2,3]].values
y=dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fir_transform(X_train)
X_test=sc.transform(X_test)

classifier=LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
probs_y=classifier.predict_proba(X_test)
from sklearn.metrix import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

ax=plt.axes()
df_cm=cm
sns.heatmap(df_cm,annot=True,annot_kws={"size":30},fmt='d',cmap="Blues",ax=ax)
ax.set_title('confusion matrix')
plt.show()