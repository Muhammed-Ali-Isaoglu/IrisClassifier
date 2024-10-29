
import pandas as pd 
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# Loading dataset
iris=load_iris()
data=pd.DataFrame(data=iris.data ,columns=iris.feature_names)
data['species']=iris.target

# print(data.head())

# data.info()
# print(data.describe())

sns.pairplot(data, hue='species')
plt.show()

le=LabelEncoder()
data['species']=le.fit_transform(data['species'])

# split dataset:
X=data.drop(columns='species')
Y=data['species']
X_train,X_test,Y_train,Y_test=train_test_split( X, Y ,test_size=0.2, random_state=42)

# train the model :
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)

# Evaluating the model:
Y_pred=knn.predict(X_test)
accuracy=accuracy_score(Y_pred,Y_test)
print(f'Accuracy of K_mean: {accuracy:.2f}')

# another model :
dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)
Y_pred_dt=dt.predict(X_test)
accuracy_dt=accuracy_score(Y_pred_dt,Y_test)
print(f"Accuracy for Decision Tree Classifier: {accuracy_dt}")


cm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm ,annot=True, fmt='d',cmap='Blues')
plt.xlabel=('predicted')
plt.ylabel=('actuall')
plt.show()

# cm_dt=confusion_matrix(Y_test,Y_pred_dt)
# sns.heatmap(cm_dt ,annot=True, fmt='d' ,cmap='Reds')
# plt.xlabel=("predicate")
# plt.ylabel=('Actuall')
# plt.show()

