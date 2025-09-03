#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("heartdata.csv")
dataset.describe()

#understanding the attributes



#analysing the target variable
y = dataset["target"]
target_temp = dataset.target.value_counts()
#print(target_temp)

#ignoring the warnings

import warnings
warnings.simplefilter('ignore')
#printing % of patients with & without heart problem
print("\nPercentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
  
obj = dataset.select_dtypes(include = 'object')
  
non_obj = dataset.select_dtypes(exclude = 'object')

for i in range(0, obj.shape[1]):
    obj.iloc[:,i]= encoder.fit_transform(obj.iloc[:,i])

    final_data = pd.concat([obj, non_obj], axis= 1)

#spliting the data
from sklearn.neighbors import KNeighborsClassifier
  
from sklearn.model_selection import train_test_split
#model fitting
from sklearn.metrics import accuracy_score
  
x = final_data.drop(['target'], axis=1)
  
y = final_data['target']

#training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=42)


#using KNN algo
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
KNeighborsClassifier(n_neighbors=7)
y_pred_knn=knn.predict(x_test)
score_knn = round(accuracy_score(y_pred_knn,y_test)*100,2)
print("The accuracy score achieved using KNN is: "+str(score_knn)+" %\n")

#using logistic algorithm

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)
score_lr = round(accuracy_score(y_pred_lr,y_test)*100,2)
print("\nThe accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, y_pred)
rf_accuracy
  
rf_model = RandomForestClassifier().fit(x_train, y_train)
  
y_pred = rf_model.predict(x_test)
score_RF = round(accuracy_score(y_pred,y_test)*100,2) 
print("The accuracy score achieved using RandomForest is: "+str( score_RF )+" %\n")

#output final score
scores = [score_lr,score_knn,score_RF]
algorithms = ["Logistic Regression","K-Nearest Neighbors"," Random Forest"]    

#finding the best model according to the given dataset
if (score_lr > score_knn) and (score_lr>score_RF) :
    print("Hence Logistic Regression works better here...")
elif (score_RF>score_lr)and ( score_RF>score_knn):
    print(" RandomForest work better...")
elif(score_lr== score_RF):
    print("Logistic accuracy and Randomforest accuracy equal")
elif(score_lr== score_knn):          
    print("Logistic accuracy and KNN accuracy equal")
          
else :
    print("Hence K-Nearest Neighbour model works better here...")


#plotting the graph
sns.set(rc={'figure.figsize':(5,5)})
x = plt.xlabel("Algorithms")
y = plt.ylabel("Accuracy score")
sns.barplot(x = algorithms, y = scores, alpha = 0.8, palette="viridis")
plt.title("Algorithm v/s Accuracy Score Plotting", fontsize = 10) 
print(plt.show())


models = ['Random Forest', 'Logistic Regression',' K-Nearest Neighbour']
accuracies = [score_RF, score_lr,score_knn]
bars = plt.bar(models, accuracies, color=colors)



    
