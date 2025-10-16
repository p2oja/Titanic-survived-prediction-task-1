# Titanic-survived-prediction-task-1
TITANIC SURVIVAL PREDICTION  Use the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project with readily available data. The dataset typically used for this project contains information about individual passengers, such as their age, gender.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

titanic_data = pd.read_csv("/content/Titanic-Dataset (1).csv")

#prining the frist 5 row of the dataframe 
titanic_data.head()

titanic_data.shape
#getting the some inforamtion about the data 
titanic_data.info()
#cheak the number of missing value in each colume
titanic_data.isnull().sum()

#drop the cabin column from the dataframe 
titanic_data = titanic_data.drop(columns='Cabin',axis=1)
#replacing the missing value in 'age' column with main value
titanic_data ['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
#finding the mode the value of 'embarked' coloume
print(titanic_data['Embarked'].mode()[0])
#replacing the misssing value in Embarked colume with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0] , inplace=True)
titanic_data.isnull().sum()
titanic_data.describe()
#finiinding the  number of survide an dnot survide
titanic_data['Survived'].value_counts()
sns.set()
sns.countplot(x='Survived', data=titanic_data, palette='pastel')
plt.title("Survival Count (0 = Not Survived, 1 = Survived)")
plt.show()
titanic_data['Sex'].value_counts()
sns.countplot(x='Sex', data=titanic_data)
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title("Number of Survivors by Gender")
plt.show()
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title("Number of Survivors by Gender")
plt.show()
sns.countplot(x='Pclass', data=titanic_data)
plt.title("Passenger Count by Class")
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title("Number of Survivors by Gender")
plt.show()
titanic_data['Sex'].value_counts()
titanic_data['Embarked'].value_counts()
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
titanic_data.head()
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)






  
 
      
    

    
  
