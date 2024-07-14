import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #split training and testing data set
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #check performance of model


#loading the data set to a pandas dataframe
credit_card_data=pd.read_csv('/creditcard.csv')


#first 5 rows of the dataset,
credit_card_data.head()


#loading last 5 rows of data
credit_card_data.tail()


#dataset information->gives data type, rows, columns, missing values(null)
credit_card_data.info()


#checking the number of missing values in each column
credit_card_data.isnull().sum()


#distribution of legit transaction and fraudulent transaction, 0=legit and 1=fraudlent. This shows dataset is unabalnced because 99% of data is in one class
credit_card_data['Class'].value_counts()


#Separating Legit(0) & Fraudulent(1) transactions
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]


print(legit.shape)
print(fraud.shape)


#statistical measures of the data
legit.Amount.describe()


fraud.Amount.describe()


#compare the values for both kinds of transactions
credit_card_data.groupby('Class').mean()


#random sampling of data, random 492 legit transactions are selected
legit_sample=legit.sample(n=492)


#axis=0 Concatenation row-wise; axis=1 Concatenation column-wise
new_dataset=pd.concat([legit_sample,fraud],axis=0)


new_dataset.head()


new_dataset.tail()


new_dataset['Class'].value_counts()


new_dataset.groupby('Class').mean()


X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']
print(X)


print(Y)


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


print(X.shape,X_train.shape,X_test.shape)


#loading one instance of Logistic Regression Model in our variable
model=LogisticRegression()


#training the Logistic Regression Model with training data
model.fit(X_train,Y_train)


#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


#accuracy on testing data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Testing data : ',testing_data_accuracy)
