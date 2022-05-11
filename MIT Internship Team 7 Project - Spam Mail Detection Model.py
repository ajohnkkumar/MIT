#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv('mail_data.csv')


# In[3]:


print(raw_mail_data)


# In[4]:


raw_mail_data.info


# In[5]:


raw_mail_data['Category'].unique()


# In[6]:


raw_mail_data.groupby('Category').count()


# In[7]:


values=(4825,747)
labels=('Ham Mails', 'Spam Mails')


# In[8]:


plt.pie(values,labels=labels)


# In[9]:


# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[10]:


# printing the first 5 rows of the dataframe
mail_data.head()


# In[11]:


# checking the number of rows and columns in the dataframe
mail_data.shape


# In[12]:


#Label Encoding
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[13]:


# separating the data as texts and label
X = mail_data['Message']
Y = mail_data['Category']


# In[14]:


print(X)


# In[15]:


print(Y)


# In[16]:


#Split to training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[17]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[18]:


#Feature Extraction
# transform the text data to feature vectors that can be used as input to the Logistic regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[19]:


print(X_train)


# In[20]:


print(X_train_features)


# In[21]:


#Logistic Regression model
model = LogisticRegression()


# In[22]:


# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)


# In[23]:


X_train_features.shape


# In[24]:


Y_train.shape


# In[25]:


#Evaluating the trained model
# prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[26]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[27]:


# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[28]:


X_test_features.shape


# In[29]:


print('Accuracy on test data : ', accuracy_on_test_data)


# In[30]:


#building a prediction model
input_mail = ["Subject: enron / hpl actuals for august 28 , 2000 teco tap 20 . 000 / enron ; 120 . 000 / hpl gas daily ls hpl lsk ic 20 . 000 / enron"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[32]:


#input_mail = ["You are a winner U have been specially selected 2 receive Rs. 1000 cash or a 4* holiday (flights inc) speak to a live operator 2 claim 0871277810810"]
input_mail = ["Hi Sir! glad to hear back from you. Hope we'll meet on Monday to finalise the deal."]
# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[ ]:




