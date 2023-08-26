#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Iris.csv')
df.head()


# In[3]:


df = df.drop(columns = ['Id'])
df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df['Species'].value_counts()


# In[7]:


df.isnull().sum()


# In[8]:


df['SepalLengthCm'].hist()


# In[9]:


df['SepalWidthCm'].hist()


# In[10]:


df['PetalLengthCm'].hist()


# In[11]:


df['PetalWidthCm'].hist()


# In[12]:


colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[13]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[14]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[15]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[16]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[17]:


df.corr()


# In[18]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# In[19]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[20]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[21]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[22]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[23]:


model.fit(x_train, y_train)


# In[24]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[26]:


model.fit(x_train, y_train)


# In[27]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[ ]:




