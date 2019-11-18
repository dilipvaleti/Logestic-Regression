#!/usr/bin/env python
# coding: utf-8

# # SUV Prediction

# In[61]:


import numpy as np
import matplotlib as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


dataset=pd.read_csv("SUV_Predic.csv")
dataset.head()


# In[63]:


x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
y_test.shape


# In[66]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[67]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[68]:


y_pred=classifier.predict(X_test)


# In[71]:


from sklearn.metrics import accuracy_score


# In[70]:


accuracy_score(y_test,y_pred)*100


# In[72]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

