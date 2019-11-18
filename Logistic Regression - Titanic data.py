#!/usr/bin/env python
# coding: utf-8

# # Logistic egression

# In[58]:


import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math

titanic_data=pd.read_csv("titanic.csv")
titanic_data.head(1)


# In[59]:


print("the number of passengers are", len(titanic_data.index))


# ## Analyzing data

# In[60]:


sns.countplot(x="survived",data=titanic_data)


# In[61]:


sns.countplot(x="survived",hue='sex',data=titanic_data)


# In[62]:


sns.countplot(x="survived",hue='pclass',data=titanic_data)


# In[63]:


titanic_data['age'].plot.hist()


# In[64]:


titanic_data['fare'].plot.hist()


# In[65]:


titanic_data.info()


# In[66]:


sns.countplot(x='sibsp',data=titanic_data)


# ## Data Wrangling (Data Cleaning)

# In[67]:


titanic_data.isnull().sum()


# In[68]:


sns.heatmap(titanic_data.isnull(), yticklabels=False,cmap='viridis')


# In[69]:


sns.boxplot(x='pclass',y='age',data=titanic_data)


# In[70]:


titanic_data.head()


# In[71]:


titanic_data.drop("cabin",axis=1,inplace=True)
titanic_data.drop("boat",axis=1,inplace=True)
titanic_data.drop("body",axis=1,inplace=True)
titanic_data.drop("home.dest",axis=1,inplace=True)


# In[72]:


sns.heatmap(titanic_data.isnull(), yticklabels=False,cbar=False)


# In[75]:


titanic_data.dropna(inplace=True)


# In[76]:


sns.heatmap(titanic_data.isnull(), yticklabels=False,cbar=False)


# In[83]:


pd.get_dummies(titanic_data['sex'],drop_first=True).head()


# In[87]:


embark=pd.get_dummies(titanic_data['embarked'],drop_first=True)
embark.head()


# In[89]:


pcl=pd.get_dummies(titanic_data['pclass'],drop_first=True)
pcl.head()


# In[92]:


titanic_data=pd.concat([titanic_data,embark,pcl],axis=1)
titanic_data.head()


# In[94]:


titanic_data.drop(['sex','embarked','name','ticket','pclass'],axis=1,inplace=True)
titanic_data.head()


# ## Train Data

# In[96]:


x=titanic_data.drop("survived",axis=1)
y=titanic_data['survived']


# In[106]:


from sklearn.model_selection import train_test_split


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[109]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[110]:


prediction=logmodel.predict(X_test)


# In[111]:


from sklearn.metrics import classification_report
classification_report(y_test,prediction)


# In[112]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)


# In[113]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)

