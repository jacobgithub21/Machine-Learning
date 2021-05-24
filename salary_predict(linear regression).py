#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('C://Users//ELCOT-Lenovo//Downloads//Salary_Data.csv')


# In[5]:


df.shape


# In[6]:


df.head()


# In[9]:


df.dtypes


# In[10]:


df.describe()


# In[11]:


df.info()


# In[12]:


df.isnull()


# In[13]:


df.isnull().sum()


# In[14]:


target = 'Salary'
#seperate object for target 
y = df[target]
#seperate object for input
x = df.drop(target, axis=1)


# In[15]:


y.head()


# In[17]:


y.shape


# In[18]:


x.head()


# In[19]:


x.shape


# In[21]:


plt.scatter(x,y)
plt.xlabel('year of experience')
plt.ylabel('salary')
plt.grid()
plt.show()


# In[22]:


#train and test split
from sklearn.model_selection import train_test_split


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)


# In[31]:


x_train.shape, y_train.shape


# In[32]:


x_test.shape, y_test.shape


# In[33]:


#apply linear regression
from sklearn.linear_model import LinearRegression


# In[34]:


regr = LinearRegression()


# In[35]:


regr.fit(x_train, y_train)


# In[36]:


#get the parametters
regr.intercept_
print('intercept(b) is:',regr.intercept_)


# In[37]:


regr.coef_
print('coefficient (m) is:',regr.coef_)


# In[39]:


#apply the model on test data to get predict values
y_predict = regr.predict(x_test)


# In[40]:


y_predict.shape


# In[45]:


#comparing the predicted values to actual value(comapring xtest to y test)
df1 = pd.DataFrame({'Actual': y_test, 'predicted': y_predict, 'varience': y_test-y_predict})


# In[46]:


df1


# In[47]:


df


# In[50]:


#prediction
pred = np.array([10.3]).reshape(-1,1)
regr.predict(pred)


# In[51]:


#visualization training model
plt.scatter(x_train, y_train ,color='red')
plt.plot(x_train, regr.predict(x_train), color='blue')
plt.title('salary vs experience(trainning set)')
plt.xlabel('year of experience')
plt.ylabel('salary')
plt.grid()
plt.show()


# In[54]:


#visualizing test results
plt.scatter(x_test, y_test, color ='red')
plt.plot(x_train, regr.predict(x_train),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('year of Experience')
plt.ylabel('salary')
plt.show()


# In[57]:


#check score(evaluation metrics of linear algorithms)
from sklearn.metrics import r2_score
score = r2_score(y_test,y_predict)*100
print('score:',score)


# In[ ]:




