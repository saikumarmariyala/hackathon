#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df =pd.read_csv(r"C:\Users\saiku\Downloads\data_2_var (1).csv",names=['f1', 'f2'])


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.rename(columns={'-122.7406674':'X','-130.5720846':'y'},inplace = True)


# In[6]:


df


# In[7]:


sns.scatterplot(data=df,x='f1',y='f2')


# In[8]:


sns.boxplot(x='f1',data = df)
plt.show()


# In[9]:


sns.boxplot(x='f2',data=df, showfliers = False)
plt.show()


# In[10]:


df.isnull().sum()


# In[11]:


df.hist()


# In[12]:


sns.scatterplot(data=df,x='f1',y='f2')


# In[13]:


iqr = 1.5 * (np.percentile(df['f2'], 75) - np.percentile(df['f2'], 25))
df.drop(df[df['f2'] > (iqr + np.percentile(df['f2'], 75))].index, inplace=True)
df.drop(df[df['f2'] < (np.percentile(df['f2'], 25) - iqr)].index, inplace=True)
sns.boxplot(df['f2'])


# In[14]:


x=df['f1']
y=df['f2']
sns.scatterplot(x,y)
plt.show


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)


# In[18]:


print("Training feature set size:",x_train.shape)
print("Test feature set size:",x_test.shape)
print("Training variable set size:",y_train.shape)
print("Test variable set size:",y_test.shape)


# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[20]:


lm = LinearRegression()


# In[ ]:





# In[ ]:




