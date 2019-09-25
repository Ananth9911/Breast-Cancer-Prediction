#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[7]:


df=pd.read_csv('Breast_cancer_data.csv')
df.head()


# In[26]:


x=df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']].values
x[0:5]


# In[27]:


y=df['diagnosis'].values
y[0:5]


# In[28]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))


# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=4)
print("Train data is ",x_train.shape,y_train.shape)
print("Test data is ",x_test.shape,y_test.shape)


# In[34]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
ks=454
mean_acc=np.zeros((ks-1))
ConfustionMx = [];
for n in range(1,ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
mean_acc


# In[40]:


plt.plot(range(1,ks),mean_acc,'g')
plt.ylabel("Accuracy")
plt.xlabel("Number of Neighbors")
plt.show()


# In[41]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[55]:


neigh=KNeighborsClassifier(n_neighbors=60).fit()
neigh.predict([[7.76,22.39,14.0,189.0,0.0456]])


# In[47]:


df.tail()


# In[ ]:




