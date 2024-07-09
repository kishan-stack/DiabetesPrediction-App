#!/usr/bin/env python
# coding: utf-8

# # Lab 10 : Model Selction

# ## Import Commomn Libraries

# In[53]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import Training Dataset

# In[54]:


df = pd.read_csv('diabetes_prediction.csv')
df.head()


# In[55]:


df.shape


# In[56]:


df.columns


# In[57]:


df.info()


# In[58]:


df.describe()


# In[59]:


df.describe(include = 'object')


# # Data visualization and pre-processing Steps

# In[60]:


df['Diagnosis'].value_counts()


# In[61]:


import seaborn as sns

bins = np.linspace(df.BMI.min(), 
                   df.BMI.max(), 10)
g = sns.FacetGrid(df, col="Gender", 
                  hue="Diagnosis", 
                  palette="Set1", 
                  col_wrap=2)
g.map(plt.hist, 
      'BMI', 
      bins=bins, 
      ec="k")

g.axes[-1].legend()
plt.show()


# In[62]:


bins = np.linspace(df.Age.min(), 
                   df.Age.max(), 10)
g = sns.FacetGrid(df, 
                  col="Gender", 
                  hue="Diagnosis", 
                  palette="Set1", 
                  col_wrap=2)
g.map(plt.hist, 
      'Age', 
      bins=bins, 
      ec="k")

g.axes[-1].legend()
plt.show()


# # Convert Categorical features to numerical values

# In[63]:


df.groupby(['Gender'])['Diagnosis'].value_counts(normalize=True)


# In[76]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame and you have the columns to be encoded
# Define your DataFrame
# df = pd.DataFrame(...)  # Your existing DataFrame

# List of columns to encode
columns_to_encode = ['Gender', 'Family History of Diabetes']  # Replace with your actual column names

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Apply LabelEncoder to each column in the list
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])


# In[77]:


Feature = df[['Age','Gender','BMI','Family History of Diabetes']]
Feature.head()
Feature.info()


# In[ ]:





# # Feature Selection¶

# In[78]:


X = Feature
X[0:5]


# In[79]:


y = df['Diagnosis']
y[0:5]
d = {'Yes':0,'No' : 1}
y = y.map(d)


# # Normalize Data

# ### Data Standardization give data zero mean and unit variance (technically should be done after train test split)
# 
# 

# In[80]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[:]


# # Split the Data into Training and Testing Set

# In[81]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # Classification

# In[82]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):  
    knn1 = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=knn1.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[83]:


from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']
pow = [1,2]
param_grid = dict(n_neighbors=k_range, 
                  weights=weight_options,
                  p = pow)
knn_gs = KNeighborsClassifier()
grid_k = GridSearchCV(knn_gs, 
                    param_grid, 
                    cv=10, 
                    scoring='accuracy')
grid_k.fit(X_train, y_train)


# In[84]:


print("Tuned Hyperparameters :", grid_k.best_params_)
print("Accuracy :",grid_k.best_score_)


# In[86]:


knn1 = KNeighborsClassifier(n_neighbors= 26, p = 2, weights = 'uniform')


# In[87]:


knn1.fit(X_train,y_train)


# In[88]:


yhat = knn1.predict(X_test)


# In[89]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
a1 = jaccard_score(y_test,yhat,pos_label=1)
b1 = f1_score(y_test, yhat, average='weighted')
c1 = accuracy_score(y_test, yhat)
print('The jaccard_score of the KNN for k = 7 classifier on train data is {:.2f}'.format(a1))
print('The F1-score of the KNN for k = 7 classifier on train data is {:.2f}'.format(b1))
print('The Accuracy_score of the KNN for k = 7 classifier on train data is {:.2f}'.format(c1))
