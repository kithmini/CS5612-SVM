#!/usr/bin/env python
# coding: utf-8

# In[9]:


print('##### SVM Classification on Wisconsin Breast Cancer Data #####')

print('----- Importing required libraries & modules-----')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn import linear_model
from scipy.special import expit


# In[10]:


print('----- Importing dataset -----')
data = pd.read_csv('wbcd.csv', header=None, na_values=['?'])

data.columns = ['sample_code_number','clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion',              'single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
feature_columns = ['clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion',              'single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses']


print ('Imported Rows, Columns - ', data.shape)
print ('Data Head :')
data.head()


# In[11]:


print('----- Preprocessing Data -----')

processedData = data.dropna() #dropping missing value rows

print ('Complete Rows, Columns - ', processedData.shape)

X = processedData[feature_columns]
Y = processedData['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)

print('----- Training Classifier -----')

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)


# In[12]:


print('----- Testing Classifier -----')
Y_pred = svclassifier.predict(X_test)

from sklearn import metrics

print("Accuracy:  %.2f%%",metrics.accuracy_score(Y_test, Y_pred)*100.0)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


# In[ ]:





# In[ ]:




