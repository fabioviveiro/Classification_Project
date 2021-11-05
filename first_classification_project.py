#!/usr/bin/env python
# coding: utf-8

# <h1><b>Import libraries</b></h1>

# In[10]:


get_ipython().system('pip install opendatasets')


# In[148]:


import pandas as pd
import numpy as np
import opendatasets as od
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import warnings
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report,f1_score, accuracy_score, precision_score, recall_score,plot_roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# <h1><b>Load dataset</b></h1>

# In[12]:


od.download('https://www.kaggle.com/ronitf/heart-disease-uci')


# In[21]:


df = pd.read_csv('/Fabio/GitHub/Classificação/heart-disease-uci/heart.csv')


# <h1><b>Data Analysis</b></h1>

# In[75]:


colunas_df = df.columns


# In[22]:


df.head()


# In[24]:


df.describe(include='all').transpose()


# In[27]:


df.info()


# In[46]:


df.isnull().sum().sort_values(ascending=False)


# In[41]:


plt.figure(figsize=(12,9))
plt.title('Features Correlation')
sns.heatmap(df.corr(),annot=True);


# In[60]:


contColNames = list(df.select_dtypes(include='number').columns)
ncols = 4
nrows = int(np.ceil(len(contColNames)/(1.0*ncols)))
fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(20,15))
counter = 0

for i in range(nrows):
  for j in range(ncols):
    ax = axes[i][j]
    if counter < len(contColNames):
      ax.hist(df.select_dtypes(include='number')[contColNames[counter]],bins=10)
      ax.set_xlabel(contColNames[counter])
      ax.set_ylabel("Frequência")
          
    else:
      ax.set_axis_off()

    counter += 1
plt.show()


# In[64]:


sns.boxplot(df['age']);


# In[65]:


sns.boxplot(df['trestbps']);


# In[66]:


sns.boxplot(df['chol']);


# In[67]:


sns.boxplot(df['thalach']);


# In[68]:


sns.boxplot(df['oldpeak']);


# In[138]:


profile = ProfileReport(df, title='Pandas Profiling Report',explorative=True)
profile


# In[76]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[0]],hue=df['target'])


# In[82]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[1]],hue=df['target'])


# In[83]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[2]],hue=df['target'])


# In[98]:


plt.figure(figsize=(20,6))
sns.countplot(df[colunas_df[3]],hue=df['target']);


# In[97]:


plt.figure(figsize=(20,6))
sns.countplot(df[colunas_df[4]],hue=df['target'])


# In[87]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[5]],hue=df['target'])


# In[88]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[6]],hue=df['target'])


# In[96]:


plt.figure(figsize=(20,6))
sns.countplot(df[colunas_df[7]],hue=df['target'])


# In[90]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[8]],hue=df['target'])


# In[91]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[9]],hue=df['target'])


# In[92]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[10]],hue=df['target'])


# In[93]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[11]],hue=df['target'])


# In[94]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[12]],hue=df['target'])


# In[95]:


plt.figure(figsize=(12,6))
sns.countplot(df[colunas_df[13]],hue=df['target'])


# <h1><b>Dataset split</b></h1>

# In[101]:


X = df.drop(columns='target')
y = df['target']


# In[107]:


pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit_transform(X))


# In[108]:


X_train, X_test, y_train, y_test = train_test_split(pca_df, y, test_size=0.20, random_state=42)


# In[109]:


print(f"Shape X_train: {X_train.shape}")
print(f"Shape y_train: {y_train.shape}")
print(f"Shape X_test: {X_test.shape}")
print(f"Shape y_test: {y_test.shape}")


# In[145]:


sns.scatterplot(pca_df[0],pca_df[1],hue=y);


# <h1><b>Model tests</b></h1>

# <h2><b>Linear SVC</b></h2>

# In[111]:


clf_svc_linear = SVC(kernel='linear')


# In[113]:


clf_svc_linear.fit(X_train, y_train)


# In[114]:


y_pred_svc_linear = clf_svc_linear.predict(X_test)


# In[115]:


confusion_matrix(y_pred= y_pred_svc_linear, y_true = y_test)


# In[116]:


print(classification_report(y_pred= y_pred_svc_linear, y_true = y_test))


# In[147]:


f1 = round(f1_score(y_test, y_pred_svc_linear, average='macro')*100, 2)
accuracy = round(accuracy_score(y_test, y_pred_svc_linear)*100,2)
precision = round(precision_score(y_test, y_pred_svc_linear)*100,2)
recall = round(recall_score(y_test, y_pred_svc_linear)*100,2)


print(f"F1 Score: {f1}%")
print(f"Accuracy Score: {accuracy}%")
print(f"Precision Score: {precision}%")
print(f"Recall Score: {recall}%")


# In[149]:


plot_roc_curve(clf_svc_linear, X_test, y_test)  
plt.grid()
plt.show()


# <h2><b>Polynomial SVC</b></h2>

# In[117]:


clf_svc_poly = SVC(kernel='poly')


# In[118]:


clf_svc_poly.fit(X_train, y_train)


# In[119]:


y_pred_svc_poly = clf_svc_poly.predict(X_test)


# In[120]:


confusion_matrix(y_pred= y_pred_svc_poly, y_true = y_test)


# In[121]:


print(classification_report(y_pred= y_pred_svc_poly, y_true = y_test))


# In[150]:


f1 = round(f1_score(y_test, y_pred_svc_poly, average='macro')*100, 2)
accuracy = round(accuracy_score(y_test, y_pred_svc_poly)*100,2)
precision = round(precision_score(y_test, y_pred_svc_poly)*100,2)
recall = round(recall_score(y_test, y_pred_svc_poly)*100,2)


print(f"F1 Score: {f1}%")
print(f"Accuracy Score: {accuracy}%")
print(f"Precision Score: {precision}%")
print(f"Recall Score: {recall}%")


# In[151]:


plot_roc_curve(clf_svc_poly, X_test, y_test)  
plt.grid()
plt.show()


# <h2><b>Decision Tree</b></h2>

# In[128]:


clf_tree = DecisionTreeClassifier(criterion="entropy")
clf_tree.fit(X_train,y_train)


# In[129]:


y_pred_tree = clf_tree.predict(X_test)


# In[130]:


confusion_matrix(y_pred= y_pred_tree, y_true = y_test)


# In[131]:


print(classification_report(y_pred= y_pred_tree, y_true = y_test))


# In[152]:


f1 = round(f1_score(y_test, y_pred_tree, average='macro')*100, 2)
accuracy = round(accuracy_score(y_test, y_pred_tree)*100,2)
precision = round(precision_score(y_test, y_pred_tree)*100,2)
recall = round(recall_score(y_test, y_pred_tree)*100,2)


print(f"F1 Score: {f1}%")
print(f"Accuracy Score: {accuracy}%")
print(f"Precision Score: {precision}%")
print(f"Recall Score: {recall}%")


# In[153]:


plot_roc_curve(clf_tree, X_test, y_test)  
plt.grid()
plt.show()


# <h2><b>MLP</b></h2>

# In[134]:


clf_MLP = MLPClassifier()
clf_MLP.fit(X_train,y_train)


# In[135]:


y_pred_MLP = clf_MLP.predict(X_test)


# In[136]:


confusion_matrix(y_pred= y_pred_MLP, y_true = y_test)


# In[137]:


print(classification_report(y_pred= y_pred_MLP, y_true = y_test))


# In[154]:


f1 = round(f1_score(y_test, y_pred_MLP, average='macro')*100, 2)
accuracy = round(accuracy_score(y_test, y_pred_MLP)*100,2)
precision = round(precision_score(y_test, y_pred_MLP)*100,2)
recall = round(recall_score(y_test, y_pred_MLP)*100,2)


print(f"F1 Score: {f1}%")
print(f"Accuracy Score: {accuracy}%")
print(f"Precision Score: {precision}%")
print(f"Recall Score: {recall}%")


# In[155]:


plot_roc_curve(clf_MLP, X_test, y_test)  
plt.grid()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




