#!/usr/bin/env python
# coding: utf-8

# # Project : Loan Prediction

# #Let's Say, You are the owner of the Housing Finance Company and you want to build your own model to predict the customers are applying for the home loan and company want to check and validate the customer are eligible for the home loan.

# #The Problem is

# #In a Simple Term, Company wants to make automate the Loan Eligibility Process in a real time scenario related to customer's detail provided while applying application for home loan forms.

# # Steps are :
# A.Gathering Data
# 
# B.Exploratory Data Analysis
# 
# C.Data Visualizations
# 
# 
# D.Machine Learning Model Decision.
# 
# 
# E.Traing the ML Model
# 
# F.Predict Model

# # Import Modules

# In[1]:


# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # A. Gathering Data
# 

# In[2]:


#Show the Dataset Path to get dataset

loan_train =pd.read_csv(r'D:\BIA\SD_Load Prediction/loan-train.csv')
loan_test = pd.read_csv(r'D:\BIA\SD_Load Prediction/loan-test.csv')


# Lets display the some few information from our large datasets
# 
# Here, We shows the first five rows from datasets

# In[3]:


loan_train.head()


# As we can see in the above output, there are too many columns, ( columns known as features as well. )
# 
# We can also use loan_train to show few rows from the first five and last five record from the dataset

# In[4]:


loan_train


# In[5]:


loan_train.shape


# #After we collecting the data, Next step we need to understand what kind of data we have
# 
# #Also we can get the column as an list(array) from dataset
# 
# #Note: DataFrame.columns returns the total columns of the dataset, Store the number of columns in variable loan_train_columns

# In[6]:


print("Number of Row",loan_train.shape[0])
print("Number of Columns",loan_train.shape[1])


# In[7]:


loan_train_columns = loan_train.columns # assign to a variable
loan_train_columns # print the list of columns


# Now, Understanding the Data
# 
# 1) First of all we use the loan_train.describe() method to shows the important information from the dataset
# 
# 2) It provides the count, mean, standard deviation (std), min, quartiles and max in its output.

# In[8]:


loan_train.describe()


# As I said the above cell, this the information of all the methamatical details from dataset. Like count, mean, standard deviation (std), min, quartiles(25%, 50%, 75%) and max
# 
# #Another method is info(), This method show us the information about the dataset, Like
# 
# What's the type of culumn have?
# 
# How many rows available in the dataset?
# 
# What are the features are there?
# 
# How many null values available in the dataset? Ans so on..

# In[9]:


pd.crosstab(loan_train['Credit_History'],loan_train['Loan_Status'],margins=True)


# In[10]:


loan_train.info()


# As we can see in the output.
# 
# 1) There are 614 entries
# 
# There are total 13 features (0 to 12)
# 
# There are three types of datatype dtypes: float64(4), int64(1), object(8)
#     
# It's Memory usage that is, memory usage: 62.5+ KB
# 
# Also, We can check how many missing values available in the Non-Null Count column
# 

# # B.Exploratory Data Analysis

# In[11]:


loan_train.isnull().sum()


# In[12]:


loan_train.isnull().sum()*100/len(loan_train)


# In[13]:


loan_train =  loan_train.drop('Loan_ID',axis=1)


# In[14]:


loan_train.head(1)


# In[15]:


columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']


# In[16]:


loan_train = loan_train.dropna(subset=columns)


# In[17]:


loan_train.isnull().sum()*100/len(loan_train)


# In[18]:


loan_train['Self_Employed'].mode()


# In[19]:


loan_train['Self_Employed'].mode()[0]


# In[20]:


loan_train['Self_Employed'] = loan_train['Self_Employed'].fillna(loan_train['Self_Employed'].mode()[0])


# In[21]:


loan_train.isnull().sum()*100/len(loan_train)


# In[22]:


loan_train['Credit_History'] = loan_train['Credit_History'].fillna(loan_train['Credit_History'].mode()[0])


# In[23]:


loan_train.isnull().sum()*100/len(loan_train)


# In[24]:


loan_train['Credit_History'].unique()


# In[25]:


loan_train['Self_Employed'].unique()


# In[26]:


loan_train['Credit_History'].mode()[0]


# In[27]:


loan_train['Credit_History'] = loan_train['Credit_History'].fillna(loan_train['Credit_History'].mode()[0])


# In[28]:


loan_train.isnull().sum()*100/len(loan_train)


# In[29]:


loan_train.sample(5)


# In[30]:


loan_train['Dependents'] = loan_train['Dependents'].replace(to_replace="3+",value='4')


# In[31]:


loan_train['Dependents'].unique()


# In[32]:


loan_train['Gender'].unique()


# In[33]:


loan_train['Gender']=loan_train['Gender'].map({'Male':1,'Female':0}).astype('int')
loan_train['Married']=loan_train['Married'].map({'Yes':1,'No':0}).astype('int')
loan_train['Education']=loan_train['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
loan_train['Self_Employed']=loan_train['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
loan_train['Property_Area']=loan_train['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
loan_train['Loan_Status']=loan_train['Loan_Status'].map({'Y':1,'N':0}).astype('int')


# In[34]:


loan_train['Property_Area'].unique()


# In[35]:


loan_train.head()


# In[36]:


X = loan_train.drop('Loan_Status',axis=1)


# In[37]:


X


# In[38]:


y = loan_train['Loan_Status']


# In[39]:


y


# In[40]:


#Features Scalling


# In[41]:


loan_train.head()


# In[42]:


cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']


# In[43]:


from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols]=st.fit_transform(X[cols])


# In[44]:


X


# # C.Data Visualizations

# In[45]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import seaborn as sns



sns.set_style('dark')


# In[46]:


loan_train.plot(figsize=(18, 8))

plt.show()


# In[47]:


plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)


loan_train['ApplicantIncome'].hist(bins=10)
plt.title("Loan Application Amount ")
X

plt.subplot(1, 2, 2)
plt.grid()
plt.hist(np.log(loan_train['LoanAmount']))
plt.title("Log Loan Application Amount ")

plt.show()


# In[48]:


plt.figure(figsize=(18, 6))
plt.title("Relation Between Applicatoin Income vs Loan Amount ")

plt.grid()
plt.scatter(loan_train['ApplicantIncome'] , loan_train['LoanAmount'], c='k', marker='x')
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()


# In[49]:


#Here we can see that apllicanat who have income lesser that 10000 take frequent loans of 100 to 200 dollars


# In[50]:


plt.figure(figsize=(12,8))
sns.heatmap(loan_train.corr(), cmap='coolwarm', annot=True, fmt='.1f', linewidths=.1)
plt.show()


# In[51]:


#In this heatmap, we can clearly seen that loan status is highly correlated with credit history


# # D.Machine Learning Model Decision.

# In[52]:


#splitting the dataset into the Traning Set and test set  & applying  K -fold cross Validation


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


# In[54]:


model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model}accuracy is {accuracy_score(y_test,y_pred)}")
    
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model}Avg cross val  score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)


# In[140]:


model_df


# # E.Training the Machine Learning Model
# 

# In[56]:


#logistic Regression


# In[57]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model_val(model,X,y)


# In[58]:


#SVC


# In[59]:


from sklearn import svm
model = svm.SVC()
model_val(model,X,y)


# In[60]:


#Decision Tree Classifier


# In[61]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_val(model,X,y)


# In[62]:


#Random forest Classifier


# In[63]:


from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model_val(model,X,y)


# In[64]:


#Gradient Boosting  Classifier


# In[65]:


from sklearn.ensemble  import  GradientBoostingClassifier
model =  GradientBoostingClassifier()
model_val(model,X,y)


# # Hyperparameter Tuning

# In[66]:


from sklearn.model_selection import RandomizedSearchCV


# In[67]:


#Logistic Regression


# In[68]:


log_reg_grid={"C":np.logspace(-4,4,20),
              "solver":['liblinear']}


# In[69]:


rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                              param_distributions=log_reg_grid,
                              n_iter=20,cv=5,
                              verbose=True)


# In[70]:


rs_log_reg.fit(X,y)


# In[71]:


rs_log_reg.best_score_


# In[72]:


svc_grid = {'C':[0.25,0.50,0.75,1],"kernel":["linear"]}


# In[73]:


rs_svc=RandomizedSearchCV(svm.SVC(),
                          param_distributions=svc_grid,
                          cv=5,
                          n_iter=20,
                          verbose=True)


# In[74]:


rs_svc.fit(X,y)


# In[75]:


rs_svc.best_score_


# In[76]:


#Random Forest  Classifier
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.ensemble import RandomForestClassifier


# In[77]:


RandomForestClassifier()


# In[78]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
print(rfc.get_params().keys())


# In[79]:


rf_grid={'n_estimators':np.arange(10,1000,10),
         'max_features':['auto','sqrt'],
         'max_depth':[None,3,5,10,20,30],
         'min_samples_split':[2,5,20,50,100],
         'min_samples_leaf':[1,2,5,10],
         
 }


# In[80]:


rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                          param_distributions=rf_grid,
                          cv=5,
                          n_iter=20,
                          verbose=True)


# In[81]:


rs_rf.fit(X,y)


# In[82]:


rs_rf.best_score_


# In[83]:


rs_rf.best_params_


# LogisticRegression score before Hyperparamter Tuning 80.48
# 
# LogisticRegression score after  Hyperparamter Tuning 80.48
# 
# SVC score before Hyperparamter Tuning 79.39
# 
# SVC score after Hyperparamter Tuning 80.66
# 
# RandomForestClassifier before Hyperparamter Tuning  78.85
# 
# RandomForestClassifier after Hyperparamter Tuning  80.66
# 
# 
# 

# # F.Predict Model

# In[84]:


#Save the Model


# In[85]:


X = loan_train.drop('Loan_Status', axis=1)
y = loan_train['Loan_Status']


# In[86]:


rf = RandomForestClassifier(n_estimators=270,
min_samples_split=5,
min_samples_leaf=5,
max_features='sqrt',
max_depth=5)


# In[87]:


rf.fit(X,y)


# In[88]:


import joblib


# In[89]:


joblib.dump(rf,'loan_status_predict')


# In[90]:


model = joblib.load('loan_status_predict')


# In[136]:


import pandas as pd
loan_train = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
    
},index=[0])


# In[137]:


loan_train


# In[138]:


result = model.predict(loan_train)


# In[139]:


if result == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




