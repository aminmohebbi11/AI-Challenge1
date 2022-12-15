#!/usr/bin/env python
# coding: utf-8

# # Import the relevant libraries

# In[1]:


# For the AI challenge we will need the following libraries and modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# In[2]:


# loading the data
raw_data = pd.read_csv('220511_monatszahlenmonatszahlen2204_verkehrsunfaelle.csv')

# explore the head of the data
raw_data.head()


# # Preprocessing

# ### description of  the variables

# In[3]:


raw_data.describe(include='all')


# ### Dealing with missing values

# In[4]:


raw_data.isnull().sum()


# In[5]:


#drop all missing values

data_no_miss = raw_data.dropna(axis=0)


# In[6]:


data_no_miss.isnull().sum()


# In[7]:


#raw_data_sort1 = data_no_miss.sort_values(by=['WERT'] )
#raw_data_sort1.head()


# In[8]:


# the head of data without the missing values
data_no_miss.head()


# In[9]:


# description  of data without the missing values
data_no_miss.describe(include='all')


# In[10]:


#changing the type of the 'MONAT' column from object to integer
data_no_miss['MONAT']  = data_no_miss['MONAT'].astype(object).astype(int)
data_no_miss['MONAT']


# In[11]:


#Here I check the data again
data_no_miss.describe(include='all')


# # Visualizations 

# In[12]:


#Plot the graph of the WERT with respect to MONAT. Here Icheck some methods and finally find a histplot
#is a good one

#sns.histplot( data = data_no_miss , x="MONAT" , binwidth=0.5,binrange=0.7,discrete=1)
#sns.barplot(x="MONAT",y="WERT",data=data_no_miss )
#sns.boxplot(x ='WERT', y ='MONAT', data = data_no_miss, hue ='JAHR', saturation=2, width=1)
sns.histplot(
    data_no_miss, x="WERT", hue="MONAT", element="step",
    stat="density", common_norm=True
)


# In[13]:


#Plot the graph of the WERT with respect to JAHR.
sns.barplot(x="JAHR",y="WERT",data=data_no_miss )
plt.show()


# In[14]:


#Plot the graph of the WERT with respect to MONATSZAHL.
sns.barplot(x="MONATSZAHL",y="WERT",data=data_no_miss)
plt.show()


# In[15]:


#Plot the graph of the WERT with respect to AUSPRAEGUNG.
sns.histplot(binwidth=0.5, x="WERT", hue="AUSPRAEGUNG", data=data_no_miss, stat="count", multiple="stack")


# In[16]:


#Another graph for  the graph of the WERT with respect to MONATSZAHL.
#Plot the graph of the WERT with respect to MONATSZAHL.
sns.boxplot(x ='MONATSZAHL', y ='WERT', data = data_no_miss, hue ='JAHR', saturation=1, width=1)


# In[17]:


#Another plot for the graph of the WERT with respect to JAHR.
sns.catplot(data=data_no_miss, x="JAHR", y="WERT" , height=10,aspect=1)


# In[18]:


sns.catplot(data=data_no_miss, x="MONATSZAHL", y="WERT" ,  height=10,aspect=2)


# In[19]:


sns.catplot(data=data_no_miss, x="AUSPRAEGUNG", y="WERT" ,  height=10 , aspect=2)


# # Exploring the PDFs

# In[20]:


#Exploring the PDF () for dependetn varialbe WERT. Give us information about the statistial model of our ML model
# And I can checkthe outliers of the variable WERT on the graph
sns.distplot(data_no_miss['WERT'])
# this looks like a normal distribution 


# ### Dealing with outliers

# In[21]:


#here I used .quantile method and removed some outliers from WERT
#q1 = data_no_miss['WERT'].quantile(0.99)
q2 = data_no_miss['WERT'].quantile(0.01)
#data1 = data_no_miss[data_no_miss['WERT']<q1]
data1 = data_no_miss[data_no_miss['WERT']>q2]


# In[22]:


# Removing the outliers did not change the model
sns.distplot(data1['WERT'])


# In[23]:


#since we droped some rows. here we need reset the index of the rows. To looking better our dataset
data_cleaned = data1.reset_index(drop=True)


# In[24]:


data_cleaned.head(10)


# In[25]:


data_cleaned.describe(include='all')


# In[26]:


sns.displot(data_cleaned['WERT'])


# In[27]:


# Here I am going to use plt.scatter() and compare some features with WERT. This make a useful sense 
#to seclect independent features. since we can see the relations of the features and WERT 

# WERT is the 'y' axis of all the plots, and we plot them side-by-side (to easily compare them)

f, (ax1, ax2, ax3,ax4,ax5 , ax6) = plt.subplots(1, 6, sharey=True, figsize =(20,5)) 
ax1.scatter(data_cleaned['VORJAHRESWERT'],data_cleaned['WERT'])
ax1.set_title('VORJAHRESWERT')
ax2.scatter(data_cleaned['VERAEND_VORMONAT_PROZENT'],data_cleaned['WERT'])
ax2.set_title('VERAEND_VORMONAT_PROZENT')
ax3.scatter(data_cleaned['VERAEND_VORJAHRESMONAT_PROZENT'],data_cleaned['WERT'])
ax3.set_title('VERAEND_VORJAHRESMONAT_PROZENT')
ax4.scatter(data_cleaned['ZWOELF_MONATE_MITTELWERT'],data_cleaned['WERT'])
ax4.set_title('ZWOELF_MONATE_MITTELWERT')
ax5.scatter(data_cleaned['MONAT'],data_cleaned['WERT'])
ax5.set_title('MONAT')
ax6.scatter(data_cleaned['JAHR'],data_cleaned['WERT'])
ax6.set_title('JAHR')


plt.show()


# In[28]:


#Try to find a good transformation for WERT 
# Let's transform 'WERT' with a log transformation
log_wert = np.log(data_cleaned['WERT'])
 
# and add it to the data frame
data_cleaned['log_wert'] = log_wert.round(3)
data_cleaned


# In[29]:


#Now I am  going to use scatterplot to compare the logWERT with other features
#perhaps, we can find some good relations between features and WERT

f, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['VORJAHRESWERT'],data_cleaned['log_wert'])
ax1.set_title('VORJAHRESWERT')
ax2.scatter(data_cleaned['VERAEND_VORMONAT_PROZENT'],data_cleaned['log_wert'])
ax2.set_title('VERAEND_VORMONAT_PROZENT')
ax3.scatter(data_cleaned['VERAEND_VORJAHRESMONAT_PROZENT'],data_cleaned['log_wert'])
ax3.set_title('VERAEND_VORJAHRESMONAT_PROZENT')
ax4.scatter(data_cleaned['ZWOELF_MONATE_MITTELWERT'],data_cleaned['log_wert'])
ax4.set_title('ZWOELF_MONATE_MITTELWERT')


plt.show()


# In[30]:


#since the log transformation gave me not a good result and relation between the feature, I omit it form the dataset
data_cleaned = data_cleaned.drop(['log_wert'],axis=1)


# In[31]:


#Control the statistical model of ''VERAEND_VORJAHRESMONAT_PROZENT''
sns.displot(data_cleaned['VERAEND_VORJAHRESMONAT_PROZENT'])


# In[32]:


#Control the statistical model of ''VERAEND_VORMONAT_PROZENT''
sns.displot(data_cleaned['VERAEND_VORMONAT_PROZENT'])


# In[33]:


#Control the statistical model of ''VORJAHRESWERT''
sns.displot(data_cleaned['VORJAHRESWERT'])


# ### Multicollinearity

# In[34]:


# Since I am going to use multilinear regression and I have several dependent variables, I have to chech the multicollinearity 
#of them. Thus I wouldlike to import variance_inflation_factor (VIF) from statsmodels.stats.outliers_influence  and 
#check the collinearity of the dependent variable. 

#if our variables have VIF begger than 5 and less than 10, they are good choices for our ML model. 

# http://www.statsmodels.org/dev/_modules/statsmodels/stats/outliers_influence.html#variance_inflation_factor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Here, I will only take the numerical ones. the categorical is not necessary for this operation
variables = data_cleaned[['MONAT','JAHR', 'VORJAHRESWERT','VERAEND_VORMONAT_PROZENT','VERAEND_VORJAHRESMONAT_PROZENT','ZWOELF_MONATE_MITTELWERT']]

# we create a new data frame with all VIF's
VIF = pd.DataFrame()

# by the variance_inflation_factor, we create VIF outputs  
VIF["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# Finally,  explore the result
VIF["Features"] = variables.columns


# In[35]:


VIF


# In[36]:


#Since vif is begger than 5 and less than 10 for 'VERAEND_VORMONAT_PROZENT' and 'VERAEND_VORJAHRESMONAT_PROZENT',
#So we can use them in our model, there two features have no multicollinearity. And then omit two other features 
data_no_JAHR= data_cleaned.drop(['JAHR'],axis=1)
data_no_MONAT= data_no_JAHR.drop(['MONAT'],axis=1)
data_no_VVP= data_no_MONAT.drop(['VERAEND_VORMONAT_PROZENT'],axis=1)
data_no_VVPP= data_no_VVP.drop(['VERAEND_VORJAHRESMONAT_PROZENT'],axis=1)
data_no_multicollinearity =data_no_VVPP
data_no_multicollinearity.head()
#as a remark, I have already checked the other features, the result of my model with the above features is the optimal


# ### Create dummy variables

# In[37]:


# in our data set, we have some categorical features. so we can use get_dummies' methods and find some numerical value for them  

data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=False)
#We can use drop_first=False and keep  columns, but they will not affect to the final result 


# In[38]:


#part 1 : this is the second method to get dummy features. but the above methods fave me a better result
#data1 = data_no_multicollinearity.copy()
#data1['MONATSZAHL'] = data1['MONATSZAHL'].map({'Alkoholunfälle':0 , 'Fluchtunfälle':1 , 'Verkehrsunfälle':2 })
#data1
data_with_dummies


# In[39]:


#part 2:
#data2 = data1.copy()
#data2['AUSPRAEGUNG'] = data2['AUSPRAEGUNG'].map({'insgesamt':0 , 'Verletzte und Getötete':1 , 'mit Personenschäden':2 })
#data2


# In[40]:


#part 3:
# Here's the result
#data_with_dummies= data2


# In[41]:


# We  display all possible features and then choose them
data_with_dummies.columns.values


# In[42]:


#  a new variable that will contain the preferred order
# I toke the depend variable in the beginning  and dummies at the end. 

cols =['WERT', 'VORJAHRESWERT', 'ZWOELF_MONATE_MITTELWERT',
       'MONATSZAHL_Alkoholunfälle', 'MONATSZAHL_Fluchtunfälle',
       'MONATSZAHL_Verkehrsunfälle', 'AUSPRAEGUNG_Verletzte und Getötete',
       'AUSPRAEGUNG_insgesamt', 'AUSPRAEGUNG_mit Personenschäden']


# In[43]:


# create a new df to implement the reordering with new order
data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# In[44]:


data_preprocessed.describe(include='all')


# # Linear regression model

# ### Linear regression model

# In[45]:


# The target(s) (dependent variable) is 'WERT'
targets = data_preprocessed['WERT']

# The inputs are everything WITHOUT the dependent variable, so we should drop it from the dataset
inputs = data_preprocessed.drop(['WERT'],axis=1)


# ### Train Test Split

# In[46]:


# we import the module for the split and split the data to train and test parts
from sklearn.model_selection import train_test_split

# we split the variables with an 80-20 split and some random state

x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=20)


# ### Create the regression 

# In[47]:


# Create a linear regression object 
#note that, my model is multilinear regression 
reg = LinearRegression()
# Fit the regressionwith x_train and y_train 
reg.fit(x_train,y_train)


# In[48]:


# check the outputs of the regression
y_hat = reg.predict(x_train)


# In[49]:


# to compare the targets (y_train) and the predictions (y_hat), we use a scatter plot and plot the y_train ands y_hat
# if our graph is  close to the 45-degree line, we will  have a better prediction
plt.scatter(y_train, y_hat)
#name the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)

# We want the x-axis and the y-axis to be the same
plt.xlim()
plt.ylim()
plt.show()


# In[50]:


# Using a residual plot is useful to check our model
# We plot the PDF of the residuals and check for anomalies
sns.distplot(y_train - y_hat)

# the title
plt.title("Residuals PDF", size=18)

# the best scenario is the plot be normally distributed
# The definition of the residuals is (y_train - y_hat), negative values imply
# that predictions are much higher than the targets


# In[51]:


# The R-squared of the model is
reg.score(x_train,y_train)

# This is the accuracy of the model


# ### Finding the weights and bias

# In[52]:


# The bias (intercept) of the regression
reg.intercept_


# In[53]:


# The weights (coefficients) of the regression
reg.coef_


# In[54]:


# A regression summary is helpful to compare the features with the weigths
#variables = data_cleaned[['MONAT','JAHR', 'VORJAHRESWERT','VERAEND_VORMONAT_PROZENT','VERAEND_VORJAHRESMONAT_PROZENT','ZWOELF_MONATE_MITTELWERT']]
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# ### Testing

# In[55]:


# We have trained our model, now we need to test it as follow.
# We have prepared a dataset to test our model and it is called 'x_test', and the outputs is 'y_test' 

y_hat_test = reg.predict(x_test)


# In[56]:


# We create another scatter plot with the test targets and the test predictions. This plot should be similar to 
#the scatter plot with the train set
# the argument 'alpha' is the opacity of the graph
plt.scatter(y_test, y_hat_test, alpha=0.4)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim()
plt.ylim()
plt.show()


# In[57]:


# Here we check these predictions manually 
data_with_pred = pd.DataFrame(y_hat_test, columns=['Prediction'])
data_with_pred.head()


# In[58]:


# And then include the test targets in the data frame and then compre them manually 
data_with_pred['Target'] = y_test
data_with_pred

# Note that we have a lot of missing values


# In[59]:


# Now we can use the result of y_test an  find the issue 
# to get a proper result, we neet to reset the index and drop the old indexing
y_test = y_test.reset_index(drop=True)

# Check the result
y_test.head()


# In[60]:


# we overwrite the 'Target' column 

data_with_pred['Target'] = y_test
data_with_pred


# In[61]:


# We calculate the difference between the targets and the predictions. This is actually the residual.
#This is useful to compare our results(target and prediction)
data_with_pred['Residual'] = data_with_pred['Target'] - data_with_pred['Prediction']


# In[62]:


# Finally, it makes sense to see how far off we are from the result percentage-wise
# we take the absolute difference in %, so we can easily order the data frame
data_with_pred['Difference%'] = np.absolute(data_with_pred['Residual']/data_with_pred['Target']*100)
data_with_pred
#This makes sense to see how far off from the results precentage wise


# In[63]:


data_with_pred.describe()


# In[64]:


# Again we check our outputs manually
# To see all rows, we can use the following  pandas syntax, and show the result with 2 digit after dot
pd.options.display.max_rows = 1000
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Now, we sort by difference in % 
data_with_pred.sort_values(by=['Difference%'])


# In[65]:


# we create the OLS summary table for our multilinear regression model to see the details of the model
x = sm.add_constant(inputs)
results = sm.OLS(targets,x).fit()
results.summary()


# In[66]:


# The above table is useful, but for p-values of the multilinear regression does not work. So it is better we apply
# another methods to check the p-values
#Since we applied a multilinear regression, it is better to use f_regression to check the p-values of our features .
from sklearn.feature_selection import f_regression
f_regression(inputs , targets)


# In[67]:


p_values = f_regression(inputs , targets)[1].round(3)
p_values


# # Some numerical examples for our prediction model

# In[68]:


# My indepenten variables are 'VORJAHRESWERT', 'ZWOELF_MONATE_MITTELWERT',
# 'MONATSZAHL_Alkoholunfälle', 'MONATSZAHL_Fluchtunfälle',
# 'MONATSZAHL_Verkehrsunfälle', 'AUSPRAEGUNG_Verletzte und Getötete',
# 'AUSPRAEGUNG_insgesamt', 'AUSPRAEGUNG_mit Personenschäden'

#Since we have multiple values for Category: 'Alkoholunfälle', Type: 'insgesamt and year '2021' 
#and month '202101', we need to have the precise value of 
#'VORJAHRESWERT' 202101 = 28 , 719,3139
#'ZWOELF_MONATE_MITTELWERT' 202101 = 35 , 813 ,3121

#check 1
#data_check1 = [28, 35 , 1, 0,0 ,0,1,0]
#data_check1_arr = np.array(data_check1)
#data_check1_num = data_check1_arr.reshape(1,-1)
#pred_check1 = reg.predict(data_check1_num)
#pred_check1


# In[69]:


#check 2 -34.13
#data_check2 = [719, 813 ,1,  0,0,0 ,1,0]
#data_check2_arr = np.array(data_check2)
#data_check2_num = data_check2_arr.reshape(1,-1)
#pred_check2 = reg.predict(data_check2_num)
#pred_check2


# In[70]:


#check 3 -31.41
#data_check3 = [3139, 3121, 1,  0,0 ,0,1,0]
#data_check3_arr = np.array(data_check3)
#data_check3_num = data_check3_arr.reshape(1,-1)
#pred_check3 = reg.predict(data_check3_num)
#pred_check3


# # Deploy the Model via Flask

# ### Make a Pickle file of our model

# In[71]:


import pickle


# In[72]:


#Using dump method, and pass our object
pickle.dump(reg , open('pred_model.pkl' , 'wb'))


# In[ ]:




