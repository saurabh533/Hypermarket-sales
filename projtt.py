#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


# In[2]:


findspark.init()


# In[3]:


import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark import SparkFiles


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import statsmodels.api as sm

from datetime import datetime

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[4]:


spark=SparkSession.builder.appName("Spark-MongoDB Connector").config("spark.mongodb.input.uri", "mongodb:27017//localhost/test_db.input_collection").config("spark.mongodb.output.uri", "mongodb:27017//localhost/test_db.output_collection").config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:3.0.0").getOrCreate()


# downloads the "features.csv" file from a GitHub repository and reads it into a Spark DataFrame. Here's a breakdown of the code:
# 
#     import urllib.request: This imports the urllib.request module to allow downloading files from a URL.
#     import os: This imports the os module to allow interacting with the file system.
#     url = 'https://github.com/saurabh533/weeklysales_project/blob/main/features.csv?raw=true': This sets the URL of the file to download.
#     filename = 'features.csv': This sets the name of the file to save.
#     local_path = os.path.join('/tmp', filename): This sets the local path to save the file.
#     urllib.request.urlretrieve(url, local_path): This downloads the file from the URL and saves it to the local path.
#     if os.path.exists(local_path):: This checks if the file exists.
#     features = spark.read.csv("file://" + local_path, header=True, inferSchema=True): This reads the CSV file into a Spark DataFrame with headers and schema inference.
# 
# 

# In[5]:


import urllib.request
import os
url = 'https://github.com/saurabh533/weeklysales_project/blob/main/features.csv?raw=true'
filename = 'features.csv'
local_path = os.path.join('/tmp', filename)
urllib.request.urlretrieve(url, local_path)
if os.path.exists(local_path):
    features = spark.read.csv("file://" + local_path, header=True, inferSchema=True)
else:
    print("File not downloaded")
# features = spark.read.csv("/home/saurabh/Documents/project1-main/data/features.csv", header=True, inferSchema=True)


# In[6]:


features.show(5)


# We are using Apache Spark to read a CSV file containing information about stores. The code you provided downloads the CSV file from a URL and saves it to a local path using the urllib library, then reads the file into a Spark DataFrame using spark.read.csv().
# 
# Here's a breakdown of the code:
# 
#     The url variable contains the URL of the CSV file to be downloaded.
#     The filename variable contains the name of the file to be saved locally.
#     The local_path variable is the path where the file will be saved on the local machine (in this case, it's the /tmp directory).
#     urllib.request.urlretrieve() is used to download the file from the URL and save it to the local path.
#     spark.read.csv() is used to read the CSV file into a Spark DataFrame.
# 
# Note that the header and inferSchema options are set to True, which tells Spark to use the first row of the CSV file as the header and to infer the schema of the DataFrame based on the data in the file.
# 

# In[7]:


url = 'https://github.com/saurabh533/weeklysales_project/blob/main/stores.csv?raw=true'
filename = 'stores.csv'
local_path = os.path.join('/tmp',filename)
urllib.request.urlretrieve(url, local_path)
if os.path.exists(local_path):
    stores = spark.read.csv("file://" + local_path, header=True, inferSchema=True)
else:
    print("File not downloaded")


# In[8]:


stores.show(5)


# In[9]:


url = 'https://github.com/saurabh533/weeklysales_project/blob/main/train.csv?raw=true'
filename = 'train.csv'
local_path = os.path.join('/tmp', filename)
urllib.request.urlretrieve(url, local_path)
if os.path.exists(local_path):
    train = spark.read.csv("file://" + local_path, header=True, inferSchema=True)
else:
    print("File not downloaded")
#train = spark.read.csv("/home/saurabh/Documents/project1-main/data/train.csv", header=True, inferSchema=True)


# In[10]:


train.show(50)


# In[11]:


url = 'https://github.com/saurabh533/weeklysales_project/blob/main/test.csv?raw=true'
filename = 'test.csv'
local_path = os.path.join('/tmp', filename)
urllib.request.urlretrieve(url, local_path)
if os.path.exists(local_path):
    test = spark.read.csv("file://" + local_path, header=True, inferSchema=True)
else:
    print("File not downloaded")


# In[12]:


test.show(5)


# # PREPROCESSING 
# #CHECKING NULL VALUES

# The above code is used to count the number of null values present in each column of the features DataFrame.
# 
# The code first creates a dictionary called checkNullValues, which will store the count of null values in each column of the features DataFrame.
# 
# The code then iterates over each column in the features DataFrame using a for loop and the columns attribute of the DataFrame. For each column, it calls the isNull() method on the column to get a DataFrame of only the rows where the value in that column is null, and then calls the count() method to count the number of rows in that DataFrame. The count is then stored in the checkNullValues dictionary using the name of the column as the key.
# 
# Finally, the code prints the checkNullValues dictionary, which shows the count of null values in each column of the features DataFrame.
# 

# In[13]:


chcekNullValues = {col:features.filter(features[col].isNull()).count() for col in features.columns}
print(chcekNullValues)


# In[14]:


chcekNullValues1 = {col:stores.filter(stores[col].isNull()).count() for col in stores.columns}
print(chcekNullValues1)


# In[15]:


chcekNullValues2 = {col:test.filter(test[col].isNull()).count() for col in test.columns}
print(chcekNullValues2)


# In[16]:


chcekNullValues3 = {col:train.filter(train[col].isNull()).count() for col in train.columns}
print(chcekNullValues3)


# # JOINING DATA USING INNER JOIN

# #joining (train+stores+features)
# #joining (test+stores+features)

# In[17]:


train.printSchema()


# In[18]:


stores.printSchema()


# In[19]:


features.printSchema()


# In[20]:


train_bt=train.join(stores, ["Store"])
train1 = train_bt.join(features, on=['Store', 'Date','IsHoliday'], how='inner')


# In[21]:


train1.show(2)


# In[22]:


test_bt=test.join(stores,["Store"])
test1 = test_bt.join(features, on=['Store', 'Date','IsHoliday'], how='inner')


# In[23]:


test1.show(2)


# In[24]:


type(test)


# In[25]:


test1.printSchema()


# In[26]:


train1.printSchema()


# In[27]:


features.printSchema()


# In[28]:


print(train1.printSchema())
print("*****************************************")
print(test.printSchema())


# # dump data in mongo

# The below code writes the train1 dataframe to a MongoDB database named "shopping" and a collection named "data_train". It uses the mongo format to specify the output format and option to provide the connection URI to the MongoDB instance running on the local machine on port 27017. The save() method is called to save the data to the specified MongoDB collection.

# In[29]:


train1.write.format("mongo").option("uri","mongodb://localhost:27017/shopping.data_train").save()


# In[30]:


test1.write.format("mongo").option("uri","mongodb://localhost:27017/shopping.data_test").save()


# # getting data from mongo to pandas

# In[31]:


import pymongo
from pymongo import MongoClient


# The below code connects to a MongoDB database hosted on the local machine using the pymongo driver for Python. The MongoClient object is created with the hostname and port number of the database. Then, the database and collection names are specified as shopping and data_train, respectively.
# 
# Next, the code fetches all the documents from the data_train collection using the find() method and converts the resulting cursor into a list of dictionaries. This list of dictionaries is then passed to the pd.DataFrame() function to create a Pandas DataFrame. Finally, the _id column is dropped from the DataFrame using the drop() method with the axis parameter set to 1.
# 

# In[32]:


client = MongoClient('localhost',27017)
db = client.shopping
train = db.data_train

train = pd.DataFrame(list(train.find())).drop(['_id'],axis=1)


# The given code connects to a MongoDB database named shopping using a MongoClient object client. Then, it selects a collection named data_test from the shopping database using the db object. The data_test collection may contain some test data related to the retail industry, as the data_train collection contains the training data.
# 
# Next, the code uses the find() method to retrieve all the documents (rows) from the data_test collection and converts the result to a list. The list is then passed to the pd.DataFrame() method to create a pandas DataFrame object. The _id column is then dropped from the DataFrame using the drop() method, as the _id column is automatically generated by MongoDB and may not be useful for analysis.
# 
# Overall, the given code retrieves the test data from the data_test collection of the shopping database and converts it to a pandas DataFrame for further analysis.
# 

# In[33]:


db = client.shopping
test = db.data_test
test = pd.DataFrame(list(test.find())).drop(['_id'],axis=1)


# In[34]:


train.columns


# In[35]:


test.columns


# In[36]:


print(train.info())
print ("*****************************************")
print(test.info())


# # cleaning data

# The given code converts the Date column in the pandas DataFrame train to np.datetime64 data type.
# 
# np.datetime64 is a NumPy data type used to represent date and time values. By converting the Date column to np.datetime64 data type, the column values will be recognized and treated as datetime objects, which can be useful for time series analysis and manipulation.
# 
# In the given code, the astype method is used to cast the data type of the Date column to np.datetime64. The astype method creates a new pandas Series object with the same data as the original Date column, but with the data type changed to np.datetime64. The original Date column is not modified by this operation.

# In[37]:


train['Date']=train['Date'].astype(np.datetime64)


# In[38]:


train.dtypes


# In[39]:


test.dtypes


# In[40]:


# train['MarkDown1'] = train['MarkDown1'].astype(np.float64)
# train['MarkDown2'] = train['MarkDown1'].astype(np.float64)
# train['MarkDown3'] = train['MarkDown1'].astype(np.float64)
# train['MarkDown4'] = train['MarkDown1'].astype(np.float64)
# train['MarkDown5'] = train['MarkDown1'].astype(np.float64)
# train['CPI'] = train['MarkDown1'].astype(np.float64)
# train['Unemployment'] = train['Unemployment'].astype(np.float64)

# test['MarkDown1'] = test['MarkDown1'].astype(np.float64)
# test['MarkDown2'] = test['MarkDown1'].astype(np.float64)
# test['MarkDown3'] = test['MarkDown1'].astype(np.float64)
# test['MarkDown4'] = test['MarkDown1'].astype(np.float64)
# test['MarkDown5'] = test['MarkDown1'].astype(np.float64)
# test['CPI'] = test['MarkDown1'].astype(np.float64)
# test['Unemployment'] = test['Unemployment'].astype(np.float64)




# The given code cleans and preprocesses the test data by replacing missing values with zero (0) and converting certain columns to np.float64 or np.datetime64 data type.
# 
# Specifically, the code replaces the string value "NA" in columns MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, CPI, and Unemployment with the numeric value zero (0) using the str.replace() method. This is because "NA" values are likely placeholders for missing data, and replacing them with zero can make it easier to perform numerical calculations and analysis.
# 
# Next, the astype() method is used to convert the columns to their respective data types. The astype() method creates a new pandas Series object with the same data as the original column, but with the data type changed to np.float64 or np.datetime64.
# 
# Finally, note that the last line of the given code is a mistake and should be corrected to test['Date']=test['Date'].astype(np.datetime64). This line converts the Date column in the test DataFrame to np.datetime64 data type.
# 

# In[41]:


test['MarkDown1'] = test['MarkDown1'].str.replace('NA','0').astype(np.float64)
test['MarkDown2'] = test['MarkDown2'].str.replace('NA','0').astype(np.float64)
test['MarkDown3'] = test['MarkDown3'].str.replace('NA','0').astype(np.float64)
test['MarkDown4'] = test['MarkDown4'].str.replace('NA','0').astype(np.float64)
test['MarkDown5'] = test['MarkDown5'].str.replace('NA','0').astype(np.float64)
test['CPI'] = test['CPI'].str.replace('NA','0').astype(np.float64)
test['Unemployment'] = test['Unemployment'].str.replace('NA','0').astype(np.float64)
test['Date']=test['Date'].astype(np.datetime64)


# In[42]:


test


# In[43]:


train['MarkDown1'] = train['MarkDown1'].str.replace('NA','0').astype(np.float64)
train['MarkDown2'] = train['MarkDown2'].str.replace('NA','0').astype(np.float64)
train['MarkDown3'] = train['MarkDown3'].str.replace('NA','0').astype(np.float64)
train['MarkDown4'] = train['MarkDown4'].str.replace('NA','0').astype(np.float64)
train['MarkDown5'] = train['MarkDown5'].str.replace('NA','0').astype(np.float64)
train['CPI'] = train['CPI'].astype(np.float64)
train['Unemployment'] = train['Unemployment'].astype(np.float64)


# In[44]:


test.dtypes


# In[45]:


train.dtypes


# In[46]:


# Calculate mean for each column
mean1 = test['MarkDown1'].mean()
mean2 = test['MarkDown2'].mean()
mean3 = test['MarkDown3'].mean()
mean4 = test['MarkDown4'].mean()
mean5 = test['MarkDown5'].mean()

mean11 = train['MarkDown1'].mean()
mean21 = train['MarkDown2'].mean()
mean31 = train['MarkDown3'].mean()
mean41 = train['MarkDown4'].mean()
mean51 = train['MarkDown5'].mean()


# In[47]:


train['MarkDown1'] = train['MarkDown1'].replace('0.00',mean1).astype(np.float64)
train['MarkDown2'] = train['MarkDown2'].replace('0.00',mean2).astype(np.float64)
train['MarkDown3'] = train['MarkDown3'].replace('0.00',mean3).astype(np.float64)
train['MarkDown4'] = train['MarkDown4'].replace('0.00',mean4).astype(np.float64)
train['MarkDown5'] = train['MarkDown5'].replace('0.00',mean5).astype(np.float64)
# 
test['MarkDown1'] = test['MarkDown1'].replace('0.00',mean11).astype(np.float64)
test['MarkDown2'] = test['MarkDown2'].replace('0.00',mean21).astype(np.float64)
test['MarkDown3'] = test['MarkDown3'].replace('0.00',mean31).astype(np.float64)
test['MarkDown4'] = test['MarkDown4'].replace('0.00',mean41).astype(np.float64)
test['MarkDown5'] = test['MarkDown5'].replace('0.00',mean51).astype(np.float64)


# In[48]:


test


# In[49]:


print(train.isnull().sum())
print("*"*30)
print(test.isnull().sum())


# # EDA

# The given code generates a heatmap of the correlation matrix for the columns in the train DataFrame using the seaborn library.
# 
# Here's what the code does line-by-line:
# 
#     fig, ax = plt.subplots(figsize=(10, 8)): This line creates a new figure and axes with a size of 10 inches by 8 inches using the subplots() function from the matplotlib library, and assigns the figure and axes objects to the variables fig and ax, respectively.
# 
#     heatmap = sns.heatmap(train.corr(), annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', linewidths=3, linecolor='black', ax=ax): This line generates a heatmap of the correlation matrix for the columns in the train DataFrame using the heatmap() function from the seaborn library. The corr() method of the train DataFrame is used to compute the correlation matrix. The annot=True argument specifies that the correlation coefficients should be displayed in the heatmap. The vmin=-1, vmax=1, and center=0 arguments set the minimum, maximum, and center values of the color scale, respectively. The cmap='coolwarm' argument sets the color map to use. The linewidths=3 and linecolor='black' arguments specify the width and color of the lines separating the cells in the heatmap, respectively. Finally, the ax=ax argument specifies that the heatmap should be plotted on the axes object ax.
# 
#     ax.set_title('Correlation Heatmap'): This line sets the title of the heatmap to "Correlation Heatmap" using the set_title() method of the ax object.
# 
#     plt.show(): This line displays the heatmap on the screen using the show() function from the matplotlib library.

# In[50]:


fig, ax = plt.subplots(figsize=(10, 8))

# Generate the heatmap with annotations and custom formatting
heatmap = sns.heatmap(train.corr(), annot=True, vmin=-1, vmax=1, center=0,
                      cmap='coolwarm', linewidths=3, linecolor='black', ax=ax)

# Set the title of the heatmap
ax.set_title('Correlation Heatmap')

# Display the heatmap
plt.show()



# In[51]:


train.corr()


# # This correlation matrix shows the correlation between pairs of variables in the dataset. The values in the table range from -1 to 1, where a value of 1 indicates a perfect positive correlation (when one variable increases, the other variable increases proportionally), a value of -1 indicates a perfect negative correlation (when one variable increases, the other variable decreases proportionally), and a value of 0 indicates no correlation between the variables.
# 
# In this specific matrix, we can see that there is a strong positive correlation between Size and Weekly_Sales (0.24) and between CPI and Unemployment (-0.21). On the other hand, there is a weak negative correlation between Store and Weekly_Sales (-0.085) and between Size and Store (-0.183).
# 
# Additionally, we can see that the markdowns (MarkDown1, MarkDown2, MarkDown3, MarkDown4, and MarkDown5) are positively correlated with Weekly_Sales, which suggests that markdowns may have a positive impact on sales. However, the strongest correlation is between MarkDown1 and MarkDown4 (0.84), which suggests that these two variables may be measuring similar things.
# 
# Finally, we can see that IsHoliday is weakly correlated with most variables, but has a moderate positive correlation with MarkDown3 (0.21) and MarkDown2 (0.21), which suggests that markdowns may be more likely to occur during holiday periods.
# 

# # MARKDOWN 1 AND 4 ARE HIGHLY COORELATED , WEEKELY SALES IS HIGHYLY IMPACTED BY DEPARTMENT SIZE AND TYPE OF DEPARTMENT

#  
# 
# Looking at the matrix, we can see that the strongest correlation with the target variable, Weekly_Sales, is with the Size feature (0.24). Other features that are moderately correlated with the target variable include CPI (-0.02), Unemployment (-0.03), and MarkDown1 (0.05).
# 
# There are also strong correlations between some of the other features. For example, MarkDown1 is strongly correlated with MarkDown4 (0.84) and Fuel_Price is moderately correlated with CPI (-0.16) and Size (-0.06).
# 
# It's important to note that correlation does not imply causation. Just because two variables are strongly correlated does not mean that one causes the other. Further analysis would be needed to establish any causal relationships.
# 

# In[52]:


sns.barplot(x=train["Weekly_Sales"],y=train["Type"])


# In[53]:


#type A catagory outlets have highest weekely sales


# In[54]:


train.plot(kind='line', x='Weekly_Sales', y='Store', alpha=0.5)


# In[55]:


train['Store'].value_counts(normalize=True).plot(kind = 'bar',fig=(4,5))


# In[56]:


#STORE NUMBER 13 HAS HIGHEST DATA AVALIABLE


import dtale as dt

import seaborn as sns
# df=sns.load_dataset(‘planets’)
dt.show(train, ignore_duplicate=True)


# In[57]:


#https://seaborn.pydata.org/generated/seaborn.pairplot.html
sns.pairplot(train, vars=['Weekly_Sales', 'Fuel_Price', 'Size', 'CPI', 'Dept', 'Temperature', 'Unemployment'],height=3, aspect=1)


# # feature extraction

# In[58]:


train.info()


# In[59]:


test.info()


# In[60]:


# Extract date features
# train['Day_week'] =train['Date'].dt.dayofweek
train['Date_month'] =train['Date'].dt.month 
# train['Date_year'] =train['Date'].dt.year
train['Date_day'] =train['Date'].dt.day 

# test['Day_week'] =test['Date'].dt.dayofweek
test['Date_month'] =test['Date'].dt.month 
# test['Date_year'] =test['Date'].dt.year
test['Date_day'] =test['Date'].dt.day 

# train.drop(['Date_dayofweek'],axis=1,inplace=True)
# test.drop(['Date_dayofweek'],axis=1,inplace=True)


# In[61]:


train


# In[62]:


test


# In[63]:


print(train.Type.value_counts())
print("*"*30)
print(test.Type.value_counts())


# In[64]:


print(train.IsHoliday.value_counts())
print("*"*30)
print(test.IsHoliday.value_counts())


# In[65]:


train_test_data = [train, test]


# In[66]:


train_test_data


# #   Converting Categorical Variable 'Type' into Numerical Variable 
#     For A=1 , B=2, C=3

# In[67]:


type_mapping = {"A": 1, "B": 2, "C": 3}
for dataset in train_test_data:
    dataset['Type'] = dataset['Type'].map(type_mapping)


# # This code snippet mapping the values in the "Type" column of each dataset in train_test_data to a numerical value according to the type_mapping dictionary.
# 
# For example, if the "Type" column contains the string "A" in a given dataset, it will be replaced with the number 1. Similarly, "B" will be replaced with 2, and "C" will be replaced with 3.
# 
# The modified datasets will have a new column called "Type" that contains the numerical values instead of the original string values.
# 

# # Converting Categorical Variable 'IsHoliday' into Numerical Variable

# In[68]:


type_mapping = {False: 0, True: 1}
for dataset in train_test_data:
    dataset['IsHoliday'] = dataset['IsHoliday'].map(type_mapping)


#     Creating Extra Holiday Variable.
#     If that week comes under extra holiday then 1(=Yes) else 0(=No)

# These lines of code are adding new columns to the train and test dataframes to indicate whether a particular date corresponds to a special event.
# 
# For instance, the first line of code creates a new column called Valentine_Day in the train dataframe, which is set to 1 if the Date column corresponds to Valentine's Day in any of the years 2010 to 2013, and 0 otherwise. Similarly, the second line of code creates a new column called diwali in the train dataframe, which is set to 1 if the Date column corresponds to the Diwali festival in any of the years 2010 to 2013, and 0 otherwise.
# 
# The same four lines of code are repeated for the test dataframe to create these columns in the test data as well.

# In[69]:


train['Valentine_Day'] = np.where((train['Date']==datetime(2010, 2, 12)) | (train['Date']==datetime(2011, 2, 11)) | (train['Date']==datetime(2012, 2, 10)) | (train['Date']==datetime(2013, 2, 8)),1,0)
train['diwali'] = np.where((train['Date']==datetime(2010, 9, 10)) | (train['Date']==datetime(2011, 9, 9)) | (train['Date']==datetime(2012, 9, 7)) | (train['Date']==datetime(2013, 9, 6)),1,0)
train['Guru_Nanak_Jayanti'] = np.where((train['Date']==datetime(2010, 11, 26)) | (train['Date']==datetime(2011, 11, 25)) | (train['Date']==datetime(2012, 11, 23)) | (train['Date']==datetime(2013, 11, 29)),1,0)
train['Christmas'] = np.where((train['Date']==datetime(2010, 12, 31)) | (train['Date']==datetime(2011, 12, 30)) | (train['Date']==datetime(2012, 12, 28)) | (train['Date']==datetime(2013, 12, 27)),1,0)
#........................................................................
test['Valentine_Day'] = np.where((test['Date']==datetime(2010, 2, 12)) | (test['Date']==datetime(2011, 2, 11)) | (test['Date']==datetime(2012, 2, 10)) | (test['Date']==datetime(2013, 2, 8)),1,0)
test['diwali'] = np.where((test['Date']==datetime(2010, 9, 10)) | (test['Date']==datetime(2011, 9, 9)) | (test['Date']==datetime(2012, 9, 7)) | (test['Date']==datetime(2013, 9, 6)),1,0)
test['Guru_Nanak_Jayanti'] = np.where((test['Date']==datetime(2010, 11, 26)) | (test['Date']==datetime(2011, 11, 25)) | (test['Date']==datetime(2012, 11, 23)) | (test['Date']==datetime(2013, 11, 29)),1,0)
test['Christmas'] = np.where((test['Date']==datetime(2010, 12, 31)) | (test['Date']==datetime(2011, 12, 30)) | (test['Date']==datetime(2012, 12, 28)) | (test['Date']==datetime(2013, 12, 27)),1,0)


# This code use to be altering the IsHoliday column in both the train and test dataframes. Specifically, it is setting the value of IsHoliday to True if any of the following holidays occurred on that date: Valentine's Day (Valentine_Day), Diwali (diwali), Guru Nanak Jayanti (Guru_Nanak_Jayanti), or Christmas (Christmas).
# 
# The | symbol between each holiday is a bitwise OR operator, which returns True if either of the operands are True.
# 
# So, for example, if Valentine_Day is True for a particular row in the train dataframe, the IsHoliday value for that row will be set to True as well.
# 
# It's worth noting that this code assumes that these holidays are already present as columns in the dataframes and have been appropriately labeled. If not, this code will not work as intended.

# In[70]:


# Altering the isHoliday value depending on these new holidays...
train['IsHoliday']=train['IsHoliday']|train['Valentine_Day']|train['diwali']|train['Guru_Nanak_Jayanti']|train['Christmas']
test['IsHoliday']=test['IsHoliday']|test['Valentine_Day']|test['diwali']|test['Guru_Nanak_Jayanti']|test['Christmas']


# In[71]:


train


# In[72]:


train.shape


# #week can have normal holiday or 4 events has holiday 
# #isholiday if 1 then 1 is ok but if week has one of 4 events then also isholiday is imputed as 1 

# In[73]:


print(train.Christmas.value_counts())
print(train.Valentine_Day.value_counts())
print(train.diwali.value_counts())
print(train.Guru_Nanak_Jayanti .value_counts())


# In[74]:


print(test.Christmas.value_counts())
print(test.Valentine_Day.value_counts())
print(test.diwali.value_counts())
print(test.Guru_Nanak_Jayanti.value_counts())


# In[75]:


# Since we have Imputed IsHoliday according to Extra holidays..These extra holiday variable has redundant..
# Droping the Extra holiday variables because its redundant..
dp=['Valentine_Day','diwali','Guru_Nanak_Jayanti','Christmas']
train.drop(dp,axis=1,inplace=True)
test.drop(dp,axis=1,inplace=True)


# # Feature Selection

# # Droping irrevelent variable:
# 
# -Since we have imputed markdown variables therefore we will not be removing the all markdown variables.
# -Removing MarkDown5 because its Highly Skewed

# In[76]:


features_drop=['Unemployment','CPI','MarkDown5']
train=train.drop(features_drop, axis=1)
test=test.drop(features_drop, axis=1)


# In[77]:


test


# In[78]:


train


# # Classification & Accuracy
#     Define training and testing set

# In[79]:


#### train X= Exery thing except Weekly_Sales
train_X=train.drop(['Weekly_Sales','Date'], axis=1)

#### train Y= Only Weekly_Sales 
train_y=train['Weekly_Sales'] 

#testing data
test_X=test.drop(['Date'],axis=1).copy()

train_X.shape, train_y.shape, test_X.shape


# In[80]:


train_X


# In[81]:


test_X                  


# # polynomial regression model

# In[82]:


# Extract the independent and dependent variables from the DataFrame
X = train_X.iloc[:].values
y = train_y.iloc[:].values


# # This code uses the PolynomialFeatures transformer from scikit-learn to create a new feature matrix X_poly that includes polynomial terms of the original features in X.
# 
# The degree of the polynomial terms is specified by the degree parameter, which is set to 3 in this example. This means that the transformer will create new features that are the products of the original features raised to powers from 0 to 3. For example, if the original feature matrix X had two features x1 and x2, the new feature matrix X_poly would include terms like 1, x1, x2, x1^2, x1x2, x2^2, x1^3, x1^2x2, x1x2^2, and x2^3.
# 
# Adding polynomial features can be useful for linear models when the relationship between the features and the target variable is non-linear. By including polynomial terms, the model can capture non-linear relationships between the features and the target variable.
# 
# Note that adding polynomial features can also increase the complexity of the model and potentially lead to overfitting, so it's important to use regularization or cross-validation to prevent overfitting.

# In[83]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
# Transform the independent variables to include polynomial terms
X_poly = poly.fit_transform(X)



# In[84]:


# Fit the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)


# In[85]:


# Print the coefficients of the model
print(poly_model.coef_)


# In[86]:


poly_model.score(X_poly,y)


# # Building models & comparing their RMSE values

# # .Linear Regression

# # This code trains a linear regression model (clf) on the training data (train_X and train_y) and then uses the trained model to make predictions on the test data (test_X). The predicted values are stored in the y_pred_linear variable.
# 
# The code then calculates the R^2 score of the trained model on the training data using the score() method of the linear regression object, and multiplies the score by 100 to express it as a percentage. The R^2 score is a measure of how well the model fits the training data, with higher scores indicating better fit.

# In[87]:


## Methood 1..
clf = LinearRegression()
clf.fit(train_X, train_y)
y_pred_linear=clf.predict(test_X)
acc_linear=clf.score(train_X, train_y)*100
print ('scorbe:'+str(acc_linear) + ' percent')


# In[88]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


# Fit a linear regression model
reg = LinearRegression().fit(train_X, train_y)

# Make predictions
y_pred = reg.predict(train_X)

# Calculate R-squared
r2 = r2_score(train_y, y_pred)

# Calculate mean squared error
mse = mean_squared_error(y, y_pred)

# Calculate root mean squared error
rmse = np.sqrt(mse)

# Calculate mean absolute error
mae = mean_absolute_error(y, y_pred)

# Print the results
print("R-squared:", r2)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)


# # . Random Forest

# # This code performs a randomized search for hyperparameters of a Random Forest Regressor using the RandomizedSearchCV function from scikit-learn. The hyperparameters that are being tuned are max_depth, n_estimators, min_samples_leaf, and min_samples_split. The scoring metric used to evaluate the models is the coefficient of determination (R^2).
# 
# After fitting the model to the training data (train_X and train_y), the code prints the best estimator found by the randomized search and the R^2 score on the training data.

# In[89]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()

from sklearn.model_selection import RandomizedSearchCV
param_grid = {'max_depth':[1,5,10,15,20,25,30],'n_estimators':[20,50,100],
  'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10]}
grid = RandomizedSearchCV(rf,param_grid ,scoring = 'r2')
grid.fit(train_X,train_y)


print(grid.best_estimator_)
print(grid.score(train_X, train_y))



# # 3. Decision Tree

# In[90]:


from sklearn.tree import DecisionTreeRegressor 
dt=DecisionTreeRegressor()
from sklearn.model_selection import GridSearchCV


# # This code performs a grid search for hyperparameters of a decision tree regressor using the GridSearchCV function from scikit-learn. The hyperparameters that are being tuned are max_depth, max_features, min_samples_leaf, and min_samples_split. The scoring metric used to evaluate the models is the coefficient of determination (R^2).
# 
# After fitting the model to the training data (train_X and train_y), the code prints the best estimator found by the grid search and the R^2 score on the training data.
# 
# The predict() method is then used to make predictions on the test data (test_X) using the best estimator found by the grid search, and the predicted values are stored in the var variable.

# In[91]:


param_grid = {'max_depth':[1,5,10,15,20,25,30],
 'max_features': ['auto', 'sqrt','log2'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10]}
grid = GridSearchCV(estimator = dt,param_grid=param_grid ,scoring = 'r2')
grid.fit(train_X, train_y)
var=grid.predict(test_X)

print(grid.best_estimator_)
print(grid.score(train_X, train_y))


# In[92]:


dt=DecisionTreeRegressor(max_depth=10, max_features='auto', min_samples_leaf=4,
                      min_samples_split=10)
dt.fit(train_X, train_y)
     


# In[93]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import median_absolute_error
preds = dt.predict(train_X)
print("R2 score : %.2f" % r2_score(train_y,preds ))
print("Mean squared error: %.2f" % mean_squared_error(train_y,preds))


# # Comparing Models
# Let's compare the accuracy score of all the regression models used above.

# In[94]:


models = pd.DataFrame({
    'Model': ['Linear Regression','Random Forest','Decision Tree'],
    
    'Score': [acc_linear, acc_rf,acc_dt]
    })

models.sort_values(by='Score', ascending=False)


# # **Predicting Sales value for test data based on highest score model.**

# In[ ]:


# Prediction value using Random Forest model..
# submission = pd.DataFrame({
#         "Store_Dept_Date": test.Store.astype(str)+'_'+test.Dept.astype(str)+'_'+test.Date.astype(str),
#         "Weekly_Sales": y_pred_rf
#     })

# submission.to_csv('weekly_sales predicted.csv', index=False)
# submission.to_excel('Weekly_sales Pred',index=False)


# In[ ]:


import pickle


# # This code saves a trained machine learning model to a file using the pickle module in Python.
# 
# The trained model object that is being saved is clf_random, and the file name is 'trained_model.sav'. The 'wb' option in the open() function specifies that the file should be opened for writing in binary mode.
# 
# After the model is saved to the file, it can be loaded into memory at a later time using the pickle.load() function. This can be useful for deploying the trained model in a production environment, or for reusing the model in a different Python script without having to retrain it.

# In[ ]:


filename='trained_model.sav'
pickle.dump(clf_random,open(filename,'wb'))


# In[ ]:


#load saved model
loaded_model=pickle.load(open('trained_model.sav','rb'))


# In[ ]:


input_data=(45,0,9,2,118221,50,3.046,10000,10000,10000,10000,5,4,2012,15)


# In[ ]:


#changing input data into numpy array
np_array=np.asarray(input_data)


# In[ ]:


np_array.ndim


# In[ ]:


#reshape array as we are predecting for one instance
input_reshaped=np_array.reshape(1,-1)


# In[ ]:


input_reshaped


# In[ ]:


input_reshaped.ndim


# In[ ]:


predict=loaded_model.predict(input_reshaped)


# In[ ]:


predict


# In[ ]:




