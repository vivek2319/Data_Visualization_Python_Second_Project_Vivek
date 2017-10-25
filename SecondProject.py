# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
from matplotlib import *
import matplotlib.pyplot as plt
import matplotlib as plt
import matplotlib as matplot
import seaborn as sns
from pandas.tools.plotting import radviz
%matplotlib inline

# Read the analytics csv file and store our dataset into a dataframe called "df"
df = pd.read_csv('G:\BDA\Data Science Must have books\Mercyhurst diary\Fall 18\Data visualization using Python\Second Project\HR_comma_sep.csv', index_col=None)

# Check to see if there are any missing values in our data set
df.isnull().any()


# Get a quick overview of what we are dealing with in our dataset
df.head()

# Output can be viewed at: https://tinyurl.com/yck873n6 

# Renaming certain columns for better readability
df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })




# Move the reponse variable "turnover" to the front of the table
front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)
df.head()

# output can be viewed at: https://tinyurl.com/ybws47hc

#Exploring the data


# The dataset contains 10 columns and 14999 observations
df.shape



# Check the type of our features. 
df.dtypes

# Output can be viewed at: https://tinyurl.com/ydccbv4s

# Looks like about 76% of employees stayed and 24% of employees left. 
# NOTE: When performing cross validation, its important to maintain this turnover ratio
turnover_rate = df.turnover.value_counts() / len(df)
turnover_rate


# Display the statistical overview of the employees
df.describe()

# Output can be viewed at: https://tinyurl.com/y8e5b3dp

# Overview of summary (Turnover V.S. Non-turnover)
turnover_Summary = df.groupby('turnover')
turnover_Summary.mean()

# Output can be viewed at: https://tinyurl.com/y832k99s 

# Correlation matrix and heatmap
# Correlation Matrix
corr = df.corr()

corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

corr

#Output for correlation matrix can be viewed at: https://tinyurl.com/yc2r5bqs

#Output for heatmap can be viewed at: https://tinyurl.com/y9rscxcn

# Graph Employee Satisfaction
g=sns.distplot(df.satisfaction, kde=False, color="g").set_title('Employee Satisfaction Distribution')
g

#Output can be viewed at: https://tinyurl.com/yaj66q57

#Looks like the average employees who stayed worked about 200hours/month. Those that had a turnover worked about 250hours/month and 150hours/month

v=sns.boxplot(x="yearsAtCompany", y="averageMonthlyHours", hue="turnover", data=df)
v
# Output can be viewed at: https://tinyurl.com/yavotboe

#Lets have a look at lmplot satisfaction vs evaluation 
p = sns.lmplot(x='satisfaction', y='evaluation', data=df,
           fit_reg=True, # Print regression line
           hue='turnover')   # Color by evolution stage
p

# Output can be viewed at: https://tinyurl.com/y9nucr28

#Summary: We have used different kinds of plots here using variety of packages. 
#If you have any suggestions or improvements please feel free to do so.
#For more analysis: please visit https://tinyurl.com/y8b2yz8v
