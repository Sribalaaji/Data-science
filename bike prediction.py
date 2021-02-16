# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:26:36 2020

@author: USER
"""

# bike demand prediction
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import math

#step 1 - read the data

bikes = pd.read_csv("hour.csv")

#step 2 - prelim analysis and feature selection

bikes_prep = bikes.copy()

bikes_prep = bikes_prep.drop(["index","date","casual","registered"] , axis= 1)

#basic checks of missing value

bikes_prep.isnull().sum()

#visualise data using pandas histograms

bikes_prep.hist(rwidth= 0.8)

plt.tight_layout()

#visualising the continuous feathers with demand

plt.subplot(2,2, 1)
plt.title("Temperature Vs Demand")
plt.scatter(bikes_prep["temp"],bikes_prep["demand"],s=2 , c ="g")

plt.subplot(2,2,2)
plt.title("aTemp Vs Demand") 
plt.scatter(bikes_prep["atemp"],bikes_prep["demand"],s=2 , c= "b")

plt.subplot(2,2,3)
plt.title("Humidity Vs Demand")
plt.scatter(bikes_prep["humidity"],bikes_prep["demand"], s=2, c = "m")

plt.subplot(2,2,4)
plt.title("Windspeed Vs Demand")
plt.scatter(bikes_prep["windspeed"],bikes_prep["demand"], s=2 , c = "c")

plt.tight_layout()


# plot and visualise the categorical features
colours = ["g", "r", "m","b"]
plt.subplot(3, 3, 1)
plt.title("Average demand per season")
#create list of unique value in season
cat_list = bikes_prep["season"].unique()

#average the demand based on season
cat_average = bikes_prep.groupby("season").mean()["demand"]
plt.bar(cat_list,cat_average, color= colours)

plt.subplot(3,3,2)
plt.title("Average demand per month")
cat_list = bikes_prep["month"].unique()
cat_average = bikes_prep.groupby("month").mean()["demand"]
plt.bar(cat_list,cat_average , color=colours)

plt.subplot(3,3,3)
plt.title("Average demand per holiday")
cat_list = bikes_prep["holiday"].unique()
cat_average = bikes_prep.groupby("holiday").mean()["demand"]
plt.bar(cat_list,cat_average,color = colours)

plt.subplot(3,3,4)
plt.title("Average demand per weekday")
cat_list = bikes_prep["weekday"].unique()
cat_average = bikes_prep.groupby("weekday").mean()["demand"]
plt.bar(cat_list,cat_average,color = colours)

plt.subplot(3,3,5)
plt.title("Average demand per weather")
cat_list = bikes_prep["weather"].unique()
cat_average = bikes_prep.groupby("weather").mean()["demand"]
plt.bar(cat_list,cat_average,color = colours)

plt.subplot(3,3,6)
plt.title("Average demand per year")
cat_list = bikes_prep["year"].unique()
cat_average = bikes_prep.groupby("year").mean()["demand"]
plt.bar(cat_list,cat_average,color = colours)

plt.subplot(3,3,7)
plt.title("Average demand per hour")
cat_list = bikes_prep["hour"].unique()
cat_average = bikes_prep.groupby("hour").mean()["demand"]
plt.bar(cat_list,cat_average,color = colours)

plt.subplot(3,3,8)
plt.title("Average demand per workingday")
cat_list = bikes_prep["workingday"].unique()
cat_average = bikes_prep.groupby("workingday").mean()["demand"]
plt.bar(cat_list,cat_average,color = colours)

plt.tight_layout()


#check for outliers

bikes_prep["demand"].describe()

bikes_prep["demand"].quantile([0.05,0.1,0.15,0.2,0.9,0.95,0.99])

# step 4 - Multilinear regression  assumtion

#Linearity using correlation coefficient matrix using corr

correlation = bikes_prep[["temp", "atemp", "humidity", "windspeed", "demand"]].corr()

bikes_prep = bikes_prep.drop(["atemp","windspeed", "workingday", "year", "weekday"], axis = 1)

# Check the auto correlation in demand using acorr

df1 = pd.to_numeric(bikes_prep["demand"], downcast="float")

plt.acorr(df1, maxlags =12)


#Step 6 - Create / Modify the new feature

df2 = np.log(df1)

plt.figure()
df1.hist(rwidth=0.9 , bins = 20)
plt.figure()
df2.hist(rwidth = 0.9, bins= 20)

bikes_prep["demand"] = np.log(bikes_prep["demand"])

# Autocorrelation in the demand column

t_1 = bikes_prep["demand"].shift(+1).to_frame()
t_1.columns = ["t-1"]

t_2 = bikes_prep["demand"].shift(+2).to_frame()
t_2.columns = ["t-2"]

t_3 = bikes_prep["demand"].shift(+3).to_frame()
t_3.columns = ["t-3"]

bikes_prep_lag = pd.concat([bikes_prep,t_1,t_2,t_3], axis = 1)
bikes_prep_lag = bikes_prep_lag.dropna()

#
# Step 7 - create the dummy variables and drop first
#          to avoid dummy variables trap using get_dummy
# columns - season,month,holiday,hour,weather

bikes_prep_lag.dtypes

bikes_prep_lag[["season","holiday","month","hour","weather"]]= \
    bikes_prep_lag[["season","holiday","month","hour","weather"]].astype("category")
                              
bikes_prep_lag = pd.get_dummies(bikes_prep_lag, drop_first= True )             


#Step 8 - Split data to X and Y for train and test

Y = bikes_prep_lag[["demand"]]
X = bikes_prep_lag.drop(["demand"], axis=1)

#Create the size of 70 % of the data

tr_size = 0.7* len(X)
tr_size = int(tr_size)

#Creating train and test

X_train = X.values[0: tr_size]
X_test = X.values[tr_size : len(X)]

Y_train = Y.values[0: tr_size]
Y_test = Y.values[tr_size : len(Y)]


#step 9 - Fit or train the data and score

from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()

std_reg.fit(X_train,Y_train)

#R square
r2_train = std_reg.score(X_train,Y_train)
r2_test = std_reg.score(X_test,Y_test)

#predict the data
Y_predict = std_reg.predict(X_test)

#Root mean square error
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(Y_test,Y_predict))

#Final step - to calculate RMSLE

Y_test_e = []
Y_predict_e = []

#
for i in range(0 , len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(Y_predict[i]))
    
log_sum = 0.0
#RMSLE
for i in range(0, len(Y_test_e)):
    log_a = math.log(Y_test_e[i] + 1)
    log_p = math.log(Y_predict_e[i] +1)
    log_diff = (log_p - log_a)**2
    log_sum = log_sum + log_diff
    
rmsle = math.sqrt(log_sum/len(Y_test))

print(rmsle)

