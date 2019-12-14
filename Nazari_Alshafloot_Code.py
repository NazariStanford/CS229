# Code for CS229 Final
import numpy as np
import sklearn as skl
import skimage as ski
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_regression

# Seed the randomness of the simulation so this outputs the same thing each time
np.random.seed(0)

# Reading the data
# You need to put the location of the data on your computer between ''
Data = pd.read_excel ('RawData.xlsx', sheet_name='All_OL') 
df = pd.DataFrame(Data, columns = ['Group', 'Gross', 'Oil', 'Water', 'waterCut', 'GOR', 'GLR',
                                   'FWHP', 'Press', 'Temp', 'chokeSize', 'CHK_US', 'CHK_DS'])
df2 = pd.DataFrame(Data, columns = ['Gross', 'GLR', 'FWHP', 'Temp', 'chokeSize', 'GOR'])

# Reading and Defining Features
Gross = df[['Gross']]
GLR = df[['GLR']]
FWHP = df[['FWHP']]
Temp = df[['Temp']]
chokeSize = df[['chokeSize']]
PDS = df[['CHK_DS']]
WC = df[['waterCut']]
WC = 100 - WC.to_numpy()
df_arr = df.to_numpy()
DP = df_arr[ : ,7] - df_arr[ : ,12]
Pratio = df_arr[ : ,12]/df_arr[ : ,7]
Crit = Pratio <= 0.5
SubCrit = Pratio > 0.5
df_arr2 = df_arr[Crit,:]
Gross = Gross[ : ]
GLR = GLR[ : ]
FWHP = FWHP[ : ]
Temp = Temp[ : ]
chokeSize = chokeSize[ : ]

PTSGLR = FWHP.to_numpy()*Temp.to_numpy()*chokeSize.to_numpy()/GLR.to_numpy()
Gillbert = 0.1*FWHP.to_numpy()*np.power(chokeSize.to_numpy(), 1.89)/np.power(GLR.to_numpy(), 0.546)
Slog = np.log(chokeSize)
GLRwc = GLR.to_numpy()*WC
DPPratio = DP/df_arr[ : ,7]
TPratio = df_arr[ : ,9]/df_arr[ : ,7]
Slog_inv = 1.0/Slog

# Gross/Pressure
array1 = np.array(df_arr[ : ,7] - df_arr[ : ,12])
DP = np.where(array1==0, 1, array1) 
GrossP = df_arr[:,1]/(df_arr[:,7])
df2[['Gross']] = GrossP

# Plotting Gross flowrate vs. different features
fig = plt.figure()
fig.set_size_inches(20.5, 3.5, forward=True)

plt.subplot(1, 5, 1)
plt.plot(GLR, Gross, 'o', color='blue')
plt.xlabel('GLR (SCF/STB)')
plt.ylabel('Gross (bbl/Day)')
plt.title('Gross vs. GLR')

plt.subplot(1, 5, 2)
plt.plot(FWHP, Gross, 'o', color='red')
plt.xlabel('FWHP (psi)')
plt.title('Gross vs. FWHP')

plt.subplot(1, 5, 3)
plt.plot(Temp, Gross, 'o', color='yellow')
plt.xlabel('Temperature (F)')
plt.title('Gross vs. Temperature')

plt.subplot(1, 5, 4)
plt.plot(chokeSize, Gross, 'o', color='green')
plt.xlabel('Choke Size')
plt.title('Gross vs. Choke Size')

plt.subplot(1, 5, 5)
plt.plot(DP, Gross, 'o', color='purple')
plt.xlabel('DP(psi)')
plt.title('Gross vs. DP')

plt.savefig('All in one_All Data.png')
plt.close()

# Adding all possible features and pd to np and random permutation
df2_arr = df2.to_numpy()
df2_arr = np.append(df2_arr, np.reshape(DP, (DP.shape[0],1)), axis=1)
df2_arr = np.append(df2_arr, np.reshape(WC, (WC.shape[0],1)), axis=1)
df2_arr = np.append(df2_arr, np.reshape(PTSGLR, (WC.shape[0],1)), axis=1)
df2_arr = np.append(df2_arr, np.reshape(Gillbert, (WC.shape[0],1)), axis=1)
df2_arr = np.append(df2_arr, np.reshape(Slog, (WC.shape[0],1)), axis=1)
df2_arr = np.append(df2_arr, np.reshape(GLRwc, (WC.shape[0],1)), axis=1)
df2_arr = np.append(df2_arr, np.reshape(DPPratio, (WC.shape[0],1)), axis=1)
df2_arr = np.append(df2_arr, np.reshape(TPratio, (WC.shape[0],1)), axis=1)
df2_arr = np.append(df2_arr, np.reshape(Slog_inv, (WC.shape[0],1)), axis=1)
df2_arr = np.append(df2_arr, np.reshape(Crit+1, (WC.shape[0],1)), axis=1)

# Fields Info
#Field A: 0,1488
#Field B: 1489,3058
#Field C: 3059:
BeginIdx = 0
EndIdx = -1
df2_arr = df2_arr[BeginIdx:EndIdx,:]
# For calling fields A and C
#df2_arr = df2_arr[np.r_[0:1488,3059:-1],:]

df2_arr = df2_arr.astype(int)
for i in range(df2_arr.shape[0]):
    for j in range(df2_arr.shape[1]):
        if df2_arr[i,j] == 0:
            df2_arr[i,j] = 1
df2_arr = df2_arr[:]
df2_arr = df2_arr[np.random.permutation(df2_arr.shape[0]), :]

# Calculating Mean and std
df2_mean = np.mean(df2_arr, axis = 0)
df2_std = np.std(df2_arr, axis = 0)

# Excluding outliers (out of 3sigma)
df2_filt_3s = [x for x in df2_arr if (x[0] > df2_mean[0] - 3 * df2_std[0])]
df2_filt_3s = [x for x in df2_filt_3s if (x[0] < df2_mean[0] + 3 * df2_std[0])]
for i in range(5):
    df2_filt_3s = [x for x in df2_filt_3s if (x[i+1] > df2_mean[i+1] - 3 * df2_std[i+1])]
    df2_filt_3s = [x for x in df2_filt_3s if (x[i+1] < df2_mean[i+1] + 3 * df2_std[i+1])]
    
ymax = 10000
df2_filt_3s = [x for x in df2_filt_3s if (x[0]*x[2] < ymax)]
df2_filt_3s = np.array(df2_filt_3s)
df2_filt = df2_filt_3s

n = df2_filt.shape[0]
m = int(n * 0.8)
Idx = [1,3,4,5,6,7,12,15]

train_y = df2_filt[:m,0]
train_x = df2_filt[:m,Idx]
test_y = df2_filt[m:,0]
test_x = df2_filt[m:,Idx]

# Gilbert
Gillbert_train = df2_filt[:m,9]
Gillbert_test = df2_filt[m:,9]

# Plotting measured vs. Predicted Gross flowrate
plt.figure()
plt.scatter(test_y*df2_filt[m:,2], Gillbert_test, color='blue', linewidth=1)
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2) 
plt.savefig('Gillbert Correlation.png')
plt.close()
R2Score = r2_score(test_y*df2_filt[m:,2], Gillbert_test)
print('Score:', R2Score)
Correlation_Coefficient_Gilbert = np.corrcoef(test_y*df2_filt[m:,2], Gillbert_test)[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_Gilbert)

# Features' Standardization
scaler = preprocessing.StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

# Linear Regression
Linreg = linear_model.LinearRegression()
Linreg.fit(train_x, train_y)
Coefficients = Linreg.coef_
Bias = Linreg.intercept_

# Make predictions using the testing set
Gross_pred = Linreg.predict(test_x)

# Plot outputs
plt.figure()
LinregPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Linear Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_Linreg = Linreg.score(train_x, train_y)
print('Train_Score:',Train_Score_Linreg)
Test_Score_Linreg = Linreg.score(test_x, test_y)
print('Test_Score:',Test_Score_Linreg)
Correlation_Coefficient_Linreg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_Linreg)
plt.savefig('Linear Resgression.png')
plt.close()

# Ridge Regression
Ridgreg = linear_model.Ridge(alpha=.5)
Ridgreg.fit(train_x, train_y)
Coefficients = Ridgreg.coef_
Bias = Ridgreg.intercept_

# Make predictions using the testing set
Gross_pred = Ridgreg.predict(test_x)

# Plot outputs
plt.figure()
RidgregPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Ridge Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_Ridgreg = Ridgreg.score(train_x, train_y)
print('Train_Score:',Train_Score_Ridgreg)
Test_Score_Ridgreg = Ridgreg.score(test_x, test_y)
print('Test_Score:',Test_Score_Ridgreg)
Correlation_Coefficient_Ridgreg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_Ridgreg)
plt.savefig('Ridge Resgression.png')
plt.close()

# Bayesian Regression
Bayreg = linear_model.BayesianRidge()
Bayreg.fit(train_x, train_y)
Coefficients = Bayreg.coef_
Bias = Bayreg.intercept_

# Make predictions using the testing set
Gross_pred = Bayreg.predict(test_x)

# Plot outputs
plt.figure()
BayregPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Bayesian Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_Bayreg = Bayreg.score(train_x, train_y)
print('Train_Score:',Train_Score_Bayreg)
Test_Score_Bayreg = Bayreg.score(test_x, test_y)
print('Test_Score:',Test_Score_Bayreg)
Correlation_Coefficient_Bayreg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_Bayreg)
plt.savefig('Bayesian Resgression.png')
plt.close()

# Polynomial Linear Regression
PolyLinreg = Pipeline([('poly', PolynomialFeatures(degree = 2)), ('linear', LinearRegression(fit_intercept=False))])
PolyLinreg.fit(train_x, train_y)
Coefficients = PolyLinreg.named_steps['linear'].coef_

# Make predictions using the testing set
Gross_pred = PolyLinreg.predict(test_x)

# Plot outputs
plt.figure()
PolyLinregPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Polynomial Linear Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_PolyLinreg = PolyLinreg.score(train_x, train_y)
print('Train_Score:',Train_Score_PolyLinreg)
Test_Score_PolyLinreg = PolyLinreg.score(test_x, test_y)
print('Test_Score:',Test_Score_PolyLinreg)
Correlation_Coefficient_PolyLinreg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_PolyLinreg)
plt.savefig('Polynomial Linear Resgression.png')
plt.close()

# Polynomial Ridge Regression
BestAlpha = 0
BestScore = 0
BestDegree = 0
for i in range(0,5,1):
    for j in range(1,5):
        alpha = i/10
        PolyRidgreg = Pipeline([('poly', PolynomialFeatures(degree = j)), ('linear', Ridge(alpha=alpha,fit_intercept=False))])
        PolyRidgreg.fit(train_x, train_y)
        Coefficients = PolyRidgreg.named_steps['linear'].coef_
        #print(Coefficients)

        # Make predictions using the testing set
        Gross_pred = PolyRidgreg.predict(test_x)
        Test_Score = PolyRidgreg.score(test_x, test_y)
        if Test_Score>BestScore:
            BestAlpha = alpha
            BestScore = Test_Score
            BestDegree = j
print(BestAlpha)
print(BestDegree)
print(BestScore)

PolyRidgreg = Pipeline([('poly', PolynomialFeatures(degree = BestDegree)), ('linear', Ridge(alpha=BestAlpha,fit_intercept=False))])
PolyRidgreg.fit(train_x, train_y)
Coefficients = PolyRidgreg.named_steps['linear'].coef_
#print(Coefficients)

# Make predictions using the testing set
Gross_pred = PolyRidgreg.predict(test_x)
# Plot outputs
plt.figure()
PolyRidgregPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Polynomial Ridge Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_PolyRidgreg = PolyRidgreg.score(train_x, train_y)
print('Train_Score:',Train_Score_PolyRidgreg)
Test_Score_PolyRidgreg = PolyRidgreg.score(test_x, test_y)
print('Test_Score:',Test_Score_PolyRidgreg)
Correlation_Coefficient_PolyRidgreg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_PolyRidgreg)
plt.savefig('Polynomial Ridge Resgression.png')
plt.close()

# Multi Layer Perceptron Regression
MLPreg = MLPRegressor(hidden_layer_sizes = (10,),
                                       activation = 'relu',
                                       solver = 'adam',
                                       learning_rate = 'constant',
                                       max_iter = 1000,
                                       learning_rate_init = 0.001,
                                       alpha = 0.1)
MLPreg.fit(train_x, train_y)

# Make predictions using the testing set
Gross_pred = MLPreg.predict(test_x)

# Plot outputs
plt.figure()
MLPregPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Multi Layer Perceptron Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_MLPreg = MLPreg.score(train_x, train_y)
print('Train_Score:',Train_Score_MLPreg)
Test_Score_MLPreg = MLPreg.score(test_x, test_y)
print('Test_Score:',Test_Score_MLPreg)
Correlation_Coefficient_MLPreg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_MLPreg)
plt.savefig('Multi Layer Perceptron Resgression.png')
plt.close()

# Nearest Neighbor Regression
neighReg = KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto', 
                               leaf_size=30, p=2, metric='minkowski', 
                               metric_params=None, n_jobs=None)
neighReg.fit(train_x, train_y)


# Make predictions using the testing set
Gross_pred = neighReg.predict(test_x)

# Plot outputs
plt.figure()
neighRegPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Nearest Neighbor Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_neighReg = neighReg.score(train_x, train_y)
print('Train_Score:',Train_Score_neighReg)
Test_Score_neighReg = neighReg.score(test_x, test_y)
print('Test_Score:',Test_Score_neighReg)
Correlation_Coefficient_neighReg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_neighReg)
plt.savefig('Nearest Neighbor Regression.png')
plt.close()

# Random Forrest Regression
RFreg = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, 
                               min_samples_split=2, min_samples_leaf=1, 
                               min_weight_fraction_leaf=0.0, max_features='auto', 
                               max_leaf_nodes=None, min_impurity_decrease=0.0, 
                               min_impurity_split=None, bootstrap=True, oob_score=False, 
                               n_jobs=None, random_state=None, verbose=0, warm_start=False)
RFreg.fit(train_x, train_y)

# Make predictions using the testing set
Gross_pred = RFreg.predict(test_x)

# Plot outputs
plt.figure()
RFregPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Random Forrest Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_RFreg = RFreg.score(train_x, train_y)
print('Train_Score:',Train_Score_RFreg)
Test_Score_RFreg = RFreg.score(test_x, test_y)
print('Test_Score:',Test_Score_RFreg)
Correlation_Coefficient_RFreg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_RFreg)
plt.savefig('Random Forrest Resgression.png')
plt.close()

# Gradient Tree Boosting Regression
GTBreg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, 
                                   max_depth=10, random_state=0, loss='ls')
GTBreg.fit(train_x, train_y)

# Make predictions using the testing set
Gross_pred = GTBreg.predict(test_x)

# Plot outputs
plt.figure()
GTBregPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Gradient Tree Boosting')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_GTBreg = GTBreg.score(train_x, train_y)
print('Train_Score:',Train_Score_GTBreg)
Test_Score_GTBreg = GTBreg.score(test_x, test_y)
print('Test_Score:',Test_Score_GTBreg)
Correlation_Coefficient_GTBreg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_GTBreg)
plt.savefig('Gradient Tree Boosting.png')
plt.close()

# Extra Trees Regression
XTreg = ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None, 
                            min_samples_split=2, min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, max_features='auto', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, 
                            min_impurity_split=None, bootstrap=False, oob_score=False, 
                            n_jobs=None, random_state=None, verbose=0, warm_start=False)
XTreg.fit(train_x, train_y)

# Make predictions using the testing set
Gross_pred = XTreg.predict(test_x)

# Plot outputs
plt.figure()
XTregPlot = plt.scatter(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Extra Trees Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_XTreg = XTreg.score(train_x, train_y)
print('Train_Score:',Train_Score_XTreg)
Test_Score_XTreg = XTreg.score(test_x, test_y)
print('Test_Score:',Test_Score_XTreg)
Correlation_Coefficient_XTreg = np.corrcoef(test_y*df2_filt[m:,2], Gross_pred*df2_filt[m:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_XTreg)
plt.savefig('Extra Trees Resgression.png')
plt.close()

#plotting Results
fig = plt.figure()
fig.set_size_inches(20.5, 7, forward=True)

plt.subplot(2, 5, 1)
LinregPlot = plt.scatter(test_y*df2_filt[m:,2], Linreg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Linear Regression')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(2, 5, 2)
RidgregPlot = plt.scatter(test_y*df2_filt[m:,2], Ridgreg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Ridge Regression')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(2, 5, 3)
BayregPlot = plt.scatter(test_y*df2_filt[m:,2], Bayreg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Bayesian Regression')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(2, 5, 4)
PolyLinregPlot = plt.scatter(test_y*df2_filt[m:,2], PolyLinreg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Polynomial Linear Regression')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(2, 5, 5)
PolyRidgregPlot = plt.scatter(test_y*df2_filt[m:,2], PolyRidgreg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Polynomial Ridge Regression')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(2, 5, 6)
MLPregPlot = plt.scatter(test_y*df2_filt[m:,2], MLPreg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Multi Layer Perceptron Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(2, 5, 7)
RFregPlot = plt.scatter(test_y*df2_filt[m:,2], RFreg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Random Forrest Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(2, 5, 8)
GTBregPlot = plt.scatter(test_y*df2_filt[m:,2], GTBreg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Gradient Tree Boosting')
plt.xlabel('Measured Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(2, 5, 9)
neighRegPlot = plt.scatter(test_y*df2_filt[m:,2], neighReg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Nearest Neighbor Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(2, 5, 10)
XTregPlot = plt.scatter(test_y*df2_filt[m:,2], XTreg.predict(test_x)*df2_filt[m:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Extra Trees Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.savefig('All Results - All Data.png')
plt.close()

# Plotting Top-3 Models Results
fig = plt.figure()
fig.set_size_inches(12.3, 3.5, forward=True)

plt.subplot(1, 3, 1)
XTregPlot = plt.scatter(test_y*df2_filt[m:,2], XTreg.predict(test_x)*df2_filt[m:,2], color='green', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Extra Trees Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(1, 3, 2)
RFregPlot = plt.scatter(test_y*df2_filt[m:,2], RFreg.predict(test_x)*df2_filt[m:,2], color='green', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Random Forrest Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.subplot(1, 3, 3)
neighRegPlot = plt.scatter(test_y*df2_filt[m:,2], neighReg.predict(test_x)*df2_filt[m:,2], color='green', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Nearest Neighbor Regression')
plt.xlabel('Measured Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))

plt.savefig('Top 3 - All Fields.png')
plt.close()

n = df2_filt.shape[0]
m = int(n * 0.8)
print(n)
v = int(n * 0.9)
Idx = [1,3,4,5,6,7,12,15]

train_y = df2_filt[:m,0]
train_x = df2_filt[:m,Idx]
test_y = df2_filt[m:v,0]
test_x = df2_filt[m:v,Idx]
valid_y = df2_filt[v:,0]
valid_x = df2_filt[v:,Idx]

# Extra Trees Regression Hyperparameter Tuning
BestEst = 0
BestScore = 0
BestSplit = 0
BestLeaf = 0
for i in range(100,150,10):
    for j in range(2,4,1):
        for k in range(1,3,1):
            XTreg = ExtraTreesRegressor(n_estimators=i, criterion='mse', max_depth=None, 
                                min_samples_split=j, min_samples_leaf=k, 
                                min_weight_fraction_leaf=0.0, max_features='auto', 
                                max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                min_impurity_split=None, bootstrap=False, oob_score=False, 
                                n_jobs=None, random_state=None, verbose=0, warm_start=False)

            XTreg.fit(train_x, train_y)

            # Make predictions using the testing set
            Gross_pred = XTreg.predict(test_x)
            Test_Score = XTreg.score(test_x, test_y)

            if Test_Score>BestScore:
                BestEst = i
                BestScore = Test_Score
                BestSplit = j
                BestLeaf = k
print(BestEst)
print(BestSplit)
print(BestScore)
print(BestLeaf)
XTreg = ExtraTreesRegressor(n_estimators=BestEst, criterion='mse', max_depth=None, 
                            min_samples_split=BestSplit, min_samples_leaf=BestLeaf, 
                            min_weight_fraction_leaf=0.0, max_features='auto', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, 
                            min_impurity_split=None, bootstrap=False, oob_score=False, 
                            n_jobs=None, random_state=None, verbose=0, warm_start=False)
XTreg.fit(train_x, train_y)

# Make predictions using the testing set
Gross_pred = XTreg.predict(valid_x)

# Plot outputs
plt.figure()
XTregPlot = plt.scatter(valid_y*df2_filt[v:,2], Gross_pred*df2_filt[v:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Extra Trees Regression Hyper Tuned')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_XTreg = XTreg.score(train_x, train_y)
print('Train_Score:',Train_Score_XTreg)
Test_Score_XTreg = XTreg.score(test_x, test_y)
print('Test_Score:',Test_Score_XTreg)
Valid_Score_XTreg = XTreg.score(valid_x, valid_y)
print('Valid_Score:',Valid_Score_XTreg)
Correlation_Coefficient_XTreg = np.corrcoef(valid_y*df2_filt[v:,2], Gross_pred*df2_filt[v:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_XTreg)
plt.savefig('Extra Trees Resgression Hyper Tuned.png')
plt.close()

# Random Forest Regression Hyperparameter Tuning
BestEst = 0
BestScore = 0
BestSplit = 0
BestLeaf = 0
for i in range(100,150,10):
    for j in range(2,5,1):
        for k in range(1,5,1):
            RFreg = RandomForestRegressor(n_estimators=i, criterion='mse', max_depth=None, 
                                min_samples_split=j, min_samples_leaf=k, 
                                min_weight_fraction_leaf=0.0, max_features='auto', 
                                max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                min_impurity_split=None, bootstrap=True, oob_score=False, 
                                n_jobs=None, random_state=None, verbose=0, warm_start=False)

            RFreg.fit(train_x, train_y)

            # Make predictions using the testing set
            Gross_pred = RFreg.predict(test_x)
            Test_Score = RFreg.score(test_x, test_y)

            if Test_Score>BestScore:
                BestEst = i
                BestScore = Test_Score
                BestSplit = j
                BestLeaf = k
print(BestEst)
print(BestSplit)
print(BestScore)
print(BestLeaf)

RFreg = RandomForestRegressor(n_estimators=BestEst, criterion='mse', max_depth=None, 
                               min_samples_split=BestSplit, min_samples_leaf=BestLeaf, 
                               min_weight_fraction_leaf=0.0, max_features='auto', 
                               max_leaf_nodes=None, min_impurity_decrease=0.0, 
                               min_impurity_split=None, bootstrap=True, oob_score=False, 
                               n_jobs=None, random_state=None, verbose=0, warm_start=False)
RFreg.fit(train_x, train_y)

# Make predictions using the testing set
Gross_pred = RFreg.predict(valid_x)

# Plot outputs
plt.figure()
RFregPlot = plt.scatter(valid_y*df2_filt[v:,2], Gross_pred*df2_filt[v:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Random Forrest Regression Hyper Tuned')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_RFreg = RFreg.score(train_x, train_y)
print('Train_Score:',Train_Score_RFreg)
Test_Score_RFreg = RFreg.score(test_x, test_y)
print('Test_Score:',Test_Score_RFreg)
Valid_Score_RFreg = RFreg.score(valid_x, valid_y)
print('Valid_Score:',Valid_Score_RFreg)
Correlation_Coefficient_RFreg = np.corrcoef(valid_y*df2_filt[v:,2], Gross_pred*df2_filt[v:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_RFreg)
plt.savefig('Random Forrest Regression Hyper Tuned.png')
plt.close()

# Nearest Neighbor Regression Hyperparameter Tuning
Bestneigh = 0
BestScore = 0
BestLeaf = 0
BestP = 0
for i in range(2,10,1):
    for j in range(30,70,10):
        for k in range(2,5,1):
            neighReg = KNeighborsRegressor(n_neighbors=i, weights='uniform', algorithm='auto', 
                               leaf_size=j, p=k, metric='minkowski', 
                               metric_params=None, n_jobs=None)
            neighReg.fit(train_x, train_y)

            # Make predictions using the testing set
            Gross_pred = neighReg.predict(test_x)
            Test_Score = neighReg.score(test_x, test_y)

            if Test_Score>BestScore:
                Bestneigh = i
                BestScore = Test_Score
                BestLeaf = j
                BestP = k
print(Bestneigh)
print(BestScore)
print(BestLeaf)
print(BestP)
neighReg = KNeighborsRegressor(n_neighbors=Bestneigh, weights='uniform', algorithm='auto', 
                               leaf_size=BestLeaf, p=BestP, metric='minkowski', 
                               metric_params=None, n_jobs=None)
neighReg.fit(train_x, train_y)

# Make predictions using the testing set
Gross_pred = neighReg.predict(valid_x)

# Plot outputs
plt.figure()
neighRegPlot = plt.scatter(valid_y*df2_filt[v:,2], Gross_pred*df2_filt[v:,2], color='blue', linewidth=1)
plt.plot([0, ymax], [0, ymax], color = 'red', linewidth = 2)
plt.title('Nearest Neighbor Regression Hyper Tuned')
plt.xlabel('Measured Gross (bbl/Day)')
plt.ylabel('Predicted Gross (bbl/Day)')
plt.xlim((0, ymax))
plt.ylim((0, ymax))
Train_Score_neighReg = neighReg.score(train_x, train_y)
print('Train_Score:',Train_Score_neighReg)
Test_Score_neighReg = neighReg.score(test_x, test_y)
print('Test_Score:',Test_Score_neighReg)
Valid_Score_neighReg = neighReg.score(valid_x, valid_y)
print('Valid_Score:',Valid_Score_neighReg)
Correlation_Coefficient_neighReg = np.corrcoef(valid_y*df2_filt[v:,2], Gross_pred*df2_filt[v:,2])[1,0]
print('Correlation_Coefficient:',Correlation_Coefficient_neighReg)
plt.savefig('Nearest Neighbor Regression Hyper Tuned.png')
plt.close()
