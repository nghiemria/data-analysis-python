import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns

# LINEAR REGRESSION AND MULTIPLE LINEAR REGRESSION
lm = LinearRegression()
## train the model using highway-mpg and price
X = df[['highway-mpg']] # independent variable
Y = df['price'] # dependent variable
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]
print(lm.intercept_)
print(lm.coef_)

## train the model using engine-size and price
lm1 = LinearRegression()
U = df[["engine-size"]]
V = df[["price"]]
lm1.fit(U,V)
Yhat = lm1.predict(X)
print(lm1.intercept_)
print(lm1.coef_)

## equation
Yhat=-7963.34 + 166.86*X

## train model using horsepower, curb-weight, engine-size, highway-mpg and price
lm2 = LinearRegression()
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm2.fit(Z, df['price'])
print(lm2.intercept_)
print(lm2.coef_)

# MODEL EVALUATION USING VISUALISATION
## regression plot of highway-mpg as a potential predictor for price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/model_development/regression_highwaympg.png")

## regression plot of peak-rpm as a potential predictor for price
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/model_development/regression_peakrpm.png")

print(df[["peak-rpm","highway-mpg","price"]].corr())

# RESIDUAL PLOT
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/model_development/residual.png")

## make a prediction
Y_hat = lm2.predict(Z)
plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/model_development/actual vs fitted for price.png")

# POLYNOMIAL REGRESSION AND PIPELINES
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

x = df['highway-mpg']
y = df['price']
# Here we use a polynomial of the 11th order
f = np.polyfit(x, y, 11)
p = np.poly1d(f)
print(p)
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 11)
pr=PolynomialFeatures(degree=2)
Z_pr=pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape)

# PIPELINE
## creating a list of tuples including the name of the model or estimator and its corresponding constructor
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
## input the list as an argument to the pipeline constructor
pipe=Pipeline(Input)

## convert data type Z to float and then normalise the data
Z = Z.astype(float)
pipe.fit(Z,y)

## normalise the data, perform a transform and produce a prediction simultaneously
ypipe=pipe.predict(Z)
print(ypipe[0:4])

# MEASURES FOR IN-SAMPLE EVALUATION
## model 1: simple linear regression
lm.fit(X, Y)
## Find the R^2
print('The R-square is: ', lm.score(X, Y))
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
## compare the predicted and actual result
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

## model 2: multiple linear regression
## fit the model
lm.fit(Z, df['price'])
## Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))
## produce a prediction
Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

## model 3: polynomial fit
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
mean_squared_error(df['price'], p(x))

# PREDICTION AND DECISION MAKING
new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
yhat=lm.predict(new_input)
yhat[0:5]
plt.plot(new_input, yhat)
plt.show()

matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/model_development/decision making.png")

#done