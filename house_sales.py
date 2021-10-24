import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)
print(df.head())

# QUESTION 1
print(df.dtypes)

# QUESTION 2
df.drop('Unnamed: 0', axis = 1, inplace=True)
df.drop('id', axis = 1, inplace=True)
print(df.describe())

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

# QUESTION 3
unique_floor = df['floors'].value_counts().to_frame()
print(unique_floor)

# QUESTION 4
import seaborn as sns
sns.boxplot(x='waterfront', y='price', data=df)
plt.ylim(0,)
plt.show()

# QUESTION 5
sns.regplot(x="sqft_above", y= "price", data=df)
plt.ylim(0,)
plt.show()
df.corr()['price'].sort_values()

# QUESTION 6
X = df[['sqft_living']]
Y = df[['price']]
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X,Y)

# QUESTION 7
features = df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
X = features
Y = df[['price']]
lm1 = LinearRegression()
lm1.fit(features, df['price'])
lm1.score(features, Y)

# QUESTION 8
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
from sklearn.pipeline import Pipeline

y = df['price']
pipe = Pipeline(Input)
Z = features.astype(float)
pipe.fit(Z,y)
ypipe = pipe.predict(Z)
ypipe[0:4]

# QUESTION 9
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

from sklearn.linear_model import Ridge
lre=LinearRegression
lre.fit(x_train[features], y_train)
lre.score(x_test[features],y_test)

# QUESTION 10
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[features])
x_test_pr=pr.fit_transform(x_test[features])

## fit model
RigeModel=Ridge(alpha=1)
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)