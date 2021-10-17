# DATA ACQUISION

# import pandas library
import pandas as pd
import numpy as np
other_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df = pd.read_csv(other_path, header=None)

# ADD HEADER
# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
# replace ? with NaN
df1=df.replace('?',np.NaN)
# drop missing value from price column
df=df1.dropna(subset=["price"], axis=0)
print(df.head(5))

# SAVE DATA SET
df.to_csv("automobile.csv", index=False)
print(df.dtypes)
df.describe()
df.describe(include='all')