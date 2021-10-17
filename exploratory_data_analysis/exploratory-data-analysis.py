import pandas as pd
import numpy as np
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
import matplotlib.pyplot as plt
import matplotlib.pylab
import seaborn as sns
from scipy import stats

# FIND CORRELATION
df[['bore','stroke','compression-ratio','horsepower']].corr()

## find correlation of engine-size and price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/exploratory_data_analysis/engine_size.png")
print(df[["engine-size", "price"]].corr())

## find correlation of highway-mpg and price
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/exploratory_data_analysis/highway_mpg.png")
print(df[['highway-mpg', 'price']].corr())

## find correlation of peak-rpm and price
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/exploratory_data_analysis/peakrpm.png")
print(df[['peak-rpm','price']].corr())

## find correlation of stroke and price
sns.regplot(x="stroke", y="price", data=df)
plt.ylim(0,)
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/exploratory_data_analysis/stroke.png")
print(df[['stroke','price']].corr())

# CATEGORICAL VARIABLE
sns.boxplot(x="body-style", y="price", data=df)
plt.ylim(0,)
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/exploratory_data_analysis/bloxplot_bodystyle.png")

sns.boxplot(x="engine-location", y="price", data=df)
plt.ylim(0,)
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/exploratory_data_analysis/bloxplot_engine_location.png")

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.ylim(0,)
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/exploratory_data_analysis/bloxplot_drive_wheel.png")

# DESCRIPTIVE STATISTICAL ANALYSIS
#print(df.describe(include=['object']))

## value counts
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

## engine-location
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)

# BASICS OF GROUPING
df['drive-wheels'].unique()
df_group_one = df[['drive-wheels','body-style','price']]
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
#print(df_group_one)

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
#print(grouped_pivot)

# variables: drive wheels and body style vs. price
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
print(plt.show())
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/exploratory_data_analysis/variables vs price.png")

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
print(plt.show())
matplotlib.pylab.savefig("D:/3 - Programming/data-analysis-python/exploratory_data_analysis/grouped_pivot.png")

# CORRELATION AND CAUSATION
## wheel-base vs. price
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# ANOVA: Analysis of Variance
## to see if different types of drive-wheel impact prive
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

grouped_test2.get_group('4wd')['price']
## ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
                              grouped_test2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

## fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)

## 4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)

## 4wd and fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)