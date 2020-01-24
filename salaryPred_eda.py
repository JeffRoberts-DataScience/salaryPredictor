'''This script loads, explores, and visualizes the salary prediction datasets'''

__author__ = 'Jeff Roberts'
__email__ = 'jeffroberts.ds@gmail.com'
__website__ = 'www.portoexpress.com'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in files and print the information
train_feature_df = pd.read_csv('data/train_features.csv')
train_target_df = pd.read_csv('data/train_salaries.csv')
test_feature_df = pd.read_csv('data/test_features.csv')

#Identify numeric and categorical columns
numeric_cols = ['yearsExperience', 'milesFromMetropolis']
categorical_cols = ['jobId', 'companyId', 'jobType', 'degree', 'major', 'industry']

#Merge Job Description table with Job Salary table into Training set
train_df = pd.merge(train_feature_df, train_target_df, on='jobId')

#View plotted data to check for outliers or NaN data
plt.figure(figsize = (14, 6))
plt.subplot(1,2,1)
sns.boxplot(train_df.salary)
plt.subplot(1,2,2)
sns.distplot(train_df.salary, bins=20)
plt.show()

#High JUNIOR salaries are possible in the OIL and FINANACE industries.
#However need to remove the jobIds where the salary is 0.0
train_df = train_df[train_df.salary > 8.5]


# Define a function to show a visual representation of each feature information
def plot_feature(df, col):
    '''
    Make plot for each features
    left, the distribution of samples on the feature
    right, the dependance of salary on the feature
    '''
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    if df[col].dtype == 'int64':
        df[col].value_counts().sort_index().plot()
    else:
        # change the categorical variable to category type and order their level by the mean salary
        # in each category
        mean = df.groupby(col)['salary'].mean()
        df[col] = df[col].astype('category')
        levels = mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels, inplace=True)
        df[col].value_counts().plot()
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Counts')
    plt.subplot(1, 2, 2)

    if df[col].dtype == 'int64' or col == 'companyId':
        # plot the mean salary for each category and fill between the (mean - std, mean + std)
        mean = df.groupby(col)['salary'].mean()
        std = df.groupby(col)['salary'].std()
        mean.plot()
        plt.fill_between(range(len(std.index)), mean.values - std.values, mean.values + std.values, \
                         alpha=0.1)
    else:
        sns.boxplot(x=col, y='salary', data=df)

    plt.xticks(rotation=45)
    plt.ylabel('Salaries')
    plt.show()


plot_feature(train_df, 'companyId')
plot_feature(train_df, 'jobType')
plot_feature(train_df, 'degree')
plot_feature(train_df, 'major')
plot_feature(train_df, 'industry')
plot_feature(train_df, 'yearsExperience')
plot_feature(train_df, 'milesFromMetropolis')

#Encode each categorical feature into a salary mean number for graphing
def encode_label(df, col):
    #encode the categories using average salary for each category to replace label
    cat_dict ={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = train_df[train_df[col] == cat]['salary'].mean()
    df[col] = df[col].map(cat_dict)
    df[col] = df[col].astype('float64')


#Place numerical equivalents in place of each categorical value
for col in train_df.columns:
    if train_df[col].dtype.name == "category":
        encode_label(train_df, col)


#Create heat map graph of feature means and numerical means
fig = plt.figure(figsize=(12, 10))
features = ['companyId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']
sns.heatmap(train_df[features + ['salary']].corr(), cmap='Blues', annot=True)
plt.xticks(rotation=45)
plt.show()

