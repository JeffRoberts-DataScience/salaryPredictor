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

print(train_feature_df.head(10))
