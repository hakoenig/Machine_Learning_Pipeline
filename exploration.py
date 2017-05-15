"""
Module for exploration functions.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


plt.style.use('ggplot')


def plot_histograms(df, columns, min_max = False):
    """
    Plots histograms for columns in columns.
    In:
        - df: pandas df
        - columns: (list) of columns
        - min_max: (bool) whether or not to plot min and max values on same plot
    """
    for col in columns:

        if min_max:
            plt.figure();
            df[[col+"_min", col+"_max"]].plot.hist(alpha=0.5, title= col + ' histogram');
            plt.show();
        else:
            plt.figure();
            df[col].plot.hist(alpha=0.5, title= col + ' histogram');
            plt.show();


def read_data_from_csv(file_path, index_col=None, split=False, target=None, test_size=None):
    """
    Loads data from csv file.
    In:
        - file_path: path to csv file
        - split: (bool) split data into train & test?
        - target: column name of target
        - test_size: % of test data
    Out:
        - data: pandas Dataframe
    """
    data = pd.read_csv(file_path, index_col=index_col)

    if split:
        X_train, X_test, y_train, y_test = train_test_split(data.drop([target], axis=1),
                                                            data[target], test_size=test_size,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test
    return data


def plot_correlations(df, title):
    """
    In:
        - df: pandas dataframe
        - title: title for plot
    Out:
        -
    """
    ax = plt.axes();
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax);
    ax.set_title(title);
