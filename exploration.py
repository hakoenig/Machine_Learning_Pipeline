"""
Module for exploration functions.
"""

import psycopg2
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split

plt.style.use('ggplot')

def plot_sorted_continuous_var(df, col, ylabel, title):
    """
    Plots sorted continous var.
    In:
        - df: pandas df
        - col: column to be plotted
        - ylabel: label of y axis
        - title: title of plot
    Adjusted from Kaggle user sudalairajkumar @
    https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-mercedes
    """
    plt.figure(figsize=(8,6))
    plt.scatter(range(df.shape[0]), np.sort(df[col].values))
    plt.xlabel('Index', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim(-50, len(df) + 500)
    plt.title(title)
    plt.show()

def distribution_plot(df, col, title):
    """
    Plots sorted continous var.
    In:
        - df: pandas df
        - col: column to be plotted
        - title: title of plot
    Adjusted from Kaggle user sudalairajkumar @
    https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-mercedes
    """
    plt.figure(figsize=(12,8))
    sns.distplot(df[col].values, bins=50, kde=False)
    plt.xlabel('column value', fontsize=12)
    plt.title(title)
    plt.show()

def unique_values_per_column(df, exclude_cols_list):
    """
    In:
        - df: pandas df
        - exclude_cols_list: list of columns that shall not be considered.

    Adjusted from Kaggle user sudalairajkumar @
    https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-mercedes
    """
    unique_values_dict = {}
    for col in df.columns:
        if col not in exclude_cols_list:
            unique_value = str(np.sort(df[col].unique()).tolist())
            tlist = unique_values_dict.get(unique_value, [])
            tlist.append(col)
            unique_values_dict[unique_value] = tlist[:]
    for unique_val, columns in unique_values_dict.items():
        print("Columns containing the unique values: ", unique_val)
        print(columns)
        print("--------------------------------------------------")

def distribution_of_target_by_categorical_columns(df, target_var, categorical_columns):
    """
    In:
        - df: pandas df
        - target_var: name of target variable
        - categorical_columns: list of categorical columns
    """
    for var_name in categorical_columns:
        col_order = np.sort(df[var_name].unique()).tolist()
        _ = plt.figure(figsize=(12,6))
        if len(col_order) < 20:
            _ = sns.violinplot(x=var_name, y=target_var, data=df, order=col_order)
        else:
            _ = sns.boxplot(x=var_name, y=target_var, data=df, order=col_order)
        _ = plt.xlabel(var_name, fontsize=12)
        _ = plt.ylabel(target_var, fontsize=12)
        _ = plt.title("Distribution of target variable by " + var_name, fontsize=15)
        plt.show()

def create_graph_for_decision_tree(X, y, max_depth):
    """
    Prints decision tree trained on X and y with max_depth max_depth.
    In:
        - X: training data
        - y: targets
        - max_depth: max_depth of tree
    """

    tree_clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    tree_clf = tree_clf.fit(X, y)

    dot_data = tree.export_graphviz(tree_clf, out_file=None,
                             feature_names=X.columns,
                             filled=True, rounded=True,
                             special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return graph


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

def read_data_from_db(query, conn_string, index_col=None, split=False, target=None, test_size=None):
    """
    Opens a connection with DB and runs query.
    In:
        - query: (str) SQL query
        - conn_string: connection string to db; connection gets opened and closed by query
        - index_col: index col for pd DF
        - split: (bool) split data into train & test?
        - target: column name of target
        - test_size: % of test data
    Out:
        - data: pandas df
    """
    conn = psycopg2.connect(conn_string)

    data = pd.read_sql_query(query, conn, index_col=index_col)
    if split:
        X_train, X_test, y_train, y_test = train_test_split(data.drop([target], axis=1),
                                                            data[target], test_size=test_size,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    conn.close()
    return data

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
