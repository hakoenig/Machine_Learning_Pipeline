"""
Module for evaluation and comparision of models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_f1_score(df, precision_col, recall_col, target_col):
    """
    Adds column for f1 score to df.
    In:
        - df: pandas df
        - precision_col: name of column with precision
        - recall_col: name of column with recall
        - target_col: name of new f1 column
    """
    df[target_col] = 2 * ((df[precision_col] * df[recall_col])
                             / (df[precision_col] + df[recall_col]))


def print_mean_max(df, groupby_column, column, time_in_min = True):
    """
    Prints mean and max of groupby_column.
    In:
        - df: pandas df
        - groupby_column: column to group by
        - column: column in pandas df
        - time_in_min: divide column by 60 to get min
    """
    if time_in_min:
        print("Mean values for {} by {} is:".format(column, groupby_column))
        print(df.groupby([groupby_column]).mean()[column].apply(lambda x: x / 60).sort_values(ascending=False))
        print("\nMax values for {} by {} is:".format(column, groupby_column))
        print(df.groupby([groupby_column]).max()[column].apply(lambda x: x / 60).sort_values(ascending=False))
    else:
        print("Mean values for {} by {} is:".format(column, groupby_column))
        print(df.groupby([groupby_column]).mean()[column].sort_values(ascending=False))
        print("\nMax values for {} by {} is:".format(column, groupby_column))
        print(df.groupby([groupby_column]).max()[column].sort_values(ascending=False))

def get_and_plot_feature_importance_top_n_features(n, clf, data):
    """
    Prints list of top n import features from tree based cls and shows plot.
    In:
        - n: (int) how many top features?
        - clf: trained classifier
        - data: data that was cls was trained on. Used to get names of features.
    """
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(n):
        print("%d. feature %s (%f)" % (f + 1, data.columns[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    _ = plt.figure()
    _ = plt.title("Feature importances")
    _ = plt.bar(range(data.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    _ = plt.xticks(range(data.shape[1]), data.columns[indices], rotation='vertical')
    _ = plt.xlim([-1, 15])
    plt.show()

def report(results, n_top=3):
    """
    Source:
    http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#sphx-glr-auto-examples-model-selection-randomized-search-py
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
