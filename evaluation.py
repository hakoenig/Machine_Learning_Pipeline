"""
Module for evaluation and comparision of models.
"""

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
