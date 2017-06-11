"""
Module for preprocessing functions
"""
import re
import numpy as np
import pandas as pd

def keep_track_of_missing_values(df, missing_ind):
    """
    Adds columns indicating if original columns had
    missing values.
    In:
        - df: pandas df
        - missing_ind: (str) indicates columns that keep track of
            missing data
    Out:
        - df: function modifies df inplace
        - list of newly created columns
    """
    new_columns = []

    for column in df:

        if df[column].isnull().values.any():
            new_col_name = column + missing_ind
            new_col = np.zeros(len(df), dtype=np.int)
            new_col[df[column].isnull()] = 1
            df[new_col_name] = new_col
            new_columns.append(new_col_name)
    return new_columns


def append_column_group_for_list_of_columns(l, column_group_dict, end_of_columngroup_str = "_YY_"):
    """
    For each column in l, prepends column group.
    In:
        - l: (list) of columns
        - column_group_dict: (dict) with 'column group': [col1, col2]
        - end_of_columngroup_str: (str) end of columngroup in col name
    Out:
        - new_list
    """
    reversed_column_group_dict = {entry: key for key, val in column_group_dict.items() for entry in val}
    new_list = []
    for col in l:
        if type(col) == tuple:
            # in this case, we support the input of list of tuples as well.
            # helpful for input like: [(name of binary column, value to replace missing values with)]
            col_group = reversed_column_group_dict[col[0]]
            new_list.append((col_group + end_of_columngroup_str + col[0],col[1]))
        else:
            col_group = reversed_column_group_dict[col]
            new_list.append(col_group + end_of_columngroup_str + col)
    return new_list


def rename_columns(df, column_group_dict, end_of_columngroup_str = "_YY_", inplace=True):
    """
    Renames columns in df in place by adding
    key in front of column name. Example:
    column_group_dict = {'group1': ['col1', 'col2']}
    would lead to column names: 'group1_col1', 'group1_col2'
    In:
        - df: pandas df
        - column_group_dict: (dict) with group names as keys
                             and list of associated columns as value
        - end_of_columngroup_str: (str) end of columngroup in col name
        - inplace: (bool) whether or not to replace colnames inplace
    """
    new_col_names_dict = {} # we will use this to update col names
    for column_group, col_list in column_group_dict.items():
        for col in col_list:
            new_col_names_dict[col] = column_group + end_of_columngroup_str + col

    if inplace:
        df.rename(columns=new_col_names_dict, inplace=inplace)
    else:
        return df.rename(columns=new_col_names_dict, inplace=inplace)


def get_dict_of_updated_column_groups(df, old_dict, list_of_appended_str, end_of_columngroup_str = "_YY_"):
    """
    Creates an updated dict of column groups. Due to feature engineering, column groups
    might have expanded. This function gets an updated dict.
    In:
        - df: pandas df
        - old_dict: (dict) of old column groups
        - list_of_appended_str: (list) of str that got appended to newly created features,
                                e.g. '_missing', or '_dummy'
        - end_of_columngroup_str: (str) end of columngroup in col name
    """
    new_dict = {}
    list_of_current_col = df.columns

    for column_group, col_list in old_dict.items():
        new_dict[column_group] = []

        for col in col_list:
            colgroup_col = column_group + end_of_columngroup_str + col

            for current_col in list_of_current_col:
                if current_col.startswith(colgroup_col):
                    if current_col not in new_dict[column_group]:
                            new_dict[column_group].append(current_col)
    return new_dict


def fill_missing_categorical_values(df, list_of_categorical_vars = [], df_to_get_mode_from = None):
    """
    Fills missing categorical values. Missing categorical vars in
    list_of_categorical_vars are replaced by most often occuring category.
    In:
        - df: pandas df
        - list_of_categorical_vars: (list) of categorical variables
        - df_to_get_mode_from: pandas df; if passed; using that df's to get mode for categorical
                                column
    Out:
        - df
    """
    for cat_var in list_of_categorical_vars:
        if df_to_get_mode_from is not None:
            df[cat_var] = df[cat_var].fillna(df_to_get_mode_from[cat_var].mode()[0])
        else:
            df[cat_var] = df[cat_var].fillna(df[cat_var].mode()[0])
    return df


def fill_missing_values(df, df_to_get_mean_from = None):
    """
    Replaces missing values with mean.
    In:
        - df: pandas df
        - df_to_get_mean_from: pandas df; if passed; using that df's to get
                                mean of column
    Out:
        - df
    """
    for col in df.columns[df.isnull().any()]:
        if df_to_get_mean_from is not None:
            try:
                df[col] = df[col].fillna(df_to_get_mean_from[col].mean())
            except:
                print("Warning: Tried to get mean of col: ", col)
        else:
            try:
                df[col] = df[col].fillna(df[col].mean())
            except:
                print("Warning: Tried to get mean of col: ", col)
    return df


def in_bound_test(value, start, end):
    """
    Helper function to recreate discretized dummy
    vars in test set.
    In:
        - value:
        - start:
        - end:
    Out:
        - 1 if value in range; 0 o/w
    """
    if value >= start and value <= end:
        return 1
    else:
        return 0


def insert_discretize_quantiles(df, col_to_value_dict, drop_original=False):
    """
    In:
        - df: pandas dataframe
        - col_to_value_dict: (dict) with columns (keys) to be
            created in df as dummy vars according to values
            {original_col: [(dummy_col, start, end),
                            (dummy_col, start, end)]}
        - drop_original: (bool) whether or not to drop original column
                        that discrete dummies are generated from
    Out:
        - df
    """
    for original_col in col_to_value_dict:

        list_of_dum_cols = col_to_value_dict[original_col]

        for list_of_dum_col in list_of_dum_cols:
            dummy_col, start, end = list_of_dum_col
            df[dummy_col] = df.apply(lambda row: in_bound_test(row[original_col], start, end), axis=1)

        if drop_original:
            del df[original_col]

    return df


def build_col_to_value_dict(df, dummy_code):
    """
    Function that builds dict with discretized columns
    and their dummy columns with cut-off values.
    In:
        - df: pandas dataframe
        - dummy_code: (str) to append to dummy columns
    Out:
        - dict
    """
    col_to_value_dict = {}

    for col in df.columns:
        if dummy_code in col:

            start_pos = col.find(dummy_code + "_")
            original_col = (col[:start_pos])
            if not original_col in col_to_value_dict:
                col_to_value_dict[original_col] = []

            start_pos = col.find(dummy_code + "_")
            cut_off_vals = col[start_pos + len(dummy_code + "_"):]

            start = cut_off_vals[1: cut_off_vals.find(",")]
            if cut_off_vals[0] == "[":
                start = float(start)
            else:
                start = float(start) + 1

            end = cut_off_vals[cut_off_vals.find(",") + 1:]
            numbers = re.findall('[\d\.]+', end)
            if numbers:
                end = float(numbers[0])
            else:
                raise ValueError('Could not find number value for end in cut_off_vals.')
            if cut_off_vals[:-1] == ")":
                end -= 1

            col_to_value_dict[original_col].append((col, start, end))

    return col_to_value_dict


def create_missing_value_colum_in_testset(traindf, testdf, missing_ind):
    """
    Creates same columns indicating missing values as we
    have in training set.
    In:
        - traindf: pandas dataframe with training data
        - testdf: pandas dataframe with test data
        - missing_ind: (str) indicates columns that keep track of
            missing data
    Out:
        - df
    """
    for col in traindf.columns:

        if col[-len(missing_ind):] == missing_ind:

            original_col = col[:col.find(missing_ind)]

            new_col = np.zeros(len(testdf), dtype=np.int)
            new_col[testdf[original_col].isnull()] = 1
            testdf[col] = new_col


def discretize_cont_var(df, cont_var_list, n, dummy_code, drop=False):
    """
    Discretizes  continuous variable.
    In:
        - df: pandas dataframe
        - cont_var_list: list of continues variable to be discretized
        - n: number of percentiles
        - dummy_code: (str) to append to dummy columns
        - drop: (bool) to drop continous variable
    Out:
        - df
    """
    if type(cont_var_list) == list:

        for cont_var in cont_var_list:
            step_size = 1/n
            bucket_array = np.arange(0, 1+step_size, step_size)

            df[cont_var + dummy_code] = pd.qcut(df[cont_var], bucket_array)
            df = pd.get_dummies(df, columns=[cont_var + dummy_code])

            if drop:
                del df[cont_var]
    else:
        step_size = 1/n
        bucket_array = np.arange(0, 1+step_size, step_size)

        df[cont_var + dummy_code] = pd.qcut(df[cont_var_list], bucket_array)
        df = pd.get_dummies(df, columns=[cont_var_list + dummy_code])

        if drop:
            del df[cont_var_list]

    return df


def dummify_var(df, cat_vars, drop=False):
    """
    Takes categorical variable and creates binary/dummy variables from it.
    In:
        - df: pandas dataframe
        - cat_vars: list of categorical variables
        - drop: (bool) whether or not to drop first dummy
    Out:
        - df: pandas dataframe
    """
    return pd.get_dummies(df, columns=cat_vars, drop_first=drop)
