"""
Module for predictions.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import ParameterGrid
from collections import Counter
from util.Pipeline.grid import define_clfs_params
#import concurrent.futures
from joblib import Parallel, delayed, cpu_count
from copy import deepcopy
import pdb
import faulthandler
faulthandler.enable()

def run_clf_loop_with_rescaling_resampling_by_feature_group(grid_size, models_to_run, list_ks, list_thresholds,
                                                            train_data, val_data, y_train, y_val,
                                                            dict_of_rescalers, dict_of_rebalancers,
                                                            columns_to_be_scaled, updated_col_group_dict,
                                                            print_plots, file_name=None, verbose=False,
                                                            column_groups_to_drop_list=None,
                                                            parallel_jobs=1):
    """
    Runs clf loop with rescaling, resampling and selection of feature groups.
    Returns final df with results and time it took. Writes out final result df.
    df.

    In:
        - grid_size: (str) size of grid to run
        - models_to_run: (list) of models to run
        - list_ks: (list) of ints for performance at top # individual evaluation
        - list_thresholds: (list) of tresholds
        - train_data: pd df with train data; do not include target
        - val_data: pd df with val data; do not include target
        - y_train: pd df target for training
        - y_val: pd df target for validation
        - dict_of_rescalers: (dict) which scalers to use; 'scalername': scaler
        - dict_of_rebalancers: (dict) which rebalancer to use; 'rebalancername': rebalancer
        - columns_to_be_scaled: (list) of columns to be scaled
        - updated_col_group_dict: (dict) dict of updated column name to column group relationship
        - print_plots: (bool) whether or not to print plots
        - file_name: (str) name of file to write results to
        - verbose: (bool) additional print outs for progress
        - column_groups_to_drop_list: (list) of column groups that are supposed to be dropped.
                                        if no list is passed, all column_groups are dropped once.
        - parallel_jobs: Number of jobs to parallelize by classifier type
    Out:
        - (df with results, time_it_took)
    """
    if not column_groups_to_drop_list:
        column_groups_to_drop_list = [col_group for col_group in updated_col_group_dict]

    start_time = time.time()

    clfs, grid = define_clfs_params(grid_size)

    results_df = None

    #getting list of column groups so that we can keep track of which
    #columns groups we have already excluded
    list_col_groups = [col_group for col_group in updated_col_group_dict]

    for name_rescaler, rescaler in dict_of_rescalers.items():

        print("\n\nWorking with rescaler: ", name_rescaler)

        if name_rescaler == "None":
            rescaled_train_data = train_data
            rescaled_val_data = val_data
        else:
            # Setting up independent copies of train and val data. We want to keep
            # the unscaled dataframes for other rescales.
            rescaled_train_data = train_data.copy()
            rescaled_val_data = val_data.copy()

            # Important: we only fit the scaler on X_train_val and then use the "learned" transformation on
            # the other data sets.
            scaler = rescaler.fit(train_data[columns_to_be_scaled])
            rescaled_train_data.loc[:,columns_to_be_scaled] = scaler.transform(train_data[columns_to_be_scaled])
            rescaled_val_data.loc[:,columns_to_be_scaled] = scaler.transform(val_data[columns_to_be_scaled])

        for name_rebalancer, rebalancer in dict_of_rebalancers.items():

            print("\n\tWorking with rebalancer: ", name_rebalancer)

            # we also want to support working on unbalanced data.
            if name_rebalancer == "None":
                train_data_resampled, y_train_resampled = rescaled_train_data, y_train
            else:
                train_data_resampled, y_train_resampled = rebalancer.fit_sample(rescaled_train_data, y_train)
                train_data_resampled = pd.DataFrame(data=train_data_resampled,
                                                    columns=rescaled_train_data.columns.tolist())

            included_all_columns = False
            for col_group_to_be_dropped, col_group in updated_col_group_dict.items():

                if col_group_to_be_dropped in column_groups_to_drop_list:

                    list_col_groups_used = list(list_col_groups)
                    list_col_groups_used.remove(col_group_to_be_dropped)

                    print("\t\tDropping column group: ", col_group_to_be_dropped)
                    current_train = train_data_resampled.drop(col_group, axis=1)
                    current_val = rescaled_val_data.drop(col_group, axis=1)

                    # ensuring that val_data has same column order as train_data
                    current_val = current_val[current_train.columns.tolist()]

                    current_results = clf_loop(models_to_run, clfs, grid, current_train, current_val,
                                                           y_train_resampled, y_val, list_ks,
                                                           list_thresholds, print_plots, str(list_col_groups_used),
                                                           col_group_to_be_dropped, name_rescaler, name_rebalancer, verbose,
                                                           parallel_jobs=parallel_jobs)

                    if results_df is not None:
                        results_df = results_df.append(current_results, ignore_index=True)
                    else:
                        results_df = current_results

                # support including all columns as well
                if not included_all_columns:

                    print("\n\t\tIncluding all columns")
                    list_col_groups_used = list_col_groups
                    col_group_to_be_dropped = "None"
                    included_all_columns = True

                    current_train = train_data_resampled.drop([], axis=1)
                    current_val = rescaled_val_data.drop([], axis=1)

                    # ensuring that val_data has same column order as train_data
                    current_val = current_val[current_train.columns.tolist()]

                    current_results = clf_loop(models_to_run, clfs, grid, current_train, current_val,
                                                           y_train_resampled, y_val, list_ks,
                                                           list_thresholds, print_plots, str(list_col_groups_used),
                                                           col_group_to_be_dropped, name_rescaler, name_rebalancer, verbose,
                                                           parallel_jobs=parallel_jobs)

                    if results_df is not None:
                        results_df = results_df.append(current_results, ignore_index=True)
                    else:
                        results_df = current_results

    took_time = time.time() - start_time

    results_df.head()

    if not file_name:
        file_name = 'results_' + grid_size + '_grid.csv'
    results_df.to_csv(file_name, index=False)

    return(results_df, took_time)


def setup_results_df(list_ks):
    """
    Returns result df with precision and recall at k columns.
    In:
        - list_ks: (list) of ints to get precision and recall at.
    Out:
        - results_df
    """
    precision_at_k_lst = ["p_at_" + str(k) for k in list_ks]
    recall_at_k_lst = ["rec_at_" + str(k) for k in list_ks]
    columns=['model_type', 'clf', 'parameters',
             'shape_training_set', 'shape_test_set',
             'counter_train', 'counter_test',
             'normalizer', 'rebalancer',
             'col_groups_used', 'col_groups_excluded', 'features',
             'train_time', 'predict_time', 'auc-roc', 'threshold']
    columns = columns + precision_at_k_lst + recall_at_k_lst
    results_df = pd.DataFrame(columns=columns)
    return results_df


def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test,
             list_ks, list_thresholds, print_plots=False,
             col_groups_used="All", col_groups_excluded="None",
             normalizer="None", rebalancer="None", verbose=False,
             parallel_jobs=1):
    """
    Loops through classifiers and stores metrics in pandas df.
    Df gets returned.
    Adjusted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py

    In:
        - models_to_run: (list) of models to run
        - clfs: (dict) of classifiers
        - grid: (dict) of classifiers with set of parameters to train on
        - X_train: features from training set
        - X_test: features from test set
        - y_train: targets of training set
        - y_test: targets of test set
        - list_ks: list of k's to use for precision at k calculations
        - list_thresholds: list of thresholds to use for binary decision (1 or not)
        - print_plots: (bool) whether or not to print plots
        - col_groups_used: (str) which column groups were used; default: "All"
        - col_groups_excluded: (str) which column groups were excluded; default: "None"
        - normalizer: (str) which normalizer was used; default: "None"
        - rebalancer: (str) which rebalancer was used; default: "None"
        - verbose: (bool) additional print outs for progress
        - parallel_jobs: (int) number of concurrent worker processes, detected if < 1
    Out:
        - pandas df
    """
    results_df = setup_results_df(list_ks)
    shape_training_set = X_train.shape
    shape_test_set = X_test.shape

    joblib_verbose = 0
    if verbose:
        joblib_verbose = 5
    if parallel_jobs < 1:
        parallel_jobs = cpu_count()
    
    #get execution matrix
    execution_clf = []
    for model in models_to_run:
        execution_clf = execution_clf + [(clfs[model], params, model) for params in ParameterGrid(grid[model])]

    results = Parallel(n_jobs=parallel_jobs, verbose=joblib_verbose)(delayed(classifier_worker)\
        (deepcopy(conf[0]), X_train, y_train, X_test, y_test, conf[1], conf[2])\
        for conf in execution_clf)

    for y_pred_probs, train_time, predict_time, p, clf_name, clf_string in results:
        try:
            if verbose:
                total = str(predict_time + train_time)
                print('\t{} seconds with parameters: {}'.format(total, str(p)))

            row = [clf_name, clf_string, p, shape_training_set,
                    shape_test_set, str(Counter(y_train)),
                    str(Counter(y_test)), normalizer, rebalancer,
                    col_groups_used, col_groups_excluded,
                    str(list(X_train.columns.values)),
                    train_time, predict_time]

            append_results(results_df, row, list_ks, list_thresholds, y_pred_probs, y_test)
            if print_plots:
                _ = plot_precision_recall_n(y_test, y_pred_probs, clf_string)

        except IndexError as e:
            print('Error:', e)
            continue
        except e:
            print(e)
            continue

    return results_df

def classifier_worker(clf, X_train, y_train, X_test, y_test, p, clf_name):
    '''
    For internal use, function used by the parallel workers to train and validate a classifier.
    In:
        - clf: The classifier object
        - X_train: the training data
        - y_train: the training label
        - X_test: the testing data
        - y_test: the testing label
        - p: parameters for the classifier, from a ParameterGrid
    Out:
        -Tuple with the predicted probabilities, training time, testing time and the
        parameters passed
    '''
    #pdb.set_trace()
    clf.set_params(**p)

    start_time_training = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time_training

    start_time_predicting = time.time()
    y_pred_probs = clf.predict_proba(X_test)[:, 1]
    predict_time = time.time() - start_time_predicting
    return (y_pred_probs, train_time, predict_time, p, clf_name, str(clf))

def append_results(results_df, row, list_ks, list_thresholds, y_pred_probs, y_test):
    """
    Appends results to results_df.
    In:
        - results_df: pd dataframe for results
        - row: (list) data from previous calculations
        - list_ks: (list) of k's to use for precision at k calculations
        - list_thresholds: (list) of thresholds to use for binary decision (1 or not)
        - y_pred_probs: predicted probability for targets
        - y_test: targets
    Out:
        -
    """
    roc_score = roc_auc_score(y_test, y_pred_probs)
    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))

    for threshold in list_thresholds:
        precision_at_k_scores = []
        recall_at_k_scores = []

        for first_k in list_ks:
            precision, recall = calc_precision_recall(y_pred_probs_sorted, y_test_sorted, threshold, first_k)
            precision_at_k_scores.append(precision)
            recall_at_k_scores.append(recall)

        result_row = row + [roc_score, threshold] + precision_at_k_scores + recall_at_k_scores
        results_df.loc[len(results_df)] = result_row

def plot_precision_recall_n(y_true, y_prob, model_name):
    """
    Function to plot precision recall curve.
    Adjusted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    """
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_prob)

    for value in pr_thresholds:
        num_above_thresh = len(y_prob[y_prob >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    _ = plt.clf()
    fig, ax1 = plt.subplots()
    _ = ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    _ = ax1.set_xlabel('percent of population')
    _ = ax1.set_ylabel('precision', color='b')

    ax2 = ax1.twinx()
    _ = ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    _ = ax2.set_ylabel('recall', color='r')
    _ = ax1.set_ylim([0,1.05])
    _ = ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    _ = ax2.set_xlim([0,1])
    _ = ax2.set_ylim([0,1.05])
    _ = ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    _ = plt.title(model_name)
    _ = plt.show()


def calc_precision_recall(predicted_values, y, threshold, top_k):
    """
    Calculates precision and recall for given threshold.
    In:
        - predicted_values: numpy array of predicted scores
        - y: target values
        - threshold: threshold to use for calculation
        - top_k: int or "All" - how many first entries are we considering?
    Out:
        - (precision_score, recall_score)
    """
    y = np.asarray(y)
    x = [1 if predicted_value >= threshold else 0 for predicted_value in predicted_values]

    # for recall, we want to know the number of positives in the whole set.
    # independent of our top_k value
    total_true_positives = sum(y)

    # Initializing true positives, false positives, and false negatives to be 0.
    tp = fp = fn = 0

    # Depending on the input, we might have to change our top_k value
    if (top_k == "All") or (top_k == None) or (top_k > len(x)):
        top_k = len(x)

    # looping through the predictions and labels to update tp, fp, and fn.
    for i in range(top_k):
        if x[i] == 1 and y[i] == 1:
            tp += 1
        elif x[i] == 1:
            fp += 1
        elif x[i] == 0 and y[i] == 1:
            fn += 1

    # catch cases when denominator is 0
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if total_true_positives == 0:
        recall = 0
    else:
        recall = tp / total_true_positives

    return precision, recall
