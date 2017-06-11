from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from util.Pipeline.baseline import Zero_Predictor, SAPSIIClassifier
from util.Pipeline.baseline import Average_Predictor
from util.Pipeline.baseline import SAPSIICalculatedClassifier, SAPSIIFittedClassifier, SuperLearnerClassifier


def define_clfs_params(grid_size):
    """
    Defines classifiers and param grid that are used
    to find best model.
    Adjusted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    """

    clfs = {
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'NN': MLPClassifier(),
        'SAPS': SAPSIIClassifier(),
        'SC': SAPSIICalculatedClassifier(),
        'SF': SAPSIIFittedClassifier(penalty='l1', C=1e5),
        'SL': SuperLearnerClassifier(),
        'AVG': Average_Predictor(),
        'ZERO': Zero_Predictor(),
        }

    large_grid = {
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 1,5,10,20,50,100], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,5,10]},
    'RF': {'n_estimators': [1,10,100,1000,10000], 'max_depth': [None, 1,5,10,20,50,100], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,5,10]},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'SVM': {'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10], 'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate': [0.001,0.01,0.05,0.1,0.5],'subsample': [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB': {},
    'KNN': {'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'NN': {'activation': ["tanh", "relu"], 'hidden_layer_sizes':[(100, ), (100,50), (100,50,10), (10,50,50), (20,10,10), (50,25,10,5)],
            'solver': ["lbfgs", "sgd"], "alpha": [0.0001, 0.1], 'learning_rate': ["constant", "adaptive"], 'learning_rate_init': [0.001, 0.01, 0.1]},
    'SAPS': {},
    'SC': {},
    'SF': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SL': {},
    'AVG': {},
    'ZERO': {},
    }

    small_grid = {
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'DT': {'criterion': ['gini'], 'max_depth': [20,50,100], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,5,10]},
    'RF': {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, 50], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,10]},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'SVM' :{'C' :[0.0001,0.001,0.01,0.1,1,10], 'kernel':['linear', 'poly']},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5], 'subsample': [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB': {},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100], 'weights': ['uniform','distance'], 'algorithm': ['auto','ball_tree','kd_tree']},
    'NN': {'activation': ["relu"], 'hidden_layer_sizes':[(50,), (100,50,10), (20,10,10)],
            'solver': ["sgd"], "alpha": [0.0001, 0.1],'learning_rate': ["adaptive"], 'learning_rate_init': [0.001, 0.01]},
    'SAPS': {},
    'SC': {},
    'SF': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'SL': {},
    'AVG': {},
    'ZERO':{},
    }

    test_grid = {
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'], 'min_samples_split': [10]},
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'], 'min_samples_split': [10]},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'SVM' :{'C' :[0.01], 'kernel':['linear']},
    'GB': {'n_estimators': [1], 'learning_rate': [0.1], 'subsample': [0.5], 'max_depth': [1]},
    'NB': {},
    'KNN' :{'n_neighbors': [5], 'weights': ['uniform'], 'algorithm': ['auto']},
    'NN': {'activation': ["relu"], 'hidden_layer_sizes':[(10, ), (50,10)],
            'solver': ["sgd"], "alpha": [0.1], 'learning_rate': ["constant"], 'learning_rate_init': [0.01]},
    'SAPS': {},
    'SC': {},
    'SF': { 'penalty': ['l1'], 'C': [0.01]},
    'SL': {},
    'AVG':{},
    'ZERO':{},
    }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0
