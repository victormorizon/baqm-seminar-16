import pandas as pd
import numpy as np
import lightgbm as lgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, average_precision_score, make_scorer, mean_absolute_error, median_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import doubleml as dml

def split_data(df, num_splits):

    df_nothing = df[df["welcome_discount"] == 0].copy()
    df_others = df[df['welcome_discount'] != 0].copy()
    
    df_others['split'] = pd.qcut(df_others['welcome_discount'], q=num_splits)
    split_names = df_others['split'].unique()

    split_dfs = {label: pd.concat([df_nothing, df_others[df_others['split'] == label]]).drop("split", axis=1) for label in split_names}
    
    return dict(sorted(split_dfs.items()))

def data_setup(df, num_splits, log):

    # Keep only after 2021
    df = df[df["first_data_year"] >= 2021]

    # Check if the data contains any null columns
    nulls = [col for col, val in df.isnull().any().to_dict().items() if val == True]
    assert len(nulls) == 0, f"There are null columns in the data!! {nulls}"

    # Drop columns that won't be used ever (apart from EDA)
    cols_to_drop = ["policy_nr_hashed", "last_data_year", "first_data_year", "control_group", 'count', 'first_datapoint_year', 'last_datapoint_year']
    df = df[[col for col in df.columns.to_list() if (col not in cols_to_drop)]]

    # Set data type and create dummies
    categorical_features = []
    continuous_features = []
    binary_features = []

    # Define a threshold for the maximum number of unique values for a categorical column
    max_unique_values_for_categorical = 5

    # Iterate through each column to determine if it's categorical, continuous, or binary
    for column in df.columns:
        unique_values = df[column].nunique()
        if unique_values == 2:
            # If exactly 2 unique values, treat column as binary
            binary_features.append(column)
        elif (df[column].dtype == 'object' or unique_values <= max_unique_values_for_categorical) and unique_values > 2:
            # If object type or up to the threshold of unique values (and more than 2), treat as categorical
            categorical_features.append(column)
        else:
            # Otherwise, treat as continuous
            continuous_features.append(column)

    categorical_features = [col for col in categorical_features if col != "nr_years"]
    continuous_features = continuous_features + ['nr_years']

    if log:
        print(f'Binary Features: {binary_features}')
        print(f'Categorical Features: {categorical_features}')
        print(f'Continuous Features: {continuous_features}')

    # Create dummies
    df = pd.get_dummies(df, columns=categorical_features, dtype="int")

    # Get data splits
    splits = split_data(df, num_splits)

    return splits

def mae_prob(y_true, y_pred_probs):
    return mean_absolute_error(y_true, y_pred_probs)

def medae_prob(y_true, y_pred_probs):
    return median_absolute_error(y_true, y_pred_probs)

mae_prob_scorer = make_scorer(mae_prob, needs_proba=True)
medae_prob_scorer = make_scorer(medae_prob, needs_proba=True)

def run_first_stage_general(df, cols_to_drop_manual, target, iters, log, score):
    # Data prep
    cols_to_drop = ["churn", "policy_nr_hashed", "last_data_year", "first_data_year", "control_group", 'welcome_discount', 'count', 'first_datapoint_year', 'last_datapoint_year'] + cols_to_drop_manual
    selected_columns = [col for col in df.columns if not any(col.startswith(prefix) for prefix in cols_to_drop)]

    X = df[selected_columns]
    y = df[target]

    # Run model selection
    space = {
        'max_depth': hp.uniformint('max_depth', 50, 100),
        'n_estimators': hp.uniformint('n_estimators', 50, 200),
        'num_leaves': hp.uniformint('num_leaves', 2, 200),
        'min_child_samples': hp.uniformint('min_child_samples', 7, 100),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.25, 1),
        'subsample': hp.uniform('subsample', 0.25, 1),
        'subsample_freq': hp.uniformint('subsample_freq', 1, 100),
        'reg_alpha': hp.uniform('reg_alpha', 0, 0.2),
        'reg_lambda': hp.uniform('reg_lambda', 0, 0.2),
        'min_split_gain': hp.uniform('min_split_gain', 0, 0.5),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
        'min_data_in_leaf': hp.uniformint('min_data_in_leaf', 1, 21),
    }

    def objective(params):
        clf = lgb.LGBMClassifier(
            objective='binary',
            force_row_wise=True,
            verbosity=-1,
            # is_unbalance=True,
            **params
        )
        score = cross_val_score(clf, X, y, cv=5, scoring="neg_brier_score").mean()
        return {'loss': -score, 'status': STATUS_OK}

    n_iter = iters
    trials = Trials()

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=n_iter, trials=trials)

    if log:
        print("Best Score is: ", -trials.best_trial['result']['loss'])
        print("Best Parameters: ", best)

    # Save best model
    best_params = {
        'max_depth': int(best['max_depth']),
        'n_estimators': int(best['n_estimators']),
        'num_leaves': int(best['num_leaves']),
        'min_child_samples': int(best['min_child_samples']),
        'colsample_bytree': best['colsample_bytree'],
        'subsample': best['subsample'],
        'subsample_freq': int(best['subsample_freq']),
        'reg_alpha': best['reg_alpha'],
        'reg_lambda': best['reg_lambda'],
        'min_split_gain': best['min_split_gain'],
        'learning_rate': best['learning_rate'],
        'min_data_in_leaf': int(best['min_data_in_leaf']),
    }

    lgbm_best = lgb.LGBMClassifier(
        objective='binary',
        force_row_wise=True,
        verbosity=-1,
        # is_unbalance=True,
        **best_params
    )

    # Scores
    scores = {}

    if score:
        scores['score_brier'] = -np.mean(cross_val_score(lgbm_best, X, y, cv=5, scoring='neg_brier_score'))
        scores['score_brier_root'] = np.sqrt(scores['score_brier'])
        scores['score_log_loss'] = -np.mean(cross_val_score(lgbm_best, X, y, cv=5, scoring='neg_log_loss'))
        scores['score_mae'] = np.mean(cross_val_score(lgbm_best, X, y, cv=5, scoring=mae_prob_scorer))
        scores['score_medae'] = np.mean(cross_val_score(lgbm_best, X, y, cv=5, scoring=medae_prob_scorer))

    return lgbm_best, scores

def perform_single_doubleml(df, ml_y, ml_d):
    obj_dml_data = dml.DoubleMLData(df, 'churn', 'welcome_discount')
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_y, ml_d, score="ATE", weights=None)
    dml_irm_obj.fit()
    return dml_irm_obj

def global_run(df, splits, cols_to_drop_manual, iters, log, intermediary_scores):

    # Get data splits
    splits = data_setup(df, splits, log)

    # Run first stage
    first_stage_1 = {}
    first_stage_2 = {}
    double_mls = {}

    i = 1

    for k, v in splits.items():
        print(f"Running Split {i}...")
        v["welcome_discount"] = np.ceil(v["welcome_discount"]).astype(int)

        # Run first stages
        first_stage_1_temp = run_first_stage_general(v, cols_to_drop_manual, 'churn', iters, log, intermediary_scores)
        first_stage_2_temp = run_first_stage_general(v, cols_to_drop_manual, 'welcome_discount', iters, log, intermediary_scores)

        # Save everything
        first_stage_1[k] = first_stage_1_temp
        first_stage_2[k] = first_stage_2_temp
        double_mls[k] = perform_single_doubleml(v, first_stage_1_temp[0], first_stage_2_temp[0])
        print("Done!!")

        # Increase i
        i += 1

    return first_stage_1, first_stage_2, double_mls


if __name__ == "__main__":
    df = pd.read_csv("./data/prepped_data.csv", low_memory=False, index_col=0).drop_duplicates()
    first_stage_1, first_stage_2, double_mls = global_run(df, splits=1, cols_to_drop_manual=[], iters=2, log=False, intermediary_scores=False)

    for k, v in double_mls.items():
        print(k)
        print(v.summary)
        print(v.sensitivity_analysis().sensitivity_summary)