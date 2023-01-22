#!/usr/bin/env python3

"""
Tabular Plaground Model S3 E3 
"""

# %% Imports 
import os

import catboost as cb
import lightgbm as gb
import matplotlib.pyplot as plt
import numpy as np
import optuna as op
import pandas as pd
import sklearn as sk
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


#%%  Read in the data
def read_data(file_path: str, data_path: str = "data") -> pd.DataFrame:
    """Imports the data into a pandas data frame

    Args:
        file_path (str): file path
        data_path (str, optional): _description_. Defaults to "data". Can be changed if running
        ini kaggle

    Returns:
        pd.DataFrame: pandas datafram containing the data
    """
    df = pd.read_csv(os.path.join(data_path, file_path))
    df = df.rename({x: x.replace(" ", "") for x in df.columns})

    return df


#%%
def submit(
    id_col: np.array, attrition_col: np.array, submission_name: str = "submission.csv"
) -> None:
    """Writes a competition submission

    Args:
        id (np.array): id column (numeric)
        attrition (np.array): attrition column (float)
        submission_name (str, optional): submission filepath. Defaults to "submission.csv".
    """

    # TODO: Add error handling
    pd.DataFrame({"id": id_col, "attrition": attrition_col}).to_csv(
        submission_name, index=False
    )

    return None


#%%
def str_2_cat(df: pd.DataFrame) -> pd.DataFrame:
    """Converts a pandas dataframes object columns
       catagorical type


    Args:
        df (pd.DataFrame): pandas data frame to be converted

    Returns:
        pd.DataFrame:
    """

    return pd.concat(
        [
            df.select_dtypes(include=np.number),
            df.select_dtypes(exclude=np.number).astype("category"),
        ],
        axis=1,
    )


# %%
def to_one_hot(df: pd.DataFrame, encoder: OneHotEncoder = None) -> pd.DataFrame:
    """Converts a pandas dataframe categorical features
    to one hot encoding

    Args:
        df (pd.DataFrame):

    Returns:
        pd.DataFrame:
    """

    if encoder is None:
        encoder = OneHotEncoder(
            drop="if_binary",
            sparse=False,
            handle_unknown="infrequent_if_exist",  # Don't want to error on the test set
        )

    # Note: this creates columns with spaces
    # might have to change this
    df_cat = pd.DataFrame(
        encoder.fit_transform(df.select_dtypes(exclude=np.number)),
        columns=encoder.get_feature_names_out(),
    )

    return (
        pd.concat(
            [df.select_dtypes(include=np.number), df_cat],
            axis=1,
        ),
        encoder,
    )


# %%
def get_train_test_split(
    df: pd.DataFrame, target_col: str = "Attrition", **kwargs
) -> tuple:
    """Creates the training and test dataset split

    Args:
        df (pd.DataFrame): model dataset
        target_col (str, optional):  Defaults to "Attrition".
        **kwargs: arguments to pass on to sklear.train_test_split,
                including train_size, test_size,
                random_state, shuffle, stratify
    Returns:
        tuple: training and testing data
    """
    try:
        target = df[target_col]
    except KeyError:
        raise ValueError(f"{target_col=} does not exist in dataset")

    return train_test_split(df.drop(target_col, axis=1), target, **kwargs)


# %%


def get_roc(classifier, x: np.ndarray, y: np.ndarray) -> float:
    """Computes the ROC (Receiver Operator Characteristic) score

    Args:
        classifier: a binary classifier which implements the predict_proba method
        x (np.ndarray): the dataset on which to produce the prediction
        y (np.ndarray): true scores

    Returns:
        float: the ROC score
    """
    ONE_CLASS_INDEX = 1
    return roc_auc_score(y, classifier.predict_proba(x)[:, ONE_CLASS_INDEX])


# %%
# Optuna objective to train
def objective(trial, x_train, y_train, x_test, y_test) -> float:
    """Optuna study objective function to run parameter search
    for LightGBM, XGBoost, RandomForest and CatBoost
    """

    classifier_name = trial.suggest_categorical(
        "classifier", ["LGBM", "RF", "XGB", "CAT"]
    )
    roc_score = lambda classifier: get_roc(classifier=classifier, x=x_test, y=y_test)

    roc: float = 0
    if classifier_name == "LGBM":
        lgbm_params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": 42,
            "n_estimators": trial.suggest_int("n_estimators", 50, 5000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 512),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.1, 1.0, log=True
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.1, 1.0, log=True
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 15),
        }

        lgb = gb.LGBMClassifier(**lgbm_params)
        lgb.fit(
            x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=100
        )
        roc = roc_score(lgb)

    elif classifier_name == "RF":
        pass

    # elif classifier_name == "XGB":
    # gb_params = {
    #     "max_depth": trial.suggest_int("max_depth", 2, 10),
    #     "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
    #     "n_estimators": trial.suggest_int("n_estimators", 30, 6000),
    #     "subsample": trial.suggest_float("subsample", 0.2, 1.0),
    #     "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
    # }
    # xg = xgb.XGBClassifier(**gb_params)
    # xg.fit(x_train, y_train, eval_set=[(x_test, y_test)])
    # roc = roc_score(xg)

    # elif classifier_name == "CAT":
    #     # TODO: get better param list
    #     cb_params = {
    #         "loss_function": "Logloss",
    #         "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
    #         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.001, 1, log=True),
    #         "depth": trial.suggest_int("depth", 3, 12),
    #         "boosting_type": "Ordered",
    #         "bootstrap_type": "Bernoulli",
    #         "min_data_in_leaf": trial.suggest_int("depth", 3, 45),
    #     }
    #     cbc = cb.CatBoostClassifier(**cb_params)
    #     cbc.fit(x_train, y_train, eval_set=[(x_test, y_test)])
    #     roc = roc_score(cbc)

    return roc


# %%
# Extract the best model and re-train it on the entire training set
def train_best(x, y, params: dict):
    """Train the best model on the entire training set

    Args:
        x (_type_): training features
        y (_type_): target
        params (dict): best model parameters
    """
    classifier = params.pop("classifier")
    classifiers = {
        "LGBM": gb.LGBMClassifier,
        "RF": RandomForestClassifier,
        "XGB": xgb.XGBClassifier,
        "CAT": cb.CatBoostClassifier,
    }

    # Extract the best classifier
    best_classifier = classifiers[classifier](**params)

    # Re-train on the data to get the best generalizability score
    best_classifier.fit(x, y)

    # Report ROC
    roc = get_roc(best_classifier, x, y)

    print(f"{classifier} : {roc=}")

    return best_classifier


# %%
def submit_best_model(best_model, x: np.array) -> None:
    """Submit the best model"""
    ONE_CLASS_INDEX = 1  # Class which holds the positive predictions
    probability_prediction = best_model.predict_proba(x)[:, ONE_CLASS_INDEX]
    submit(x.id.values, probability_prediction)
    return


if __name__ == "__main__":
    # %%  Import the data
    train_df = read_data(file_path="train.csv", data_path="data")
    test_df = read_data(file_path="test.csv", data_path="data")

    # %% Exploratory Data Analysis
    train_one_hot_df, enc = to_one_hot(train_df)
    train_cat_df = str_2_cat(train_df)

    # %% Train and Test Split

    # One Hot Encoding for Random Forest
    xh_train, xh_test, yh_train, yh_test = get_train_test_split(
        df=train_one_hot_df,
        target_col="Attrition",
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # Categorical for LGBM, XGB and CatBoost
    xc_train, xc_test, yc_train, yc_test = get_train_test_split(
        df=train_one_hot_df,
        target_col="Attrition",
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # %% Parameter Search
    study = op.create_study(direction="maximize")
    study.optimize(
        lambda x: objective(x, xc_train, yc_train, xc_test, yc_test),
        n_trials=500,
        timeout=600,
    )

    # %% Extract the best model from the Optuna study
    submit_best_model(
        train_best(
            train_df.drop("Attrition", axis=1),
            train_df["Attrition"],
            study.best_trial.params,
        )
    )
