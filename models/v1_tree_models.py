#!/usr/bin/env python3

"""
Tabular Plaground Model S3 E3 
"""

#%%
import os

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as gb
import catboost as cb


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

# Import the data
train_df = read_data(file_path="train.csv", data_path="data")
test_df = read_data(file_path="test.csv", data_path="data")

# %%
train_df.shape

# %% Exploratory Data Analysis
train_df, enc = to_one_hot(train_df)


# %%
x_train, x_test, y_train, y_test = get_train_test_split(
    df=train_df,
    target_col="Attrition",
    test_size=0.2,
    random_state=42,
    shuffle=True,
)


# %%
def get_best_model(classifier, param_grid: dict):
    """Runs grid search CV to obtain the best model
    using the provided parameter dictionary

    Args:
        classifier (_type_): Implements the sklearn interface
        param_grid (dict): parameter dictionary
    """
    clf = GridSearchCV(classifier, param_grid=param_grid)


# %%
# Fit Light GBM

lgbm = gb.LGBMClassifier()
lgbm.fit(x_train, y_train)

# %%
probs = lgbm.predict_proba(x_test)[:, 1]
roc_auc_score(y_test.values.reshape(-1), probs)


#%%
# Fit CatBoost
cbc = cb.CatBoostClassifier()
cbc.fit(x_train, y_train)
probs = cbc.predict_proba(x_test)[:, 1]
roc_auc_score(y_test.values.reshape(-1), probs)



# %% Random Forrest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
probs = rf.predict_proba(x_test)[:, 1]
roc_auc_score(y_test.values.reshape(-1), probs)



# %%

# Fit xgboost
xg = xgb.XGBClassifier()
xg.fit(x_train, y_train)
xg.score(x_test, y_test)


# %%
# Submit Prediction
pred_df, _ = to_one_hot(test_df, encoder=enc)
submit(pred_df.id.values, lgbm.predict(pred_df))

# %%
