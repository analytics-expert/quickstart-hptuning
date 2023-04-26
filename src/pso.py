import numpy as np
from pyswarm import pso
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from dataset import load_data_sklearn


def objective(params, X_train, y_train):
    """
    Defines the objective function to optimize for hyperparameter tuning.

    Parameters:
    ----------
    params: array-like
        The input hyperparameters to optimize.
    X_train: array-like, shape (n_samples, n_features)
        The input samples for training.
    y_train: array-like, shape (n_samples,)
        The target values (class labels) as integers for training.

    Returns:
    -------
    score: float
        The mean cross-validation accuracy score using the given hyperparameters.
    """
    criterion = ["gini", "entropy", "log_loss"][int(params[0])]
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])
    max_features = [None, "sqrt", "log2"][int(params[4])]

    model_params = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "random_state": 42,
    }

    model = DecisionTreeClassifier(**model_params)
    score = cross_val_score(
        model, X_train, y_train, cv=5, n_jobs=-1
    ).mean()
    return -score


def tune_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter tuning using Particle Swarm Optimization (PSO).

    Parameters:
    ----------
    X_train: array-like, shape (n_samples, n_features)
        The input samples for training.
    y_train: array-like, shape (n_samples,)
        The target values (class labels) as integers for training.

    Returns:
    -------
    best_params_dict: dict
        The best hyperparameters found by PSO.
    """
    lb = [0, 3, 2, 1, 0]  # lower bounds for parameters
    ub = [2, 15, 4, 5, 2]  # upper bounds for parameters

    best_params, _ = pso(
        lambda x: objective(x, X_train, y_train),
        lb,
        ub,
        maxiter=20,
        swarmsize=20,
    )

    best_params_int = [int(param) for param in best_params]
    best_params_dict = {
        "criterion": ["gini", "entropy", "log_loss"][best_params_int[0]],
        "max_depth": best_params_int[1],
        "min_samples_split": best_params_int[2],
        "min_samples_leaf": best_params_int[3],
        "max_features": [None, "sqrt", "log2"][best_params_int[4]],
    }

    return best_params_dict


def main():
    X_train, X_val, y_train, y_val = load_data_sklearn(X, y)

    # train a baseline
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)
    val_score = dtc.score(X_val, y_val)
    print(f"Validation accuracy before tuning: {val_score:.2f}")

    # tune hyperparameters with grid search
    best_params = tune_hyperparameters(X_train, y_train)
    print(f"Best hyperparameters: {best_params}")

    # train a new model with the best hyperparameters found
    dtc_best = DecisionTreeClassifier(**best_params, random_state=42)
    dtc_best.fit(X_train, y_train)
    val_score_full = dtc_best.score(X_val, y_val)
    print(f"Validation accuracy after tuning: {val_score_full:.2f}")


main()
