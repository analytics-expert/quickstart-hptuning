import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_digits


def load_data():
    """
    Loads and returns the digits dataset for classification.

    Returns:
    -------
    X: array-like, shape (n_samples, n_features)
        The input samples.
    y: array-like, shape (n_samples,)
        The target values (class labels) as integers.
    """
    wine = load_digits()
    X = wine.data
    y = wine.target
    return X, y


def split_data(X, y):
    """
    Splits the dataset into train and validation sets.

    Parameters:
    ----------
    X: array-like, shape (n_samples, n_features)
        The input samples.
    y: array-like, shape (n_samples,)
        The target values (class labels) as integers.

    Returns:
    -------
    X_train, X_val: array-like
        The input samples for train and validation sets.
    y_train, y_val: array-like
        The target values for train and validation sets.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train, params={}):
    """
    Trains a decision tree classifier on the input training data and
    returns the trained model.

    Parameters:
    ----------
    X_train: array-like, shape (n_samples, n_features)
        The input training samples.
    y_train: array-like, shape (n_samples,)
        The target values (class labels) for training samples as integers.
    params: dictionary, optional (default={})
        The parameters for the decision tree classifier.

    Returns:
    -------
    model: DecisionTreeClassifier
        The trained decision tree classifier model.
    """
    params["random_state"] = 42
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluates the performance of the trained model on the input
    validation data.

    Parameters:
    ----------
    model: DecisionTreeClassifier
        The trained decision tree classifier model.
    X_val: array-like, shape (n_samples, n_features)
        The input validation samples.
    y_val: array-like, shape (n_samples,)
        The target values (class labels) for validation samples as integers.

    Returns:
    -------
    score: float
        The accuracy score of the trained model on the input validation data.
    """
    return model.score(X_val, y_val)


def objective(trial, X_train, y_train):
    """
    Objective function for the Optuna optimization.
    Trains a decision tree classifier on the input training data with
    the hyperparameters sampled from the trial.
    Returns the cross-validation accuracy score.

    Parameters:
    ----------
    trial: optuna.trial.Trial
        The object that stores the hyperparameters to be optimized.
    X_train: array-like, shape (n_samples, n_features)
        The input training samples.
    y_train: array-like, shape (n_samples,)
        The target values (class labels) for training samples as integers.

    Returns:
    -------
    score: float
        The cross-validation accuracy score of the trained model.
    """
    params = {
        "criterion": trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 4
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 5
        ),
        "max_features": trial.suggest_categorical(
            "max_features", [None, "sqrt", "log2"]
        ),
    }
    model = train_model(X_train, y_train, params)
    score = cross_val_score(
        model, X_train, y_train, cv=5, n_jobs=-1
    ).mean()
    return score


def tune_hyperparameters(X_train, y_train, n_trials):
    """
    Performs hyperparameter tuning using Optuna library.

    vbnet
    Copy code
    Parameters:
    ----------
    X_train: array-like, shape (n_samples, n_features)
        The input training samples.
    y_train: array-like, shape (n_samples,)
        The target values (class labels) for training samples as integers.
    n_trials: int
        The number of trials for hyperparameter tuning.

    Returns:
    -------
    best_params: dictionary
        The best set of hyperparameters obtained from the Optuna optimization.
    n_trials: int
        The total number of trials that were evaluated.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
    )
    return study.best_params, n_trials


def main():
    """
    Main function to run the decision tree classification pipeline.
    """
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)

    # train a decision tree classifier on the input training data
    dtc = train_model(X_train, y_train)
    # evaluate the performance of the trained model on the input
    # validation data
    val_score = evaluate_model(dtc, X_val, y_val)
    print(f"Validation accuracy before tuning: {val_score:.2f}")

    # perform hyperparameter tuning using Optuna library
    best_params, num_trials = tune_hyperparameters(
        X_train, y_train, n_trials=20
    )

    print(f"Best hyperparameters: {best_params}")
    print(f"Number of trials: {num_trials}")

    # train a decision tree classifier with the best hyperparameters
    # on the input training data
    dtc_best = train_model(X_train, y_train, best_params)
    # evaluate the performance of the trained model on the 
    # input validation data
    val_score_full = evaluate_model(dtc_best, X_val, y_val)
    print(f"Validation accuracy after tuning: {val_score_full:.2f}")


main()
