from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)
from dataset import load_data_sklearn


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
    Evaluates the performance of the trained model on the input validation data.

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


def tune_hyperparameters(model, X_train, y_train, param_grid, n_iter):
    """
    Performs a randomized search over the hyperparameters
    space of the input decision tree classifier.

    Parameters:
    ----------
    model: DecisionTreeClassifier
        The decision tree classifier model.
    X_train: array-like, shape (n_samples, n_features)
        The input training samples.
    y_train: array-like, shape (n_samples,)
        The target values (class labels) for training samples as integers.
    param_grid: dictionary
        The hyperparameter space for the decision tree classifier model.
    n_iter: int
        Number of parameter settings that are sampled.

    Returns:
    -------
    best_params: dictionary
        The best set of hyperparameters obtained from the randomized search.
    n_iter: int
        The total number of parameter settings that were evaluated.
    """
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        n_jobs=-1,
        cv=5,
        random_state=42,
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_, n_iter


def main():
    """
    Main function to run the decision tree classification pipeline.
    """
    X_train, X_val, y_train, y_val = load_data_sklearn()
    # train a decision tree classifier on the input training data
    dtc = train_model(X_train, y_train)
    # evaluate the performance of the trained model on the input validation data
    val_score = evaluate_model(dtc, X_val, y_val)
    print(f"Validation accuracy before tuning: {val_score:.2f}")

    # define the hyperparameters space for the decision tree classifier model
    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [3, 5, 7, 10, 15],
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": [None, "sqrt", "log2"],
    }
    # perform hyperparameter tuning using randomized search
    best_params, num_combinations = tune_hyperparameters(
        dtc, X_train, y_train, param_grid, n_iter=270
    )

    print(f"Best hyperparameters: {best_params}")
    print(f"Number of combinations tried: {num_combinations}")

    # train a decision tree classifier with the best hyperparameters on the input training data
    dtc_best = train_model(X_train, y_train, best_params)
    # evaluate the performance of the trained model on the input validation data
    val_score_full = evaluate_model(dtc_best, X_val, y_val)
    print(f"Validation accuracy after tuning: {val_score_full:.2f}")


main()
