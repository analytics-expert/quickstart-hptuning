from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_digits


def load_data():
    """
    Loads the digits dataset from scikit-learn.

    Returns:
    - X (ndarray): the features of the dataset
    - y (ndarray): the targets of the dataset
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y


def split_data(X, y):
    """
    Splits the dataset into training and validation sets.

    Args:
    - X (ndarray): the features of the dataset
    - y (ndarray): the targets of the dataset

    Returns:
    - X_train (ndarray): the training features
    - X_val (ndarray): the validation features
    - y_train (ndarray): the training targets
    - y_val (ndarray): the validation targets
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train, params={}):
    """
    Trains a decision tree classifier with given hyperparameters on the
    training set.

    Args:
    - X_train (ndarray): the training features
    - y_train (ndarray): the training targets
    - params (dict): hyperparameters for the decision tree classifier

    Returns:
    - model (DecisionTreeClassifier): the trained decision tree classifier
    """
    params["random_state"] = 42
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluates the accuracy of a trained model on the validation set.

    Args:
    - model (DecisionTreeClassifier): the trained decision tree classifier
    - X_val (ndarray): the validation features
    - y_val (ndarray): the validation targets

    Returns:
    - score (float): the accuracy score of the trained model on the
        validation set
    """
    return model.score(X_val, y_val)


def tune_hyperparameters(model, X_train, y_train, param_grid):
    """
    Uses grid search to find the best hyperparameters for a given model and
    training set.

    Args:
    - model (DecisionTreeClassifier): the decision tree classifier to tune
    - X_train (ndarray): the training features
    - y_train (ndarray): the training targets
    - param_grid (dict): a dictionary of hyperparameter values to search over

    Returns:
    - best_params (dict): the best hyperparameters found by the grid search
    - num_combinations (int): the number of hyperparameter combinations tried
                              by the grid search
    """
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, len(
        grid_search.cv_results_["params"]
    )


def main():
    """
    Runs the main program.
    """
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)

    dtc = train_model(X_train, y_train)
    val_score = evaluate_model(dtc, X_val, y_val)
    print(f"Validation accuracy before tuning: {val_score:.2f}")

    # define hyperparameter grid to search over
    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": [None, "sqrt", "log2"],
    }

    # tune hyperparameters with grid search
    best_params, num_combinations = tune_hyperparameters(
        dtc, X_train, y_train, param_grid
    )

    print(f"Best hyperparameters: {best_params}")
    print(f"Number of combinations tried: {num_combinations}")

    # train a new model with the best hyperparameters found
    dtc_best = train_model(X_train, y_train, best_params)
    val_score_full = evaluate_model(dtc_best, X_val, y_val)
    print(f"Validation accuracy after tuning: {val_score_full:.2f}")

if __name__ == 'main':
    main()