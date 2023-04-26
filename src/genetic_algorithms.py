from tpot import TPOTClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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


def main():
    """
    Main function to run the decision tree classification pipeline with 
    hyperparameter tuning using TPOT library.
    """
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)

    # train a decision tree classifier on the input training data
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)
    # evaluate the performance of the trained model on the input validation data
    val_score = dtc.score(X_val, y_val)
    print(f"Validation accuracy before tuning: {val_score:.2f}")

    # perform hyperparameter tuning using TPOT library
    pipeline_optimizer = TPOTClassifier(
        generations=5,
        population_size=20,
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbosity=2,
        config_dict={
            "sklearn.tree.DecisionTreeClassifier": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": range(3, 16),
                "min_samples_split": range(2, 5),
                "min_samples_leaf": range(1, 6),
                "max_features": [None, "sqrt", "log2"],
            }
        },
    )
    pipeline_optimizer.fit(X_train, y_train)
    # obtain the best hyperparameters from the TPOT optimization
    best_params = pipeline_optimizer.fitted_pipeline_.steps[-1][
        1
    ].get_params()
    print(f"Best hyperparameters: {best_params}")

    # train a decision tree classifier with the best hyperparameters on the input training data
    dtc_best = DecisionTreeClassifier(**best_params)
    dtc_best.fit(X_train, y_train)
    # evaluate the performance of the trained model on the input validation data
    val_score_full = dtc_best.score(X_val, y_val)
    print(f"Validation accuracy after tuning: {val_score_full:.2f}")


main()
