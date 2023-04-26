from tpot import TPOTClassifier
from sklearn.tree import DecisionTreeClassifier
from dataset import load_data_sklearn


def main():
    """
    Main function to run the decision tree classification pipeline with 
    hyperparameter tuning using TPOT library.
    """
    X_train, X_val, y_train, y_val = load_data_sklearn()

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
    best_params = pipeline_optimizer.fitted_pipeline_.steps[-1][1].get_params()
    print(f"Best hyperparameters: {best_params}")

    # train a decision tree classifier with the best hyperparameters on the input training data
    dtc_best = DecisionTreeClassifier(**best_params)
    dtc_best.fit(X_train, y_train)
    # evaluate the performance of the trained model on the input validation data
    val_score_full = dtc_best.score(X_val, y_val)
    print(f"Validation accuracy after tuning: {val_score_full:.2f}")


main()
