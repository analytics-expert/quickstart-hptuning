import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam

from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from dataset import load_data_tensorflow

def build_model(hp):
    """
    Build and compile a CNN model with tunable hyperparameters using Keras Tuner.

    Args:
        hp: HyperParameters object from Keras Tuner

    Returns:
        model: Keras Sequential model
    """
    model = Sequential()
    model.add(
        Conv2D(
            16,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(
        Dense(
            units=hp.Int(
                "units", min_value=16, max_value=128, step=16
            ),
            activation="relu",
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer=Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    """
    Load the data, build and train a default model, tune the hyperparameters,
    and evaluate the best model.

    Returns:
        None
    """
    x_train, x_test, y_train, y_test = load_data_tensorflow()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Build the model with default hyperparameters
    default_hp = HyperParameters()
    default_hp.Fixed("units", value=32)
    default_hp.Fixed("learning_rate", value=0.001)
    default_model = build_model(default_hp)
    default_model.fit(
        x_train, y_train, epochs=3, validation_split=0.2
    )
    (_, val_acc) = default_model.evaluate(x_test, y_test)
    print(f"Validation accuracy before tuning: {val_acc:.2f}")

    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,
        directory="mnist_tuning",
        project_name="mnist_tuning",
    )

    # Summarize the search space of hyperparameters
    tuner.search_space_summary()

    # Search for the best hyperparameters
    tuner.search(x_train, y_train, epochs=3, validation_split=0.2)

    # Get the best hyperparameters and build the
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hyperparameters.values}")

    best_model = tuner.hypermodel.build(best_hyperparameters)
    best_model.fit(x_train, y_train, epochs=3, validation_split=0.2)

    (_, val_score_full) = best_model.evaluate(x_test, y_test)
    print(f"Validation accuracy after tuning: {val_score_full:.2f}")


main()
