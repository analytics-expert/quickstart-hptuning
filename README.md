# Hyperparameter Tuning Quick Start Guide

This repository provides a collection of Python scripts showcasing hyperparameter tuning techniques using popular libraries such as scikit-learn, Optuna, TPOT, pyswarm, and keras-tuner. The models were built with scikit-learn and Keras, and the scripts are intended as a quick start guide for hyperparameter tuning on a simple handwritten digit classification problem using the MNIST dataset. You can easily adapt these scripts to your own project. For more details, please refer to the accompanying book (link provided).

## Requirements and Installation

First, clone this repository and next, navigate to the project directory: 

```
git clone https://github.com/analytics-expert/quickstart-hptuning.git
cd quickstart-hptuning
```

To run the scripts, you need to have Python 3.10.9 installed on your system. You can create a virtual environment using conda with the following commands:

```
conda create -n htqsg python=3.10.9
conda activate htqsg
```

Then, install the required packages by running:

```
pip install -r requirements.txt
```

## Usage

The following scripts are available for running different hyperparameter tuning techniques:

- ```grid_search.py```: Grid search
- ```random_search.py```: Random search
- ```bayesian_optimization.py```: Bayesian optimization
- ```genetic_algorithms.py```: Genetic algorithms
- ```pso.py```: Particle Swarm Optimization
- ```deep_learning.py```: Tuning deep learning models

To run a script, use the following command:

```
python src/<script_name>.py
```

## Results
After running each script, you will see the validation accuracy before and after hyperparameter tuning, as well as the best hyperparameters found by each method.

### Grid Search

| Before        | After         |
| ------------- |:-------------:|
| 0.84      | 0.89 |

Best hyperparameters found:
```
{
    'criterion': 'entropy', 
    'max_depth': 15, 
    'max_features': None, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2
}
```

### Random Search

| Before        | After         |
| ------------- |:-------------:|
| 0.84      | 0.89 |

Best hyperparameters found:
```
{
    'criterion': 'log_loss', 
    'max_depth': 10, 
    'max_features': None, 
    'min_samples_leaf': 1, 
    'min_samples_split': 3
}
```

### Bayesian Optimization

| Before        | After         |
| ------------- |:-------------:|
| 0.84      | 0.89 |

Best hyperparameters found:
```
{
    'criterion': 'entropy', 
    'max_depth': 10, 
    'max_features': None, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2
}
```
Iterations
```
[I 2023-04-25 13:42:24,881] A new study created in memory with name: no-name-54ce41a4-9c4c-4bef-9f76-7f29ebe727ba
[I 2023-04-25 13:42:30,593] Trial 0 finished with value: 0.6729239256678282 and parameters: {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2'}. Best is trial 0 with value: 0.6729239256678282.
[I 2023-04-25 13:42:32,384] Trial 1 finished with value: 0.7606416957026713 and parameters: {'criterion': 'log_loss', 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': 'sqrt'}. Best is trial 1 with value: 0.7606416957026713.
[I 2023-04-25 13:42:32,492] Trial 2 finished with value: 0.7182249322493225 and parameters: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'}. Best is trial 1 with value: 0.7606416957026713.
[I 2023-04-25 13:42:32,673] Trial 3 finished with value: 0.44048586914440574 and parameters: {'criterion': 'log_loss', 'max_depth': 3, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'log2'}. Best is trial 1 with value: 0.7606416957026713.
[I 2023-04-25 13:42:32,741] Trial 4 finished with value: 0.7808265582655827 and parameters: {'criterion': 'entropy', 'max_depth': 11, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2'}. Best is trial 4 with value: 0.7808265582655827.
[I 2023-04-25 13:42:32,812] Trial 5 finished with value: 0.7564677700348431 and parameters: {'criterion': 'entropy', 'max_depth': 11, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': 'sqrt'}. Best is trial 4 with value: 0.7808265582655827.
[I 2023-04-25 13:42:32,848] Trial 6 finished with value: 0.7446017228029423 and parameters: {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_features': 'log2'}. Best is trial 4 with value: 0.7808265582655827.
[I 2023-04-25 13:42:32,949] Trial 7 finished with value: 0.7905802361595045 and parameters: {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': None}. Best is trial 7 with value: 0.7905802361595045.
[I 2023-04-25 13:42:33,041] Trial 8 finished with value: 0.7522841656987999 and parameters: {'criterion': 'gini', 'max_depth': 14, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': 'sqrt'}. Best is trial 7 with value: 0.7905802361595045.
[I 2023-04-25 13:42:33,092] Trial 9 finished with value: 0.7627105110336817 and parameters: {'criterion': 'log_loss', 'max_depth': 8, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt'}. Best is trial 7 with value: 0.7905802361595045.
[I 2023-04-25 13:42:33,156] Trial 10 finished with value: 0.6993950832365468 and parameters: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_features': None}. Best is trial 7 with value: 0.7905802361595045.
[I 2023-04-25 13:42:33,249] Trial 11 finished with value: 0.858052651955091 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': None}. Best is trial 11 with value: 0.858052651955091.
[I 2023-04-25 13:42:33,326] Trial 12 finished with value: 0.8462325783972124 and parameters: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_features': None}. Best is trial 11 with value: 0.858052651955091.
[I 2023-04-25 13:42:33,422] Trial 13 finished with value: 0.8462253193960512 and parameters: {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 2, 'min_samples_leaf': 3, 'max_features': None}. Best is trial 11 with value: 0.858052651955091.
[I 2023-04-25 13:42:33,510] Trial 14 finished with value: 0.8469197638404957 and parameters: {'criterion': 'entropy', 'max_depth': 11, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_features': None}. Best is trial 11 with value: 0.858052651955091.
[I 2023-04-25 13:42:33,643] Trial 15 finished with value: 0.8295102593883081 and parameters: {'criterion': 'entropy', 'max_depth': 11, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': None}. Best is trial 11 with value: 0.858052651955091.
[I 2023-04-25 13:42:33,797] Trial 16 finished with value: 0.8476190476190476 and parameters: {'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': None}. Best is trial 11 with value: 0.858052651955091.
[I 2023-04-25 13:42:34,011] Trial 17 finished with value: 0.8476190476190476 and parameters: {'criterion': 'entropy', 'max_depth': 13, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': None}. Best is trial 11 with value: 0.858052651955091.
[I 2023-04-25 13:42:34,091] Trial 18 finished with value: 0.8469246031746032 and parameters: {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': None}. Best is trial 11 with value: 0.858052651955091.
[I 2023-04-25 13:42:34,186] Trial 19 finished with value: 0.8538787262872628 
```

### Genetic Algorithms
| Before        | After         |
| ------------- |:-------------:|
| 0.84      | 0.89 |

Best hyperparameters found:
```
{
    'criterion': 'log_loss', 
    'max_depth': 10, 
    'max_features': None, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2
}
```

### PSO

| Before        | After         |
| ------------- |:-------------:|
| 0.84      | 0.89 |

Best hyperparameters found:
```
{
    'criterion': 'log_loss', 
    'max_depth': 12, 
    'max_features': None, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2
}
```

Stopping search: maximum iterations reached --> 20


### Tunning Deep Learning Models
| Before        | After         |
| ------------- |:-------------:|
| 0.90      | 0.95 |

Best hyperparameters:
```
{
    'units': 128,
    'learning_rate': 0.001
}
```

Iterations
```
Epoch 1/3
47/47 [==============================] - 2s 16ms/step - loss: 1.7313 - accuracy: 0.3759 - val_loss: 0.9832 - val_accuracy: 0.7427
Epoch 2/3
47/47 [==============================] - 0s 10ms/step - loss: 1.2150 - accuracy: 0.5806 - val_loss: 0.6201 - val_accuracy: 0.8778
Epoch 3/3
47/47 [==============================] - 0s 10ms/step - loss: 1.0404 - accuracy: 0.6413 - val_loss: 0.4663 - val_accuracy: 0.9023
313/313 [==============================] - 1s 3ms/step - loss: 0.4720 - accuracy: 0.9025
Validation accuracy before tuning: 0.90


Epoch 1/3
47/47 [==============================] - 2s 16ms/step - loss: 0.9229 - accuracy: 0.7188 - val_loss: 0.3092 - val_accuracy: 0.9117
Epoch 2/3
47/47 [==============================] - 0s 11ms/step - loss: 0.3865 - accuracy: 0.8844 - val_loss: 0.2186 - val_accuracy: 0.9382
Epoch 3/3
47/47 [==============================] - 0s 11ms/step - loss: 0.2984 - accuracy: 0.9120 - val_loss: 0.1701 - val_accuracy: 0.9535
313/313 [==============================] - 1s 3ms/step - loss: 0.1730 - accuracy: 0.9505
Validation accuracy after tuning: 0.95
```
