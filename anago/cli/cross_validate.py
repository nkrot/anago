import os
import sys
import plac

import numpy as np
from sklearn.model_selection import KFold

from ..utils import load_data_and_labels
from ..wrapper import Sequence

@plac.annotations(
    folds  = ("set the number of folds (k in k-fold)", "option", "k", int),
    epochs = ("set number of training epochs", "option", "e", int),
    training_file = ("Path to training dataset (tsv format)", "positional")
)

def cross_validate(folds=10,
                   epochs=15,
                   *training_file):
    """
    Perform k-fold cross validation of the model and report f1 (micro) score
    for all runs as well as mean and standard deviation.

    Note that neither a model will be produced nor test results will be saved.\
    """

    print(f"Performing cross validation with k={folds}")

    required_metrics = ['f1_micro']

    x = []; y = []
    for fpath in training_file:
        bulk = load_data_and_labels(fpath)
        x.extend(bulk[0])
        y.extend(bulk[1])
        print(type(bulk[0]))

    x, y = np.array(x), np.array(y)

    kf = KFold(n_splits=folds, shuffle=True)
    scores = []

    for idx,(train_indices,test_indices) in enumerate(kf.split(x)):
        print("Iteration #{}, train/test split sizes: {}/{}".format(
            idx, len(train_indices), len(test_indices)))

        x_train, y_train = x[train_indices], y[train_indices]
        x_test,  y_test  = x[test_indices],  y[test_indices]

        model = Sequence()
        model.fit(x_train, y_train, epochs=epochs)

        scores.append({
            'f1_micro' : model.score(x_test, y_test)
            # TODO: add here precision, recall
        })

    # An example
    # scores = [{'f1_micro': 0.2952456002101392,  'precision' : 0.98},
    #           {'f1_micro': 0.24196891191709846, 'precision' : 0.80},
    #           {'f1_micro': 0.23366013071895422, 'precision' : 0.70}]

    for m in required_metrics:
        values = np.array([onerun.get(m, 0) for onerun in scores])
        print("{}: {}".format(m, values))
        print("mean: {}".format(values.mean()))
        print("std:  {}".format(values.std()))

    # TODO: desired output format (use pandas?)
    # Include size of dataset, number of folds and training epochs?
    #           | run #1 | ... | run #N | mean | std |
    # f1-score  |        |     |        |      |     |
    # precision |        |     |        |      |     |
