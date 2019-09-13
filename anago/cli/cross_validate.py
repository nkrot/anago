import os
import sys
import plac

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import seqeval.metrics

from ..utils import load_data_and_labels
from ..wrapper import Sequence
from . import common

@plac.annotations(
    folds  = ("set the number of folds (k in k-fold)", "option", "k", int),
    epochs = ("set number of training epochs", "option", "e", int),
    training_file = ("Path to training dataset (tsv format)", "positional")
)

def cross_validate(folds=10,
                   epochs=15,
                   *training_file):
    """
    Perform k-fold cross validation of the model and report various scores
    -- precision, recall, f1 -- for all runs individually as well as mean
    and standard deviation for every score.

    Note that neither a model nor test results will be saved.\
    """

    print(f"Performing cross validation with k={folds}")


    x = []; y = []
    for fpath in training_file:
        bulk = load_data_and_labels(fpath)
        x.extend(bulk[0])
        y.extend(bulk[1])

    x, y = np.array(x), np.array(y)

    kf = KFold(n_splits=folds, shuffle=True)
    all_scores = []

    for idx,(train_indices,test_indices) in enumerate(kf.split(x)):
        run_idx = idx+1
        print("Iteration #{}, train/test split sizes: {}/{}".format(
            run_idx, len(train_indices), len(test_indices)))

        x_train, y_train = x[train_indices], y[train_indices]
        x_test,  y_test  = x[test_indices],  y[test_indices]

        model = Sequence()
        model.fit(x_train, y_train, epochs=epochs)

        y_pred = model.predict(x_test)

        onerun = {}
        for methname,options in common.REQUIRED_METRICS.items():
            meth = getattr(seqeval.metrics, methname)
            score_val = meth(y_test, y_pred, **options)
            print("RUN {} SCORE {}: {}".format(run_idx, methname, score_val))
            onerun[methname] = score_val

        all_scores.append(onerun)

    # An example
    # all_scores = [{ 'f1_score': 0.8952, 'precision' : 0.98 }, <-- 1st run
    #               { 'f1_score': 0.7420, 'precision' : 0.80 }, <-- 2nd run
    #               { 'f1_score': 0.8337, 'precision' : 0.70 }] <-- 3rd run

    df = pd.DataFrame(all_scores)
    df.index += 1
    df = df.append(df.agg(['mean', 'std']))
    df = df.round(common.NUMBER_OF_DECIMALS)

    print("=== Cross Validation Report ===")
    print("Dataset size: {}".format(len(x)))
    print("Number of folds: {}".format(folds))
    print("Number of epochs: {}".format(epochs))
    print("")
    print(df)
