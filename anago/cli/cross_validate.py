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

this = sys.modules[__name__]

this.can_save           = False
this.append             = False
this.save_dir_name      = os.path.curdir
this.save_file_infix    = ".cv.p-"
this.save_file_basename = "anago"
this.save_file_ext      = '.tsv'

def _prepare_env_for_saving(path):
    """
    Compute values of variables that will be used when saving data.
    """

    if os.path.isdir(path):
        this.save_dir_name = path
    else:
        dirpath, fname = os.path.split(path)

        if len(dirpath) > 0:
            os.makedirs(dirpath, exist_ok=True)
            this.save_dir_name = dirpath

        if len(fname) > 0:
            this.save_file_basename = fname

    this.can_save = True

def _get_filename(n):
    """
    Construct filename w/o path for the n-th part of cross validation.
    """

    fname_parts = [ this.save_file_basename,
                    this.save_file_infix,
                    n if type(n) == str else "{:02d}".format(n),
                    this.save_file_ext ]
    return "".join(fname_parts)

def _get_full_filename(n):
    """
    Construct full filename (including path) for the n-th part of cross validation.
    """
    return os.path.join(this.save_dir_name, _get_filename(n))

def _opt2path(path, n):
    """
    A helper function to be used only in @plac.annotations
    """
    _prepare_env_for_saving(path)
    return _get_full_filename(n)

def _save_predictions_to(fname, x, y_true, y_pred):
    with open(fname, "w+") as fd:
        if this.append:
            common.print_data_and_labels(x, y_true, y_pred, file=fd)
        else:
            common.print_data_and_labels(x, y_pred, file=fd)

@plac.annotations(
    folds  = ("set the number of folds (k in k-fold)", "option", "k", int),
    epochs = ("set number of training epochs", "option", "e", int),
    save   = (
        "save predictions on test subset in each run of cross validation."
        " (a) If the option value is a directory, files will be saved in it."
        f" Files themselves will be named '{_get_filename('<NN>')}', where <NN>"
        " corresponds to the number of the current fold. For example, the 5th fold"
        f" will be saved under the name '{_get_filename(5)}' in the directory specified."
        " (b) if option value is not a directory, it will be used as basename of"
        " all output files. For example, if the option value is 'validation_set,'"
        f" then the 1st fold will be saved in '{_opt2path('validation_set', 1)}'"
        " or if the option value is './results/test.txt', the 10th fold will be"
        f" saved in '{_opt2path('./results/test.txt', 10)}'."
        " (c) In any case, non-existing directories (including intermediate ones)"
        f" will be created.",
        "option", "s", str, None, "/path/to/dir_or_file"),
    append = (
        "when saving dataset, predicted tag will be appended to the original data"
        " as additional column. This mode implies --save.",
        "flag", "a"),
    training_file = ("Path to training dataset (tsv format)", "positional")
)

def cross_validate(folds=10,
                   epochs=15,
                   save=None,
                   append=False,
                   *training_file):
    """
    Perform k-fold cross validation of the model and report various scores
    -- precision, recall, f1 -- for all runs individually as well as mean
    and standard deviation for every above-mentioned score.
    Data is randomized before folds are constructed.

    Note that no model will be saved.\
    """

    print(f"Performing cross validation with k={folds}")

    if save is not None:
        _prepare_env_for_saving(save)

    this.append = append

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

        if this.can_save:
            _save_predictions_to(_get_full_filename(run_idx), x_test, y_test, y_pred)

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
