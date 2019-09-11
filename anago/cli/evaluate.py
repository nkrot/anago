# coding: utf8

import os
import sys
import plac

from ..utils import load_data_and_labels, MODEL_COMPONENTS
from .common import load_model_from_directory

@plac.annotations(
    model_dir = (
        "path to directory where model files are located. "
        "Model files are: {}".format(", ".join(MODEL_COMPONENTS.values())),
        "option", "m"),
    test_file = ("path to test dataset (tsv format)", "positional")
)

def evaluate(model_dir,
             *test_file):
    """
    Evaluate given model on given dataset, computing the following metrics:
     * f1 score (micro)
    """

    model_dir = model_dir or '.'

    # Check paths we were given
    for fpath in test_file:
        if not os.path.isfile(fpath):
            print("Test file not found: {}".format(fpath), file=sys.stderr)
            sys.exit(14)

    model = load_model_from_directory(model_dir)

    x = []; y_true = []
    for fpath in test_file:
        bulk = load_data_and_labels(fpath)
        x.extend(bulk[0])
        y_true.extend(bulk[1])

    score = model.score(x, y_true)

    print("Dataset size: {}".format(len(y_true)))
    print("F1-score: {}".format(score))
