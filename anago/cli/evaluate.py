# coding: utf8

import os
import sys
import plac

from ..utils import load_data_and_labels, MODEL_COMPONENTS
from ..wrapper import Sequence

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

    if not os.path.isdir(model_dir):
        print("Model directory not found: {}".format(model_dir))
        sys.exit(10)

    # Check paths we were given
    for fpath in test_file:
        if not os.path.isfile(fpath):
            print("Test file not found: {}".format(fpath), file=sys.stderr)
            sys.exit(14)

    model_files = { comp : os.path.join(model_dir, fname)
                    for comp,fname in MODEL_COMPONENTS.items() }

    ok = True
    for _,fpath in model_files.items():
        if not os.path.isfile(fpath):
            print("Model file not found: {}".format(fpath), file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(15)

    model = Sequence.load(model_files['weights_file'],
                          model_files['params_file'],
                          model_files['preprocessor_file'])

    x = []; y_true = []
    for fpath in test_file:
        bulk = load_data_and_labels(fpath)
        x.extend(bulk[0])
        y_true.extend(bulk[1])

    score = model.score(x, y_true)

    print("Dataset size: {}".format(len(y_true)))
    print("F1-score: {}".format(score))
