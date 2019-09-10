# coding: utf8

import os
import sys
import plac

from ..utils import load_data_and_labels, MODEL_COMPONENTS
from ..wrapper import Sequence

@plac.annotations(
    model_dir = ("path to directory where trained model will be saved", "option", "m"),
    epochs = ("set number of training epochs", "option", "e", int),
    training_file = ("Path to training dataset (tsv format)", "positional")
)

def train(model_dir,
          epochs=15,
          *training_file):
    """
    Train a Sequence model and store model files in given directory or in
    the current directory if none is given. Existing files with the same names
    will be silently overriden.
    """

    if model_dir:
        try:
            os.makedirs(model_dir, exist_ok=True)
        except FileExistsError as err:
            print("Error with model directory: {}\n".format(err), file=sys.stderr)
            sys.exit(10)

        if not os.access(model_dir, os.W_OK):
            print("Output directory not writable: {}".format(model_dir), file=sys.stderr)
            sys.exit(11)
    else:
        model_dir = '.'

    if len(training_file) == 0:
        print("No training files were given. Really?", file=sys.stderr)
        sys.exit(12)

    # Check paths we were given
    for fpath in training_file:
        if not os.path.isfile(fpath):
            print("Training file not found: {}".format(fpath), file=sys.stderr)
            sys.exit(13)

    x_train = []; y_train = []
    for fpath in training_file:
        bulk = load_data_and_labels(fpath)
        x_train.extend(bulk[0])
        y_train.extend(bulk[1])

    model = Sequence()
    model.fit(x_train, y_train, epochs=epochs)

    model.save(os.path.join(model_dir, MODEL_COMPONENTS['weights_file']),
               os.path.join(model_dir, MODEL_COMPONENTS['params_file']),
               os.path.join(model_dir, MODEL_COMPONENTS['preprocessor_file']))
