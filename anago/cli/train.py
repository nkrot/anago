# coding: utf8

import os
import sys
import plac

from ..utils import load_data_and_labels, MODEL_COMPONENTS
from ..wrapper import Sequence
from ..config import Config

@plac.annotations(
    config_file = (
        "path to configuration file. Values given on the command line"
        " override corresponding values in the configuration file.",
        "option", "c"),
    model_dir = ("path to directory where trained model will be saved.",
                 "option", "m"),
    epochs = ("set number of training epochs.",
              "option", "e", int),
    training_file = ("Path to training dataset (tsv format).",
                     "positional")
)

def train(config_file,
          model_dir,
          epochs,
          *training_file):
    """
    Train a Sequence model and store model files in given directory or in
    the current directory if none is given. Existing files with the same names
    will be silently overriden.\
    """

    cfg = Config(config_file) if config_file else Config()

    if model_dir:
        cfg.set('model_dir', model_dir)
    else:
        cfg.set('model_dir', '.')

    if epochs:
        cfg.set('max_epoch', epochs)

    try:
        os.makedirs(cfg.get('model_dir'), exist_ok=True)
    except FileExistsError as err:
        print("Error with model directory: {}\n".format(err), file=sys.stderr)
        sys.exit(10)

    if not os.access(cfg.get('model_dir'), os.W_OK):
        print("Output directory not writable: {}".format(cfg.get('model_dir')),
              file=sys.stderr)
        sys.exit(11)

    # EXPERIMENTS: dont commit

    #cfg.set('verbose', [1, 2]) # ok. fails because verbose can not be a list
    print(cfg.set('data_dir', 'mywork/data')) # returns old value
    print(cfg.get('weights_file')) # ok, model_dir was updated and appears in the path
    print(cfg.get('valid_data'))   # ok, model_dir was updated and appears in the path

    cfg.set('test_data', '/path/to/dir/file with a space in the name.txt')
    print(cfg.get('test_data')) # returns one item, not a list
    cfg.set('train_data', ['file1', '%(data_dir)s/file2'])
    print(cfg.get('train_data' )) # returns a list

    #cfg.set('glove', 'path', 'path/to/glove.txt') # ok, sets though initially not in config
    cfg.set('path', '/my/glove.txt') # ok. finds section glove and sets path
    #cfg.set('glove', 'shilo', 'path/to/glove.txt') # ok. exits with error

    cfg.write()
    exit(100)

    # Strategy for *training_file*
    # if it is set on the command line, then it is used directly (we set these
    # paths to cfg but dont use them -- this way it is safer in case the filename
    # have weird characters that configparser may fuck up)
    # In general, what is set on the command line, overrides the config file.
    # Other files -- valid_data and test_data -- are taken from the config file
    # if they exist there. For the time being, there is no way to override these
    # files from the command line nor turn off this section in the configuration
    # file. If you dont want to use valid_data dna test_data, disable these
    # parameters in the config file.

    if len(training_file) == 0:
        print("No training files were given. Really?", file=sys.stderr)
        sys.exit(12)

    # Check paths we were given
    for fpath in training_file:
        if not os.path.isfile(fpath):
            print("Training file not found: {}".format(fpath), file=sys.stderr)
            sys.exit(13)

    x_train, y_train = [], []
    for fpath in training_file:
        bulk = load_data_and_labels(fpath)
        x_train.extend(bulk[0])
        y_train.extend(bulk[1])

    # TODO: configure object from cfg
    model = Sequence()

    # TODO: configure method from cfg
    model.fit(x_train, y_train, epochs=epochs)

    # TODO: get paths from cfg
    model.save(os.path.join(model_dir, MODEL_COMPONENTS['weights_file']),
               os.path.join(model_dir, MODEL_COMPONENTS['params_file']),
               os.path.join(model_dir, MODEL_COMPONENTS['preprocessor_file']))
