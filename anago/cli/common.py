
import os
import sys

from ..utils import MODEL_COMPONENTS
from ..wrapper import Sequence

def load_model_from_directory(model_dir='.'):
    """
    Load model from given directory. A model is a collection of files
    with predefined names specified in MODEL_COMPONENTS.
    TODO: need to list the names here, but dont know how :)
    """ #TODO.format(", ".join(MODEL_COMPONENTS.values()))

    if not os.path.isdir(model_dir):
        print("Model directory not found: {}".format(model_dir))
        sys.exit(10)

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

    return model
