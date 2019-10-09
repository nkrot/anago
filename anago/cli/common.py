
import os
import sys

import seqeval.metrics

from ..utils import MODEL_COMPONENTS
from ..wrapper import Sequence

NUMBER_OF_DECIMALS = 4

REQUIRED_METRICS = {
    # NOTE: seqeval implementations ignore the options :(
    'f1_score'        : { 'average' : 'micro' },
    'precision_score' : { 'average' : 'micro' },
    'recall_score'    : { 'average' : 'micro' }
}

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

def print_data_and_labels(x, y1, y2=None, **kwargs):
    """
    Print data (x) and labels (y1 and y2 if given).

    optional keyword arguments:
    file: a file-like object (stream); defaults to sys.stdout
    """

    fd = kwargs.get('file', sys.stdout)

    for sent_idx in range(len(x)):
        if y2:
            sent_words = zip(x[sent_idx], y1[sent_idx], y2[sent_idx])
        else:
            sent_words = zip(x[sent_idx], y1[sent_idx])

        for word in sent_words:
            print("\t".join(word), file=fd)
        print(file=fd)

    pass

def collect_scores(y_true, y_pred):
    """
    Compute all required scores (as specified in REQUIRED_METRICS)
    from given ground truth and actual predictions.

    Return
      A dictionary of the form:
        { metric_1 : value, metric_2 : value }
    """

    scores = {}
    for metric_name,options in REQUIRED_METRICS.items():
        scoring_method = getattr(seqeval.metrics, metric_name)
        score = scoring_method(y_true, y_pred, **options)
        scores[metric_name] = round(score, NUMBER_OF_DECIMALS)
    return scores
