# coding: utf8

import os
import sys

import plac
import seqeval.metrics

from ..utils import load_data_and_labels, MODEL_COMPONENTS
from .common import load_model_from_directory, NUMBER_OF_DECIMALS

@plac.annotations(
    model_dir = (
        "path to directory where model files are located. "
        "Model files are: {}".format(", ".join(MODEL_COMPONENTS.values())),
        "option", "m"),
    output = (
        "output predictions to stdout (tsv format) as well",
        "flag", "o"),
    append = (
        "output test dataset with predicted tags appended to corresponding lines"
        " (tsv format). This mode implies --output.",
        "flag", "a"),
    test_file = ("path to test dataset (tsv format)", "positional")
)

def evaluate(model_dir,
             output=False,
             append=False,
             *test_file):
    """
    Evaluate given model on given dataset, computing the following metrics:
      * f1 score
      * precision
      * recall\
    """

    # TODO: this somewhat repeats the code in cross_validate and can/should
    #       be dried out.
    required_metrics = {
        # NOTE: seqeval implementations ignore the options :(
        'f1_score'        : { 'average' : 'micro' },
        'precision_score' : { 'average' : 'micro' },
        'recall_score'    : { 'average' : 'micro' }
    }

    model_dir = model_dir or '.'
    output = output or append

    # Check paths we were given
    for fpath in test_file:
        if not os.path.isfile(fpath):
            print("Test file not found: {}".format(fpath), file=sys.stderr)
            sys.exit(14)

    x, y_true = [], []
    for fpath in test_file:
        bulk = load_data_and_labels(fpath)
        x.extend(bulk[0])
        y_true.extend(bulk[1])

    model = load_model_from_directory(model_dir)
    y_pred = model.predict(x)

    # TODO: this somewhat repeats the code in cross_validate and can/should
    #       be dried out.
    scores = {}
    for metric_name,options in required_metrics.items():
        scoring_method = getattr(seqeval.metrics, metric_name)
        scores[metric_name] = scoring_method(y_true, y_pred, **options)

    # output predictions
    for sent_idx in range(len(x)):
        if append:
            sent_words = zip(x[sent_idx], y_true[sent_idx], y_pred[sent_idx])
        else:
            sent_words = zip(x[sent_idx], y_pred[sent_idx])
        for word in sent_words:
            print("\t".join(word))
        print()

    # output scores
    print("Dataset size: {}".format(len(y_true)))
    for metric_name in required_metrics.keys():
        print("{}: {}".format(metric_name, round(scores.get(metric_name, 0),
                                                 NUMBER_OF_DECIMALS)))
