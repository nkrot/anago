# coding: utf8

import os
import re
import sys

import plac
import seqeval.metrics

from .. import utils
from . import common

@plac.annotations(
    model_dir = (
        "path to directory where model files are located. "
        "Model files are: {}".format(", ".join(utils.MODEL_COMPONENTS.values())),
        "option", "m"),
    output = (
        "output predictions to stdout (tsv format) as well",
        "flag", "o"),
    append = (
        "output test dataset with predicted tags appended to corresponding lines"
        " (tsv format). This mode implies --output.",
        "flag", "a"),
    test_file = (
        "path to test dataset (tsv format)",
        "positional")
)

def evaluate(model_dir,
             output=False,
             append=False,
             *test_file):
    """
    Evaluate given model on given dataset, computing the following metrics:
      * f1 score
      * precision
      * recall
    The metrics will be computed both across all predictions as a whole and
    for each entity type (e.g. PER, ORG, etc) individually. The latter will
    be shown twice -- in machine-readable and human-friendly formats.\
    """

    model_dir = model_dir or '.'
    output = output or append

    # Check paths we were given
    for fpath in test_file:
        if not os.path.isfile(fpath):
            print("Test file not found: {}".format(fpath), file=sys.stderr)
            sys.exit(14)

    x, y_true = [], []
    for fpath in test_file:
        bulk = utils.load_data_and_labels(fpath)
        x.extend(bulk[0])
        y_true.extend(bulk[1])

    model = common.load_model_from_directory(model_dir)
    y_pred = model.predict(x)

    scores           = common.collect_scores(y_true, y_pred)
    scores_ent_types = collect_scores_per_entity_type(y_true, y_pred)

    # output predictions
    if append:
        common.print_data_and_labels(x, y_true, y_pred)
    elif output:
        common.print_data_and_labels(x, y_pred)

    # output statistics
    print("=== Statistics ===")

    # overall scores
    fmt = "{:21}: {}"
    print(fmt.format("Dataset size", len(y_true)))
    for metric_name in common.REQUIRED_METRICS.keys():
        print(fmt.format(metric_name, scores.get(metric_name, 0)))

    # per-tag scores (machine-readable)
    print("")
    fmt = "{:16}[{}]: {}"
    for ent_type,_scores in scores_ent_types.items():
        for metric_name in common.REQUIRED_METRICS.keys():
            print(fmt.format(metric_name, ent_type, _scores.get(metric_name, 0)))

    # the same as the above but more human-friendly
    print("")
    print(seqeval.metrics.classification_report(y_true, y_pred))
    print("=== End of Statistics ===")

    pass

def collect_scores_per_entity_type(y_true, y_pred):
    """
    Collect scores for each entity type individually.
    Entity types are for example PER, LOC, etc as opposed to tags B-PER, I-PER,
    B-LOC, I-LOC and so on.

    Return:
      A dictionary of the form:
        {
           entity_type_1 : { metric_1 : value, metric_2 : value, ... },
           entity_type_2 : { metric_1 : value, metric_2 : value, ... }
           ...
        }
    """
    ent_types = set()
    for sent in y_true: ent_types.update(set(sent))
    for sent in y_pred: ent_types.update(set(sent))
    ent_types = set([t[2:] for t in ent_types if re.match('[BIES]-', t)])

    scores = {}
    for ent_type in sorted(ent_types):
        _y_true = keep_entity_type(y_true, ent_type, 'O')
        _y_pred = keep_entity_type(y_pred, ent_type, 'O')
        scores[ent_type] = common.collect_scores(_y_true, _y_pred)

    return scores

def keep_entity_type(predictions, ent_type, repl):
    """
    In given <predictions>, keep only tags that correspond to the given
    entity type <ent_type> rewriting all other tags to <repl>.

    For example, if <ent_type> is PER, then B-PER, I-PER will be kept and all
    other tags will be rewritten to <repl>.
    """
    changed = [
        tag if re.match(f'[BIES]-{ent_type}$', tag) else repl
        for sent in predictions
        for tag in sent
    ]
    return changed
