# coding: utf8

import os
import sys
import plac

from ..utils import load_data_and_labels, MODEL_COMPONENTS
from ..wrapper import Sequence
from .common import load_model_from_directory

# output_as_key_value_pairs() will output the fields in this order
kv_fields = tuple(['type', 'text', 'beginOffset', 'endOffset', 'score'])

@plac.annotations(
    model_dir = (
        "path to directory where model files are located. "
        "Model files are: {}".format(", ".join(MODEL_COMPONENTS.values())),
        "option", "m"),
    input_file=("Path to input file", "positional")
)

def analyse(model_dir, *input_file):
    """
    Analyze given sample(s) of text, going through the text line per line.
    Input text must be tokenized into words.

    If input file is not given, stdin will be read.

    Results are outputted in plain text in tabular format in the hope of
    making them more readable for humans. For example:

    INPUT    Ex-president Barak Obama is speaking at the White House .
    PER      Barak Obama    1    3    1.0
    LOC      White House    7    9    1.0\
    """

    model = load_model_from_directory(model_dir)

    if len(input_file) > 0:
        for fname in input_file:
            with open(fname) as fd:
                process_stream(fd, model)
    else:
        process_stream(sys.stdin, model)

def process_stream(fd, model):
    for line in fd:
        line = line.strip()
        if len(line) > 0:
            res = model.analyze(line)
            output_as_key_value_pairs(line, res)

def output_as_key_value_pairs(srcline, res={}):
    """
    INPUT       \\tab line of text
    ENTITY_TYPE \\tab entity text
    ENTITY_TYPE \\tab entity text
    """

    print('INPUT\t{}'.format(srcline))

    for ent in res.get('entities', []):
        fields = [ str(ent[fld]) for fld in kv_fields ]
        print("\t".join(fields))

    print("")
