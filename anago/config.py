"""
API for managing configuration: reading from/writing to a file
"""

import os
import re
import sys
import time

import configparser
from collections import OrderedDict

class Config(object):
    """
    Class that holds configuration
    """

    allitems = OrderedDict({
        "dataset" : [
            "Paths to files used in training/testing. Typically overriden on the command line.",
            OrderedDict({
                "data_dir" : (
                    "(optional) path to directory with other data files.",
                    ".", "../path/to/my/datadir"),
                "train_data" : (
                    "path to a training dataset (one file only).",
                    None, "%(data_dir)s/train.txt", 'strlist'),
                "valid_data" : (
                    "path to validation dataset. for training: keep this parameter"
                    " in config file but leave the value empty is you dont want to use"
                    " validation dataset when training.",
                    None, "%(data_dir)s/valid.txt", 'strlist'),
                "test_data" : (
                    "path to test dataset.",
                    None, "%(data_dir)s/test.txt", 'strlist')
            })],

        "model files" : [
            "files that constitute one model.",
            OrderedDict({
                "model_dir" : (
                    "(optional)"
                    " path to the directory where the model files are stored/will be saved.",
                    ".", "/path/to/modeldir/"),
                "weights_file" : (
                    "path to file with weights.",
                    "%(model_dir)s/weights.h5"),
                "params_file" : (
                    "path to file with model parameters.",
                    "%(model_dir)s/params.json"),
                "preprocessor_file" : (
                    "path to file with preprocessor data.",
                    "%(model_dir)s/preprocessor.pkl")
            })],

        "training" : [
            "parameters that control training of the model",
            OrderedDict({
                "loss" : (
                    "loss function to use.",
                    "categorical_crossentropy", None,
                    # TODO: add choices from keras
                ),
                "optimizer" : (
                    "optimizer to use.",
                    "adam", None, None,
                    ['adadelta', 'adagrad', 'adam', 'adamax', 'nadam', 'rmsprop',
                     'sgd', 'tfoptimizer']
                ),
                "max_epoch" : (
                    "number of training epochs.",
                    15, None, 'int'),
                "batch_size" : (
                    "batch size (number of records in one batch).",
                    32, None, 'int'),
                "checkpoint_path" : (
                    "where to save checkpoints (if empty, checkpoint will not be saved).",
                    None, "/path/to/somewhere"),
                "log_dir" : (
                    "directory for saving log files (if empty, nothing will be logged).",
                    None, "/path/to/where/we/store/our/logs"),
                "early_stopping" : (
                    "whether to apply early stopping.",
                    True, 'yes', 'bool', [True,False]),
                "verbose" : (
                    "verbosity level.",
                    1, None, 'int', [0,1,2])
            })],

        "model" : [
            "model parameters",
            OrderedDict({
                "char_emb_size" : (
                    "character embeddings size.",
                    25, None, 'int'),
                "word_emb_size" : (
                    "word embeddings size.",
                    100, None, 'int'),
                "char_lstm_units" : (
                    "number of LSTM units in character embeddings.",
                    25, None, 'int'),
                "word_lstm_units" : (
                    "number of LSTM units.",
                    100, None, 'int'),
                "dropout" : (
                    "dropout value",
                    0.5, None, 'float'),
                "use_char_feature" : (
                    "whether to use or not character features.",
                    "yes", None, 'bool', [True, False]),
                "use_crf" : (
                    "whether to use or not CRF layer.",
                    "yes", None, 'bool', [True, False]),
                "fc_dim" : (
                    "the number of units in Dense layer before CRF layer.",
                    100, None, 'int'),
                "use_glove_emb" : (
                    "use pretrained GloVe embeddings (path to the file should be configured"
                    " elsewere. Plus word_emb_size may need to be set correspondingly.)",
                    "no", None, 'bool')
            })],

        "glove" : [
            "GloVe embeddings",
            OrderedDict({
                "path" : (
                    "path to the file with GloVe embeddings",
                    None, "/path/to/glove.6B.100d.txt")
            })
        ]
    })

    type_converters = {
        # supported by configparser.ConfigParser
        'int'     : 'getint',
        'float'   : 'getfloat',
        'bool'    : 'getboolean',
        'strlist' : 'getstrlist'
    }

    @classmethod
    def print_default(cls, commented=False):
        now = time.strftime("%a, %d %b %Y %H:%M:%S")
        print("# Config generated: {}\n".format(now))

        for secname,params in cls.allitems.items():
            print("[{}]".format(secname))
            if commented:
                print("# in this SECTION: {}".format(params[0]))

            for pname,pinfos in params[1].items():
                if commented:
                    msg = "# param '{}': {}".format(pname, pinfos[0])
                    if len(pinfos) > 3:
                        msg += " Must be of type {}.".format(pinfos[3])
                    if len(pinfos) > 4:
                        msg += " Valid values include: {}".format(
                            ", ".join([str(v) for v in pinfos[4]]))
                    print(msg) # TODO: split to be max 80 chars long

                # select a value for this parameter we will show
                value = None
                if len(pinfos) > 2 and pinfos[2] is not None:
                    value = pinfos[2] # default value
                elif len(pinfos) > 1 and pinfos[1] is not None:
                    value = pinfos[1] # value to show in description

                if value is not None:
                    print("{} : {}".format(pname, value))
                else:
                    print("{} :".format(pname))

            print("")

    def __init__(self, fpath=None):
        if fpath is not None:
            self._load_from_file(fpath)

    def get(self, *args):
        """
        Retrieve value of given parameter.

        Args:
            *args: is a list of either one ot two values (strings).
                If there are two values, then the 1st one is section name and
                the 2nd one is parameter name.
                If there is only one value, it is parameter name.

        Returns:
            the value of the parameter if the parameter was found or None otherwise.
            The value will be a list if there is more than one value.
            Exits with error if paramater is ambiguous and exists in more than one
            section.

        Examples:
            >>> get('training', 'optimizer')
            >>> get('optimizer')
        """
        val = None

        if len(args) == 2:
            val = self._parse_parameter(self.parser, args[0], args[1])

        elif len(args) == 1:
            p = args[0].lower()
            sections = [s for s in self.parser.sections() if self.parser.has_option(s, p)]
            if len(sections) == 1:
                val = self._parse_parameter(self.parser, sections[0], p)
            elif len(sections) > 1:
                msg = "Ambiguous parameter '{}', it exists in several sections: {}"
                print(msg.format(p, sections), file=sys.stderr)
                exit(23)

        if isinstance(val, list) and len(val) == 1:
            val = val[0]

        return val

    def set(self, *args):
        """
        Set value of given parameter.

        Args:
            *args: is a list of either 2 or 3 items, such that:
                #1 (string): [OPTIONAL] section name
                #2 (string): parameter name
                #3         : new value to be set. The value can be a list if
                             the parameter is allowed to be a list. If needed,
                             the value can reference another variable using
                             interpolation syntax: %(other_var)s .

        Returns:
            Old value.
            An error will be thrown if something is wrong.

        Examples:
            >>> set('training', 'loss', 'mse')
            >>> set('loss', 'mse')
          The following sets value of a parameter that can be list. Plus, it
          uses interpolation to reference another parameter.
            >>> set('train_data', ['file1.txt', '%(data_dir)s/file2.txt'])

        """

        if len(args) == 2:
            # find section where the parameter exists
            s, p, val = None, args[0], args[1]
            sections = [ sname for sname,sparams in __class__.allitems.items()
                         if p.lower() in sparams[1]]

            if len(sections) == 1:
                s = sections[0]
            elif len(sections) > 1:
                msg = "Ambiguous parameter '{}', it exists in several sections: {}"
                print(msg.format(p, sections), file=sys.stderr)
                exit(23)
            else:
                msg = "Parameter not found in any section: {}".format(p)
                print(msg, file=sys.stderr)
                exit(24)

        elif len(args) == 3:
            s, p, val = args[0:3]

        else:
            myname = sys._getframe().f_code.co_name
            msg = "{}() takes 2 or 3 arguments, {} was/were given".format(myname, len(args))
            raise TypeError(msg)

        # add section if necessary and if it is valid
        if not self.parser.has_section(s):
            if self._is_allowed(s):
                self.parser.add_section(s)
            else:
                msg = "Error: section not allowed: {}".format(s)
                print(msg, file=sys.stderr)
                exit(32)

        # add parameter if necessary and if it is a valid parameter within
        # given section.
        if self._is_allowed(s, p):
            self.parser.set(s, p, '')
        else:
            msg = "Error: Parameter not allowed: {}".format(p)
            print(msg, file=sys.stderr)
            exit(33)

        # transform value to string, because ConfigParser wants strings only
        if isinstance(val, list):
            if not self._param_can_be_list(s, p):
                msg = "Parameter '{}/{}' can not be a list".format(s, p)
                print(msg, file=sys.stderr)
                exit(30)
            val = ", ".join(val)
        else:
            val = str(val)

        # finally, set new value
        oldval = self._parse_parameter(self.parser, s, p)
        self.parser.set(s, p, val)

        return oldval

    def write(self, file=sys.stdout):
        """
        Print configuration to given file. Target file can be given as a file-like
        object ready to writing, or as as string. In the later case, the file will
        be created, silently overwriting existing file with the same name.

        Returns:
            True always
        """
        if isinstance(file, str):
            with open(file, "w+") as fd:
                self.parser.write(fd)
        else:
            self.parser.write(file)

        return True

    def _is_allowed(self, sectionname, paramname=None):
        """
        Check if given section and optionally param in this section is/are
        legal in the configuration.

        Return:
            True or False
        """

        sname = sectionname.lower()
        ok = (sname in __class__.allitems)

        if ok and paramname is not None:
            pname = paramname.lower()
            ok = (pname in __class__.allitems[sname][1])

        return ok

    def _param_can_be_list(self, sectionname, paramname):
        """
        Check if given parameter in given section is allowed to be a list.

        Returns:
            True or False
        """

        ok = False
        sname = sectionname.lower()
        valid_params = __class__.allitems[sname][1]

        if paramname in valid_params:
            expected_type = valid_params[paramname][3] \
                            if len(valid_params[paramname]) > 3 else None
            if expected_type == 'strlist':
                ok = True

        return ok

    def _create_config_parser(self):
        parser = configparser.ConfigParser(
            allow_no_value = True,
            empty_lines_in_values = True,
            comment_prefixes = ('#'),
            inline_comment_prefixes = ('#'),
            # interpolation = None,
            converters = { 'strlist' : self._get_string_list }
        )

        # ignore extra whitespace around section name (inside [])
        parser.SECTCRE = re.compile(r"\[ *(?P<header>[^]]+?) *\]")

        # add extra boolean values
        parser.BOOLEAN_STATES.update({
            'enabled' : True, 'disabled' : False
        })

        return parser

    def _get_string_list(self, s):
        """
        Recognize a comma-separated list (of strings) in given string.
        Depending on how config parser is configures, the list can span over
        multiple lines.

        Example: item1, item2,
                 item3 , item4
        """
        lst = [fname.rstrip(', ') for fname in re.compile('\s*,\s+').split(s)]
        return lst

    def _load_from_file(self, fpath):
        self.filepath = None
        if os.path.isfile(fpath):
            self.filepath = fpath
            self.parser = self._create_config_parser()

            self.parser.read(fpath)
            self._parse_configuration(self.parser)

        else:
            print("Configuration file not found: {}".format(fpath), file=sys.stderr)
            exit(20)

    def _parse_configuration(self, cfg):
        """
        Parse and validate configuration:
        - check that all section names and parameter names are valid
        - user config is allowed to contain less sections
        - perform type conversion
        """

        ok = True
        for sectname in cfg.sections():
            _sname = sectname.lower()
            if _sname in __class__.allitems.keys():
                for pname in cfg.options(sectname):
                    ok = self._parse_parameter(cfg, sectname, pname,
                                               validate=True) and ok
            else:
                ok = False
                msg = "Invalid section [{}] in configuration file: '{}'"
                print(msg.format(sectname, self.filepath), file=sys.stderr)

        if not ok:
            exit(21)

        return ok

    def _parse_parameter(self, cfg, sectname, pname, validate=False):
        """
        Retrieve and validate given parameter and its value.

        Returns:
            True or False in validation mode;
            otherwise parameter value of correct type if type conversion is
            applicable.

        TODO: test this method
        """

        ok = True
        pval = None

        _sectname = sectname.lower()
        _valid_params = __class__.allitems[_sectname][1]

        if pname in _valid_params:
            pval = cfg.get(sectname, pname)
            pval = None if len(pval) == 0 else pval

            if pval is not None:

                expected_type = _valid_params[pname][3] \
                                if len(_valid_params[pname]) > 3 else None

                if expected_type is not None:
                    converter = __class__.type_converters[expected_type]
                    pval = getattr(cfg[sectname], converter)(pname)

                choices = _valid_params[pname][4] \
                          if len(_valid_params[pname]) > 4 else []

                if len(choices) > 0 and pval not in choices:
                    ok = False
                    msg = "Error in file {} in section '{}': invalid value of parameter" \
                          " '{}': must be {} but got '{}'"
                    msg = msg.format(self.filepath, sectname, pname, choices, pval)
                    print(msg, file=sys.stderr)

        else:
            ok = False
            msg = "Illegal parameter '{}' in section [{}] in configuration file '{}'"
            print(msg.format(pname, sectname, self.filepath), file=sys.stderr)

        return ok if validate else pval
