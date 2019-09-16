# coding: utf8

import plac
from ..config import Config

@plac.annotations(
    generate = ("generate new configuration", "flag", "g"),
    commented = ("comment on each parameter", "flag", "c")
)

def config(generate=False,
           commented=False):
    """
    Manage configuration file\
    """

    # for the time being, this is the only thing we can do.
    generate = True

    if generate:
        Config.print_default(commented)

    return True
