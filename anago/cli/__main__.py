import os
import sys
import plac
import inspect

from .train import train
from .evaluate import evaluate

COMMANDS = {
    "evaluate" : evaluate,
    "train"    : train
}

def main():
    """
    usage: anago [-h] command

        Command line interface to anago.
        Available commands include: {0}.

    positional arguments:
      command       one of the above commands.

    To get help on a specific command, run it with a -h or --help option, for example:

        anago {1} --help

    """

    args = sys.argv[1:]

    if len(args) == 0 or args[0] in ['-h', '--help']:
        msg = main.__doc__.format(", ".join(COMMANDS.keys()),
                                  list(COMMANDS.keys())[0])
        print(inspect.cleandoc(msg))
        sys.exit(0)

    command = args.pop(0)

    if command in COMMANDS:
        plac.call(COMMANDS[command], args)
    else:
        print("Command not recognized: {}. Try --help.".format(command),
              file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    sys.exit(main())
