#!/usr/bin/env python
from argparse import ArgumentParser

from transformers_local.commands.convert import ConvertCommand
from transformers_local.commands.download import DownloadCommand
from transformers_local.commands.env import EnvironmentCommand
from transformers_local.commands.run import RunCommand
from transformers_local.commands.serving import ServeCommand
from transformers_local.commands.user import UserCommands


def main():
    parser = ArgumentParser("Transformers CLI tool", usage="transformers-cli <command> [<args>]")
    commands_parser = parser.add_subparsers(help="transformers-cli command helpers")

    # Register commands
    ConvertCommand.register_subcommand(commands_parser)
    DownloadCommand.register_subcommand(commands_parser)
    EnvironmentCommand.register_subcommand(commands_parser)
    RunCommand.register_subcommand(commands_parser)
    ServeCommand.register_subcommand(commands_parser)
    UserCommands.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
