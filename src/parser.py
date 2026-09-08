import argparse
from dataclasses import dataclass


@dataclass
class Args:
    functions_definition: str
    input: str
    output: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument("--functions_definition", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    return Args(
        functions_definition=args.functions_definition,
        input=args.input,
        output=args.output,
    )
