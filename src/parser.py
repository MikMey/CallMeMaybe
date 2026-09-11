from dataclasses import dataclass
import sys
import json
from typing import Any
from pathlib import Path

import argparse


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


def file_parse(args: Args) -> list[Any]:
    try:
        with open(args.functions_definition, "r") as file:
            func_def = json.load(file)
            if not func_def:
                sys.exit("Function definitions are empty")
        with open(args.input, "r") as file:
            raw_prompts = json.load(file)
            if not raw_prompts:
                sys.exit("Prompts are empty")
    except json.decoder.JSONDecodeError as err:
        sys.exit(f"Invalid json file:\n{err}")
    except (PermissionError, OSError, FileNotFoundError,
            IsADirectoryError) as err:
        sys.exit(f"Provided file path is incorrect:\n{err}")
    except Exception as err:
        sys.exit(f"Error:\n{err}")
    return [func_def, raw_prompts]


def create_out(args: Args, out: list[dict]) -> None:
    # print(out)
    direc = Path("data/output")
    direc.mkdir(parents=True, exist_ok=True)
    try:
        with open(args.output, 'w') as file:
            json.dump(out, file, indent=2)
    except Exception as err:
        sys.exit(f"cannot write to output file:\n{err}")
