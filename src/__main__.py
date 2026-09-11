import sys
import json

from pydantic import ValidationError

from . import parse_args, FuncDef, FeedbackLoop

if __name__ == "__main__":
    args = parse_args()
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

    funcs: list[FuncDef] = []
    try:
        for func in func_def:
            funcs.append(FuncDef.model_validate(func))
        for func in funcs:
            if any(func.name == comp.name
                   for comp in funcs if comp is not func):
                sys.exit(f"Duplicate function name \'{func.name}\'")
    except ValidationError as err:
        sys.exit(f"Format doesnt match requirment:\n{err}")
    except (KeyError, ValueError) as err:
        sys.exit(f"Malformated file or argument:\n{err}")
    except Exception as err:
        sys.exit(f"Error:\n{err}")

    prompts = []
    for prompt in raw_prompts:
        prompts.append(prompt["prompt"])

    loops: list[FeedbackLoop] = []
    out: list[dict] = []

    FeedbackLoop(funcdefs=funcs)
    for prompt in prompts:
        curr_loop = FeedbackLoop(prompt)
        loops.append(curr_loop)
        curr_loop.get_answer()
        print(
            f"{curr_loop.answer["prompt"]}\n"
            f"{curr_loop.answer["name"]}\n"
            f"{curr_loop.answer["parameters"]}\n"
        )
        out.append(curr_loop.answer)
        # sys.exit()

    # print(out)
    try:
        with open(args.output, 'w') as file:
            json.dump(out, file, indent=2)
    except Exception as err:
        sys.exit(f"cannot write to output file:\n{err}")
