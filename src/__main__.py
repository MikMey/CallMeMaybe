import sys
from typing import Any

from pydantic import ValidationError

from . import parse_args, FuncDef, FeedbackLoop, file_parse, create_out


def check_model(func_def: Any) -> list[FuncDef]:
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
    return funcs


def main() -> None:
    args = parse_args()
    func_def, raw_prompts = file_parse(args)
    funcs = check_model(func_def)

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

        create_out(args, out)


if __name__ == "__main__":
    main()
