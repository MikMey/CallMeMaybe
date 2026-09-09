import sys
from typing import Any, Optional
import json

from . import parse_args, FuncDef, FeedbackLoop

if __name__ == "__main__":
	args = parse_args()

	with open(args.functions_definition, "r") as file:
		func_def = json.load(file)
	with open(args.input, "r") as file:
		raw_prompts: dict[dict[str, str]] = json.load(file)

	funcs: list[FuncDef] = []
	for func in func_def:
		funcs.append(FuncDef.model_validate(func))

	prompts = []
	for prompt in raw_prompts:
		prompts.append(prompt["prompt"])

	loops: list[FeedbackLoop] = []

	FeedbackLoop(None, funcdefs=funcs)
	for prompt in prompts:
		curr_loop  = FeedbackLoop(prompt)
		loops.append(curr_loop)
		curr_loop.get_answer()
		print("\n")
