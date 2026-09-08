import sys
from typing import Any, Optional
import json

from . import parse_args, FuncDef

if __name__ == "__main__":
	args = parse_args()

	with open(args.functions_definition, "r") as file:
		func_def = json.load(file)
	with open(args.input, "r") as file:
		prompts = json.load(file)

	funcs: list[FuncDef] = []
	for func in func_def:
		curr = FuncDef.model_validate(func)
		funcs.append(curr)
		print(curr)
	for prompt in prompts:
		print(prompt)

	

		
