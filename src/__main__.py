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
	out: list[dict] = []

	FeedbackLoop(None, funcdefs=funcs)
	for prompt in prompts:
		curr_loop  = FeedbackLoop(prompt)
		loops.append(curr_loop)
		curr_loop.get_answer()
		print(f"{curr_loop.answer["prompt"]}\n{curr_loop.answer["name"]}\n{curr_loop.answer["parameters"]}\n")
		out.append(curr_loop.answer)
		# sys.exit()
	
	# print(out)
	with open("out.json", 'w') as file:
		json.dump(out, file, indent=2)


