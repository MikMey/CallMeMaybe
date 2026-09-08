import sys
from typing import Any, Optional
import json

from pydantic import BaseModel, model_validator, Field

# from llm_sdk import Small_LLM_Model
import argparse


class MaybeArgs:

	FLAGS = [
		"--functions_definition",
		"--input",
		"--output"
		]

	def __init__(self) -> None:
		args = self.Parser(self.FLAGS).args

		self.functions_definition = args.functions_definition
		self.input = args.input
		self.output = args.output

	class Parser(argparse.ArgumentParser):

		def __init__(self, flags: list[str]):
			super().__init__()

			for flag in flags:
				self.add_argument(flag, required=True, type=str)

			self.args = self.parse_args()

TYPES = {"string": str, "number": float}

class	FuncDef(BaseModel):

	name: str = Field(..., max_length=100)
	description: str = Field(..., max_length=300)
	params: dict[str, Any] = Field(...)
	returns: Any = Field(default=None)

	@model_validator(mode='before')
	@classmethod
	def set_params(cls, data: dict):
		data["params"] = {}
		for name, pack in data["parameters"].items():
			data["params"][name] = TYPES[pack["type"]]
		return data


if __name__ == "__main__":
	args = MaybeArgs()

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

		
