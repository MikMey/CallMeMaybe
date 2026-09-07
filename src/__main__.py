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



if __name__ == "__main__":
	args = MaybeArgs()
	print(args.functions_definition)
	print(args.output)
	print(args.input)