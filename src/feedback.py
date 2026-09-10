from typing import Any, Callable, Optional
import sys
import os

import numpy as np
import json
import random
import functools
import time
import re
from statemachine import StateChart, State

from llm_sdk import Small_LLM_Model
from .models import FuncDef

def Timer(func: Callable) -> Callable:
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print(f"Time: {end-start}")
		return result
	return wrapper


MSG = "Your job is function calling. You will be given a prompt and must choose from one of the functions and declare their arguments. You need to return in the following Format: [function_name],{[Argument]:[Value]}" 

post_name_pattern = re.compile(
	r"^],{"
)

arg_pattern = re.compile(
	r"(?P<arg>\w+)'?:\s?'?(?P<val>[^},]+)"
	)

class	FeedbackLoop():
	_model: Small_LLM_Model = None
	_vocab: dict[str, int] = {}
	_dec_vocab: list[str] = []
	_funcdefs: list[FuncDef] = []


	def __init__(self, prompt: str, funcdefs: Optional[list[FuncDef]] = None, model: Optional[str] = "Qwen/Qwen3-0.6B"):
		self.prompt = prompt
		if FeedbackLoop._model is None:

			FeedbackLoop._model = Small_LLM_Model(model_name=model)
			vocab_file = FeedbackLoop._model.get_path_to_vocab_file()
			with open(vocab_file, 'r') as file:
				FeedbackLoop._vocab = json.load(file)
			FeedbackLoop._dec_vocab = [None] * len(FeedbackLoop._vocab)
			for key, value in FeedbackLoop._vocab.items():
				FeedbackLoop._dec_vocab[value] = key
			FeedbackLoop._funcdefs = funcdefs


	class _CheckerMachine(StateChart):
		name = State(initial=True)
		spacer = State()
		args = State(final=True)

		advance = (
			name.to(spacer)
			| spacer.to(args)
		)

		def __init__(self, prompt: str, funcdefs: list[FuncDef], get_token: Callable):
			self.prompt = prompt
			self.funcdefs = funcdefs
			self.get_token = get_token

		@functools.singledispatch
		def loop(self, var: str | Any):
			"""
			Match given argument to returned token
			"""
			res = ""
			top_k = 10
			while True:
				matches: bool = False
				options = self.get_token(self.prompt + res, top_k)

				for option in options[0]:
					to_check: str = self.res + option
					if var.startwith(to_check) or to_check.startswith(var):
						matches: bool = True

				if not matches:
					top_k += 5
					continue
				if to_check.startswith(var):
					self.prompt += res
					return
				top_k = 10
				res = to_check

		@loop.register(Callable)
		def _1(self, func: Callable):
			"""

			"""
			top_k = 10
			res = ""
			while True:
				options = self.get_token(self.prompt + res, top_k)
				for option in options[0]:
					to_check: str = res + option
					matches: list = func(to_check)

				if not matches:
					top_k += 5
					continue
				top_k = 10
				res = to_check
				if len(matches) == 2 and matches[-1] == True:
					self.promp += res
					return

		def match_name(self, to_check: str) -> list[Any | bool]:
			self.func_matches: list[FuncDef] = [
				func for func in self.funcdefs 
				if func.name.startswith(to_check) 
				or to_check.startswith(func.name)
				]

		def match_args(self, to_check: str) -> list[Any | bool]:
			new = to_check[len(self.func_name):]
			matched: list[tuple[str]] = arg_pattern.findall(new)

		def before_cycle(self):
			self.res: str = ""
			self.top_k = 10

		def on_enter_name(self):
			self.loop(self.match_name)
			

		def on_exit_name(self):
			self.func_name = self.func_matches[0]

		def on_enter_spacer(self):
			self.loop("],{")
									

		def on_enter_args(self):
			self.loop(self.match_args)

		def on_exit_args(self):
			pass

	def get_answer(self):

		self.args: dict[str | float] = {}
		for pair in matched:
			self.args[pair[0]] = pair[1].strip("'")
		

	def _prep_prompt(self) -> str:

		text: str = MSG
		for func in self._funcdefs:
			text += (
				f"\n\nName: {func.name}" \
				f"\nDescription: {func.description}"
				"\nArgument:"
			)
			for name, types in func.params.items():
				text += (
					f" {name}:{types}, "
				)
			text += (
				f"\nReturn: {func.returns}"
			)
		text += (
			f"\n\nYour Prompt is: {self.prompt}\n"
			f"Answer: ["
		)
		return text


	def _choose_token(self, text: str) -> str:
		pass


	def _get_token(self, text: str, top_k: int = 10) -> list[list[str | float]]:
		"""
		Prompt to receive next token

		Args
		------
		top_k: int = 10
			top elements to be returned
		"""
		token_ids = self.encode(text)

		logits = self._model.get_logits_from_input_ids(token_ids)

		scores = np.array(logits)
		token_ids = np.argsort(scores, descending=True)[:top_k]
		scores = scores[token_ids]

		tokens = self.decode(token_ids)

		if len(scores) > 1:
			scores = ((scores - np.min(scores)) / (np.max(scores) - np.min(scores)) * 100)

		# for i in range(top_k):
			# print(f"{tokens[i]!r}\t{round(float(scores[i]), 3)}%")
		return [tokens, scores]


	@staticmethod
	def encode(text: str, known: Optional[bool] = True) -> list[int]:
		text = text.translate(str.maketrans({" ": "Ġ", "\n": "Ċ"}))
		token_ids = []
		i = 0
		while i < len(text):
			seq = text[i:]
			while seq and seq not in FeedbackLoop._vocab:
				seq = seq[:-1]
			if seq not in FeedbackLoop._vocab:
				sys.exit(f"skill issue {seq}")
			token_ids.append(FeedbackLoop._vocab[seq])
			i += len(seq)
		return token_ids


	@staticmethod
	def decode(token_ids: list[int]) -> list[str]:
		tokens = []
		for token_id in token_ids:
			token = FeedbackLoop._dec_vocab[int(token_id)]
			tokens.append(
				token.translate(str.maketrans({"Ġ": " ", "Ċ": "\n"}))
				)
		return tokens


if __name__ == "__main__":
	newloop = FeedbackLoop(MSG, None)
	text = ""
	start = time.time()
	for i in range(100):
		MSG += newloop.get_token(MSG, 1)
	print(MSG)
	end = time.time()
	print(f"Time: {end-start}")

