from typing import Any, Callable, Optional
import sys

import numpy as np
import json
import random
import functools
import time

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


	def get_answer(self):

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
		)
		for _ in range(25):
			text += self._get_token(text)[0][0]
		print (text)
		


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

