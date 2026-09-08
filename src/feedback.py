from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model
from models import FuncDef

class	FeedbackLoop():
	prompt: str
	name: str
	param: dict[str, Any]

	def __init__(self, prompt: str, funcdefs: list[FuncDef]):
		self.prompt = prompt
		self.funcdefs = funcdefs

	def get_token(self):
		new_llm = Small_LLM_Model()
    
		ids_tensor = new_llm.encode(self.prompt)
		ids_list = ids_tensor[0].tolist()

		print("llm")
		logits = new_llm.get_logits_from_input_ids(ids_list)
		
		token = []
		score = []
	
		print("decode")
		for token_id, logit_score in enumerate(logits):
			token_text = new_llm.decode([token_id])
			token.append(token_text)
			score.append(logit_score)
		print("sort")
		token = np.array(token)
		score = np.array(score)
		order = np.argsort(score, descending=True)
		token = token[order]
		score = score[order]
		score = ((score - np.min(score)) / (np.max(score) - np.min(score)) * 100)

		for i in range(10):
			print(f"{str(token[i])!r}\t{round(score[i], 3)}%")

if __name__ == "__main__":
	newloop = FeedbackLoop("how are your?", None)
	newloop.get_token()
