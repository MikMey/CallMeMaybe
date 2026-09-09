from typing import Any

from pydantic import BaseModel, Field, model_validator

TYPES = {"string": "str", "number": "float"}

class	FuncDef(BaseModel):

	name: str = Field(..., max_length=100)
	description: str = Field(..., max_length=300)
	params: dict[str, str] = Field(...)
	returns: Any = Field(default=None)

	@model_validator(mode='before')
	@classmethod
	def set_params(cls, data: dict[str, str | dict]):
		data["params"] = {}
		for name, pack in data["parameters"].items():
			data["params"][name] = TYPES[pack["type"]]
		return data