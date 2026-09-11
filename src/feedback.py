from typing import Any, Callable, Optional
import sys

import numpy as np
import json
import re

from llm_sdk import Small_LLM_Model  # type: ignore
from .models import FuncDef

MSG = ("Your job is function calling. "
       "You will be given a prompt and must "
       "choose from one of the functions "
       "and declare their arguments. You need "
       "to return in the following Format: "
       "[function_name],{[Argument]:[Value]}")

post_name_pattern = re.compile(
    r"^],{"
)

arg_pattern = re.compile(
    r"(?P<arg>\w+)'?:\s?'?(?P<val>[^},]+)"
)


class FeedbackLoop():
    _model: Small_LLM_Model = None
    _vocab: dict[str, int] = {}
    _dec_vocab: list[str] = []
    _funcdefs: list[FuncDef] | Any = []

    def __init__(self, prompt: Any = None,
                 funcdefs: Optional[list[FuncDef]] = None,
                 model: Optional[str] = "Qwen/Qwen3-0.6B"):
        self.prompt = prompt
        self.func_name: dict[str, str] = {}
        self.args: dict[str, dict[str, Any]] = {}
        if FeedbackLoop._model is None:
            FeedbackLoop._model = Small_LLM_Model(model_name=model)
            vocab_file = FeedbackLoop._model.get_path_to_vocab_file()
            with open(vocab_file, 'r') as file:
                FeedbackLoop._vocab = json.load(file)
            FeedbackLoop._dec_vocab = [""] * len(FeedbackLoop._vocab)
            for key, value in FeedbackLoop._vocab.items():
                FeedbackLoop._dec_vocab[value] = key
            FeedbackLoop._funcdefs = funcdefs

    class _CheckerMachine():

        def __init__(self, prompt: str, funcdefs: list[FuncDef] | Any,
                     get_token: Callable):
            self.prompt = prompt
            self.funcdefs = funcdefs
            self.get_token = get_token
            self.func: FuncDef | Any = None
            self.params: list[tuple[str]] = []

        def loop_class(self, func: Callable) -> None:
            # print("class loop")
            top_k = 10
            res = ""
            while True:
                # print(f"res: {res}")
                options = self.get_token(self.prompt + res, top_k)
                # print(f"prompt: {self.prompt}\noptions: {options[0]}")
                for option in options[0]:
                    to_check: str = res + option
                    matches: list = func(to_check)
                    if len(matches) > 1:
                        res = to_check
                        break

                if len(matches) == 1:
                    top_k += 5
                    continue
                if len(matches) == 2 and matches[-1] is True:
                    # print("exit loop class")
                    self.prompt += res
                    return
                top_k = 10

        def loop_str(self, var: str | Any) -> None:
            """Match given argument to returned token."""
            # print("str loop")
            res = ""
            top_k = 10
            while True:
                matches: bool = False
                options = self.get_token(self.prompt + res, top_k)
                # print(f"prompt: {self.prompt}\noptions: {options[0]}")
                for option in options[0]:
                    to_check: str = res + option
                    if var.startswith(to_check) or to_check.startswith(var):
                        matches = True
                        res = to_check
                        break

                if not matches:
                    top_k += 5
                    continue
                if to_check.startswith(var):
                    self.prompt += res
                    # print(f"prompt: {self.prompt}")
                    # print(f"res: {res}")
                    return
                top_k = 10

        def match_name(self, to_check: str) -> list[Any | bool]:
            func_matches: list[FuncDef | Any] = [
                func for func in self.funcdefs
                if func.name.startswith(to_check)
                or to_check.startswith(func.name)
            ]
            # if func_matches:
            # 	for func in func_matches:
            #         print (f"name: {func.name}")
            func_matches.append(False)
            if (len(func_matches) == 2 and
                    to_check.startswith(func_matches[0].name)):
                self.func = func_matches[0]
                func_matches[-1] = True
            return func_matches

        def match_args(self, to_check: str | Any) -> list[Any | bool]:
            matches: list[list | Any] = [[]]
            to_check = to_check.split(",")
            for val in to_check:
                val = val.strip()
                val = re.sub("'", "", val)
            for arg, desc in self.func.params.items():
                if (
                        any(val.startswith(arg) for val in to_check) or
                        any(arg.startswith(val) for val in to_check)
                ):
                    matches[0].append(arg)
            matches.append(False)
            matched: list[tuple[str]] = []
            to_check = ",".join([thing for thing in to_check])
            matched = (arg_pattern.findall(to_check))
            # print(matched)
            if (matched and len(matched) == len(self.func.params) and
                    to_check.strip().endswith('}')):
                # print(
                #     f"len matched: {len(matched)};"
                #     "len params: {len(self.func.params)}"
                # )
                matches[-1] = True
                self.params = matched
            return matches

        def on_enter_name(self) -> None:
            # print("enter name")
            self.loop_class(self.match_name)
            # print("test")
            self.on_enter_spacer()

        def on_enter_spacer(self) -> None:
            # print("spacer")
            self.loop_str(str("],{"))
            self.on_enter_args()

        def on_enter_args(self) -> None:
            # print("enter args", flush=True)
            self.loop_class(self.match_args)
            # print("test")
            return

    def get_answer(self) -> None:
        machine = self._CheckerMachine(self._prep_prompt(),
                                       self._funcdefs, self._get_token)
        machine.on_enter_name()
        # print("finished")
        self.answer: dict = {}
        self.answer["prompt"] = self.prompt
        self.answer["name"] = machine.func.name
        self.answer["parameters"] = {}
        for i, arg in enumerate(machine.params):
            match_type = list(machine.func.params.values())[i]
            # print(f"match: {match_type}\narg: {arg}\ni: {i}")
            try:
                match match_type:
                    case "float":
                        self.answer["parameters"][arg[0]] = float(arg[1])  # type: ignore  # noqa: E501
                    case "str":
                        val = arg[1]  # type: ignore
                        if (arg[1].count("'") % 2):  # type: ignore
                            val = arg[1].strip("'")  # type: ignore
                        self.answer["parameters"][arg[0]] = val
                    case "int":
                        self.answer["parameters"][arg[0]] = int(arg[1])  # type: ignore  # noqa: E501
                    case "bool":
                        self.answer["parameters"][arg[0]] = bool(arg[1])  # type: ignore  # noqa: E501
                    case "list":
                        self.answer["parameters"][arg[0]] = list(arg[1])  # type: ignore  # noqa: E501
                    case "dict":
                        self.answer["parameters"][arg[0]] = dict(arg[1])  # type: ignore  # noqa: E501
            except (ValueError, KeyError) as err:
                msg = f"Expected and generated types don't match:\n{err}"
                sys.exit(msg)
            except Exception as err:
                sys.exit(f"Error in argument type matching:\n{err}")
        return

    def _prep_prompt(self) -> str:
        text: str = MSG
        try:
            for func in self._funcdefs:
                # print(func)
                text += (f"\n\nName: {func.name}"
                         f"\nDescription: {func.description}"
                         "\nArgument:")
                for name, types in func.params.items():
                    text += f" {name}:{types}, "
                text += f"\nReturn: {func.returns}"
            text += (f"\n\nYour Prompt is: {self.prompt}\n"
                     f"Answer: [")
        except AttributeError as err:
            sys.exit(f"Attribute Error:\n{err}")
        return text

    def _get_token(self, text: str, top_k: int = 10) -> list[list[str | float]]:  # noqa: E501
        """Prompt to receive next token.

        Args
        ------
        top_k: int = 10
                top elements to be returned
        """
        token_ids = self.encode(text)

        logits = self._model.get_logits_from_input_ids(token_ids)

        scores = np.array(logits)
        new_token_ids = np.argsort(scores, descending=True)[:top_k]
        scores = scores[new_token_ids]

        tokens: Any = self.decode(new_token_ids)

        if len(scores) > 1:
            norm_scores = ((scores - np.min(scores)) /
                           (np.max(scores) - np.min(scores)) * 100)
            scores = norm_scores

        # for i in range(top_k):
        #     print(f"{tokens[i]!r}\t{round(float(scores[i]), 3)}%")
        return [tokens, scores]

    @staticmethod
    def encode(text: str, known: Optional[bool] = True) -> list[int]:
        fucking_keys_mypy_is_a_fucking_bitchhhh_wtf_am_i_supposed_to_do: Any = {" ": "Ġ", "\n": "Ċ"}  # noqa: E501
        text = text.translate(str.maketrans(fucking_keys_mypy_is_a_fucking_bitchhhh_wtf_am_i_supposed_to_do))  # noqa: E501
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
    def decode(token_ids: Any) -> list[str]:
        tokens = []
        for token_id in token_ids:
            token = FeedbackLoop._dec_vocab[int(token_id)]
            tokens.append(
                token.translate(str.maketrans({"Ġ": " ", "Ċ": "\n"}))  # type: ignore  # noqa: E501
            )
        return tokens
