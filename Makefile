NAME := src
PYTHON ?= python3
FLAKE8 := uv run -m flake8
FLAKE8_FLAGS := --count --show-source --filename [./*.py]
MYPY := uv run mypy
MYPY_FLAGS := --warn-return-any \
			  --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
			  --check-untyped-defs
UV_VENV := uv sync
DEF ?= data/input/functions_definition.json
IN ?= data/input/function_calling_tests.json
OUT ?= data/output/function_calling_results.json
FLAGS := --functions_definition $(DEF) --input $(IN) --output $(OUT)

all: install run

install:
	$(UV_VENV)

run:
	uv run $(PYTHON) -m $(NAME) $(FLAGS)

lint:
	$(FLAKE8) $(FLAKE8_FLAGS) ./src/*
	$(MYPY) ./src/*.py $(MYPY_FLAGS)

clean:
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: all install run lint clean