PY ?= .venv/bin/python
UV ?= uv
PNPM ?= pnpm

# Unsloth caps trl/transformers below monorl's versions, so it gets its own venv (third_party/pins.toml).
UNSLOTH_PIN := $(shell sed -n '/^\[unsloth\]/,/^\[/s/^version = "\(.*\)"/\1/p' third_party/pins.toml)

.PHONY: help install sync install-unsloth lock lock-upgrade test test-fast web serve envs lint vendor-list

help:
	@echo "install    create .venv with exactly the packages of requirements.lock, and monorl itself (editable)"
	@echo "sync       make an existing .venv match requirements.lock again (after a pull that changed it)"
	@echo "lock       rewrite requirements.lock after a change of pyproject.toml or third_party/pins.toml"
	@echo "lock-upgrade  rewrite requirements.lock with the newest versions that are allowed"
	@echo "install-unsloth  create .venv-unsloth (CUDA box) with monorl + unsloth==$(UNSLOTH_PIN) for trainer.backend=unsloth"
	@echo "test       run the full test suite"
	@echo "test-fast  run tests except 'slow' and 'network'"
	@echo "web        build the web UI (web/dist, served by 'monorl serve')"
	@echo "serve      start the web UI + API on http://127.0.0.1:8000"
	@echo "envs       build the Go/C++ example MEP/1 environments into envs/bin"
	@echo "lint       ruff check + format check"

install:
	$(UV) venv .venv --python 3.12
	$(MAKE) sync

# The environment CI tests: every package at the version and with the hashes of requirements.lock,
# from wheels only, torch from PyTorch's CPU builds (what the lock holds; scripts/lock.py). monorl
# itself is built with the locked build backend, and `pip check` fails when pyproject.toml asks for
# something the lock does not have.
sync:
	$(UV) pip sync --python $(PY) --torch-backend cpu --require-hashes --no-build requirements.lock
	$(UV) pip install --python $(PY) --no-deps --build-constraint requirements.lock -e .
	$(UV) pip check --python $(PY)

lock:
	$(PY) scripts/lock.py

lock-upgrade:
	$(PY) scripts/lock.py --upgrade

install-unsloth:
	$(UV) venv .venv-unsloth --python 3.12
	$(UV) pip install --python .venv-unsloth/bin/python -e ".[unsloth,dev]" "unsloth==$(UNSLOTH_PIN)"

test:
	$(PY) -m pytest tests -q

test-fast:
	$(PY) -m pytest tests -q -m 'not slow and not network'

web:
	cd web && $(PNPM) install --frozen-lockfile && $(PNPM) build

serve:
	$(PY) -m monorl.cli serve

envs:
	$(MAKE) -C envs all

lint:
	$(PY) -m ruff check src tests scripts examples
	$(PY) -m ruff format --check src tests scripts examples

vendor-list:
	$(PY) scripts/vendor.py --list
