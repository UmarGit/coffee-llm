# coffee-llm: one entry point for the whole pipeline.
#
#   make help            what you can run
#   make test            unit tests (no GPU, no dataset needed)
#   make check           data and environment gates before any training
#   make paper           rebuild figures, tables and paper.pdf from runs/
#
# Training runs on the GPU server, not here. See `make help-server`.
.DEFAULT_GOAL := help
SHELL := /bin/bash

UV      := uv run
BENCH   := $(UV) --extra pizza python -m bench
MODEL   ?= qwen
TASK    ?= pizza
PY      := $(UV) --extra pizza python
GPUS    ?= 3
# Training targets must see exactly one GPU: with several visible the Trainer
# wraps the model in DataParallel and splits the fixed batch. scripts/run_all.sh
# pins one per shard; a bare `make smoke` has to pin one too.
GPU     ?= 0

.PHONY: help help-server setup test test-all lint check prepare paper clean clean-runs \
        push fetch smoke sweep grid report status models

help:
	@echo "local"
	@echo "  make setup         install the local environment (no GPU packages)"
	@echo "  make test          fast unit tests"
	@echo "  make test-all      including the full-pool data gates"
	@echo "  make check         round-trip, overlap, cap and environment gates"
	@echo "  make paper         rebuild figures, tables and paper.pdf from runs/"
	@echo "  make status        what is in runs/ right now"
	@echo "  make models        the model and task registries"
	@echo "  make clean         remove build artifacts (keeps runs/)"
	@echo ""
	@echo "transfer"
	@echo "  make push          send code to the GPU server"
	@echo "  make fetch         pull runs/ back (adapters skipped)"
	@echo ""
	@echo "run `make help-server` for the server side"

help-server:
	@echo "on the GPU server, the whole pipeline for one model and task:"
	@echo "  ./scripts/tmux.sh all qwen pizza $(GPUS)    every stage, one session"
	@echo "  ./scripts/tmux.sh all lfm2 pizza $(GPUS)"
	@echo "  ./scripts/tmux.sh all qwen coffee $(GPUS)"
	@echo ""
	@echo "  ./scripts/tmux.sh status                  what is alive"
	@echo "  ./scripts/tmux.sh stop                    kill everything"
	@echo ""
	@echo "one stage at a time, if you need to:"
	@echo "  make prepare                  fetch data, compute token caps"
	@echo "  make check                    gates; stop here if it fails"
	@echo "  make smoke                    tiny end-to-end run, under 10 minutes"
	@echo "  ./scripts/tmux.sh sweep $(GPUS)      learning rate sweep, sharded"
	@echo "  ./scripts/tmux.sh select         pick learning rates once shards finish"
	@echo "  ./scripts/tmux.sh grid $(GPUS)       the main grid, sharded"
	@echo "  make report                   build tables and figures"
	@echo ""
	@echo "every target takes MODEL= and TASK=, e.g. make check MODEL=lfm2"

setup:
	uv sync --extra pizza --extra dev

test:
	$(UV) --extra dev --extra pizza pytest -q

test-all:
	$(UV) --extra dev --extra pizza pytest -q -m ""

check:
	$(BENCH) --model $(MODEL) --task $(TASK) check

prepare:
	$(BENCH) --model $(MODEL) --task $(TASK) prepare

smoke:
	CUDA_VISIBLE_DEVICES=$(GPU) $(BENCH) --model $(MODEL) --task $(TASK) smoke

report:
	$(BENCH) --model $(MODEL) --task $(TASK) report

models:
	@$(UV) --extra pizza python -c "from bench import models, tasks; \
	  [print(f'model  {m.name:6} {m.hf_id:24} vocab {m.vocab_size}') for m in models.MODELS.values()]; \
	  [print(f'task   {t:7} arms {len(tasks.get(t).arms)}  sizes {tasks.get(t).default_sizes}') for t in tasks.NAMES]"

status:
	@echo "model/task:  $(MODEL)/$(TASK)"
	@echo "runs/:       $$(find runs/$(MODEL)/$(TASK) -maxdepth 1 -mindepth 1 -type d -not -name logs 2>/dev/null | wc -l | tr -d ' ') directories"
	@echo "complete:    $$(grep -l '\"complete\": true' runs/$(MODEL)/$(TASK)/*/metrics.json 2>/dev/null | wc -l | tr -d ' ')"
	@echo "learning rates: $$(cat runs/$(MODEL)/$(TASK)/lrs.json 2>/dev/null | tr -d '\n ' || echo 'not selected yet')"
	@echo "disk:        $$(du -sh runs 2>/dev/null | cut -f1)"

paper:
	$(MAKE) -C paper all

push:
	./scripts/push_code.sh

fetch:
	./scripts/fetch_runs.sh

clean:
	$(MAKE) -C paper clean
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name .DS_Store -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	@echo "runs/, data/ and the paper's generated numbers left alone"

clean-runs:
	@echo "This deletes every training result in runs/. Ctrl-C to abort."
	@read -r -p "type 'yes' to continue: " a && [ "$$a" = yes ]
	rm -rf runs
