# Reproducing every number in the paper, from a clean checkout.
#
#   make install     python + node dependencies
#   make setup       compile the circuit and run the phase-2 ceremony
#   make test        the full suite (circuit tests skip without artifacts/)
#   make evidence    every experiment, into results/, then check the paper
#   make check-paper every number in paper/paper.tex, against results/*.json
#   make e3-repeat   how far the loss-regime consensus numbers move between runs
#   make e3-deadline whether the sub-round budget or the deadline is the limit
#
# `make evidence` needs `make setup` first, or the strict-ZK measurements have
# nothing to measure.

PY      ?= ./venv/bin/python
PYTEST  ?= $(PY) -m pytest
NODES   ?= 10
ROUNDS  ?= 5

.PHONY: help install venv setup setup-offline test test-fast test-circuit \
        lint evidence e1 e2 e3 e4 e5 sim sim-ws clean clean-artifacts

help:
	@sed -n '1,15p' Makefile

venv:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip

install: venv
	./venv/bin/pip install -r requirements.txt
	npm ci || npm install

# -- circuit ---------------------------------------------------------------

setup:
	$(PY) scripts/setup_circuit.py --features 4 --batch 8 --depth 4

# Offline build.  Phase 1 is generated locally, so the operator can forge
# proofs: fine for development, never for a claim.
setup-offline:
	$(PY) scripts/setup_circuit.py --features 4 --batch 8 --depth 4 --local-ptau

# The 6-constraint circuit from the original artifact, kept only so that
# e4_adversarial can demonstrate the attack it admits.
setup-v1:
	$(PY) scripts/setup_circuit.py --v1 --local-ptau

# -- tests -----------------------------------------------------------------

test:
	$(PYTEST) tests/ -q

test-fast:
	$(PYTEST) tests/ -q -m "not slow and not circuit"

test-circuit:
	$(PYTEST) tests/ -q -m circuit

# -- evaluation ------------------------------------------------------------

# Regenerates every results/*.json the paper quotes, then re-checks the paper
# against them.  Run it on an otherwise idle machine: the consensus medians and
# the prove/verify timings are wall-clock, and we have measured the same seed
# commit at 0.33 and at 0.71 depending only on what else the host was doing.
evidence: e1 e2 e3 e4 e5 check-paper
	@echo
	@echo "results/:"
	@ls -1 results/

e1:
	$(PY) -m experiments.e1_learning --dataset iris --nodes 5 --rounds 60

e2:
	$(PY) -m experiments.e2_proof_cost

e3:
	$(PY) -m experiments.e3_consensus --rounds 5 --sub-round-sweep

# How reproducible the loss-regime numbers are.  Run this before quoting them.
e3-repeat:
	$(PY) -m experiments.e3_repeatability

# Whether the sub-round budget or the per-height deadline is the real limit.
e3-deadline:
	$(PY) -m experiments.e3_deadline

e4:
	$(PY) -m experiments.e4_adversarial

# --strict is not optional here: the paper's audit numbers are the cost of
# re-verifying *real* Groth16 proofs offline, and without it the run produces a
# 0.003 s audit of nothing.  Dropping the flag is how results/e5_audit.json and
# the paper quietly stop describing the same experiment.
e5:
	$(PY) -m experiments.e5_audit --nodes 7 --rounds 3 --strict

# Every number in paper/paper.tex, against the JSON that produced it.
check-paper:
	$(PY) scripts/check_paper_numbers.py

# -- running it ------------------------------------------------------------

sim:
	$(PY) scripts/run_simulation.py --nodes $(NODES) --rounds $(ROUNDS)

sim-ws:
	$(PY) scripts/run_simulation.py --nodes 5 --rounds 3 --transport websocket

sim-strict:
	$(PY) scripts/run_simulation.py --nodes 4 --rounds 1 --strict --window 1.0

# -- housekeeping ----------------------------------------------------------

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

clean-artifacts:
	rm -rf artifacts/
