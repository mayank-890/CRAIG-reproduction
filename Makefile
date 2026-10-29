.PHONY: help verify test baseline-debug baseline-standard clean

PYTHON = .venv/bin/python

help:
	@echo "CRAIG Research Project - Available Commands:"
	@echo "  make verify             Verify environment and dependencies"
	@echo "  make test               Run unit test suite"
	@echo "  make baseline-debug     Run MNIST baseline (Full + Random) in debug mode"
	@echo "  make baseline-standard  Run full MNIST baseline experiment"
	@echo "  make clean              Remove caches and temporary files"

verify:
	$(PYTHON) scripts/verify_installation.py

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

baseline-debug:
	$(PYTHON) experiments/run_mnist.py --config configs/mnist.yaml --profile debug

baseline-standard:
	$(PYTHON) experiments/run_mnist.py --config configs/mnist.yaml --profile standard

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
