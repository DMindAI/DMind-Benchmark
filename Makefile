# Set Python interpreter
PYTHON = python

# Set directories
TEST_DIR = .
RESULTS_DIR = test_results
API_DIR = ./api

# Default target
.PHONY: all
all: test calculate

# Run tests for specific model
.PHONY: test
test:
	@if [ "$(model)" = "" ]; then \
		echo "Error: Please specify model name, for example: make test model=claude-3-5-haiku-20241022"; \
		exit 1; \
	fi
	@echo "Starting tests for model $(model)..."
	@$(PYTHON) $(API_DIR)/test_objective.py --model $(model)
	@$(PYTHON) $(API_DIR)/test_subjective.py --model $(model)
	@echo "Tests completed"

# Run objective tests only
.PHONY: test-objective
test-objective:
	@if [ "$(model)" = "" ]; then \
		echo "Error: Please specify model name, for example: make test-objective model=claude-3-5-haiku-20241022"; \
		exit 1; \
	fi
	@echo "Starting objective tests..."
	@$(PYTHON) $(API_DIR)/test_objective.py --model $(model)
	@echo "Objective tests completed"

# Run subjective tests only
.PHONY: test-subjective
test-subjective:
	@if [ "$(model)" = "" ]; then \
		echo "Error: Please specify model name, for example: make test-subjective model=claude-3-5-haiku-20241022"; \
		exit 1; \
	fi
	@echo "Starting subjective tests..."
	@$(PYTHON) $(API_DIR)/test_subjective.py --model $(model)
	@echo "Subjective tests completed"

# Calculate total score for all models
.PHONY: calculate
calculate:
	@echo "Calculating total score for all models..."
	@$(PYTHON) $(API_DIR)/calculate_total_score.py --all
	@echo "Total score calculation completed"

# Calculate total score for specified model
.PHONY: calculate-model
calculate-model:
	@if [ "$(model)" = "" ]; then \
		echo "Error: Please specify model name, for example: make calculate-model model=claude-3-5-haiku-20241022"; \
		exit 1; \
	fi
	@echo "Calculating total score for model $(model)..."
	@$(PYTHON) $(API_DIR)/calculate_total_score.py --model $(model)
	@echo "Total score calculation completed"

# Clean test results
.PHONY: clean
clean:
	@echo "Cleaning test results..."
	@rm -rf $(RESULTS_DIR)/*
	@echo "Cleaning completed"

# Show test results
.PHONY: show
show:
	@echo "Showing test results..."
	@$(PYTHON) -c "import json; f=open('$(RESULTS_DIR)/total_score.json'); data=json.load(f); print(f'Total Score: {data[\"total_score\"]:.2f}'); print('\nDataset Scores:'); [print(f'{k}: {v[\"dataset_score\"]*100:.2f} points') for k,v in data['dataset_scores'].items()]"

# Execute complete pipeline
.PHONY: pipeline
pipeline:
	@if [ "$(model)" = "" ]; then \
		echo "Error: Please specify model name, for example: make pipeline model=claude-3-5-haiku-20241022"; \
		exit 1; \
	fi
	@echo "Executing complete pipeline for model $(model)..."
	@$(MAKE) clean
	@$(MAKE) test model=$(model)
	@$(MAKE) calculate-model model=$(model)
	@$(MAKE) show
	@echo "Pipeline execution completed"

# Help information
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make test              - Run tests for a specific model (requires model parameter)"
	@echo "  make test-objective    - Run objective tests only (requires model parameter)"
	@echo "  make test-subjective   - Run subjective tests only (requires model parameter)"
	@echo "  make calculate         - Calculate scores for all models"
	@echo "  make calculate-model   - Calculate score for a specific model (requires model parameter)"
	@echo "  make clean             - Clean all test results"
	@echo "  make show              - Show test results"
	@echo "  make pipeline          - Run the complete pipeline (clean, test, calculate, show) for a specific model"
	@echo "  make help              - Display help information"
	@echo ""
	@echo "Example:"
	@echo "  make calculate-model model=claude-3-5-haiku-20241022  # Calculate score for claude-3-5-haiku-20241022 model" 