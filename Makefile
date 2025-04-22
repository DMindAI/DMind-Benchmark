# Set Python interpreter
PYTHON = python

# Set directories
TEST_DIR = .
RESULTS_DIR = test_results
API_DIR = ./api

# Default target
.PHONY: all
all: test calculate

# Run all tests
.PHONY: test
test:
	@if [ "$(model)" = "" ]; then \
		echo "Error: Please specify model name, for example: make test model=gpt-4o"; \
		exit 1; \
	fi
	@echo "Starting tests..."
	@$(PYTHON) $(API_DIR)/test_objective.py --model $(model)
	@$(PYTHON) $(API_DIR)/test_subjective.py --model $(model)
	@echo "Tests completed"

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
		echo "Error: Please specify model name, for example: make calculate-model model=gpt-4o"; \
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
	@if [ "$(model)" = "" ]; then \
		echo "Error: Please specify model name, for example: make pipeline model=gpt-4o"; \
		exit 1; \
	fi

pipeline: test calculate show

# Help information
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make test              - Run all tests (requires model parameter)"
	@echo "  make calculate         - Calculate total score for all models"
	@echo "  make calculate-model   - Calculate total score for specified model (requires model parameter)"
	@echo "  make clean             - Clean test results"
	@echo "  make show              - Show test results"
	@echo "  make pipeline          - Execute complete pipeline (clean, test, calculate, show)"
	@echo "  make help              - Show this help information"
	@echo ""
	@echo "Example:"
	@echo "  make calculate-model model=gpt-4o  # Calculate total score for specified model" 