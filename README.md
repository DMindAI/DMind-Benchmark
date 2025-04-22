# DMind Benchmark
A comprehensive framework for evaluating large language models (LLMs) on blockchain, cryptocurrency, and Web3 knowledge across multiple domains.

## Overview

This project provides tools to benchmark AI models on their understanding of blockchain concepts through both objective (multiple-choice) and subjective (open-ended) questions. The framework covers various domains including:

- Blockchain Fundamentals
- DeFi (Decentralized Finance)
- Smart Contracts
- DAOs (Decentralized Autonomous Organizations)
- NFTs
- Security
- Tokenomics
- MEME coins
- Blockchain Infrastructure

## Features

- Test models on multiple-choice questions with single or multiple correct answers
- Evaluate models on open-ended questions requiring detailed explanations
- Support for various question types including:
  - Calculation questions
  - Code audit questions
  - Fill-in-blank questions
  - Market reasoning questions
  - Matching questions
  - Ordering questions
  - Risk analysis questions
  - Scenario analysis questions
  - Short answer questions
  - Strategy analysis questions
  - Vulnerability classification questions
- Automated scoring and evaluation
- Calculate total scores and comparative analysis across models

## Installation

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your API settings in models.yml:

   ```bash
   api_base: "your_api_base"
   # Add other configuration settings as needed
   ```

## Usage

The project includes a Makefile with commands to run different tests:

```bash
# Run objective tests
make test-objective model=gpt-4o

# Run subjective tests
make test-subjective model=gpt-4o

# Calculate total score for a specific model
make calculate-model model=gpt-4o
```

### Testing Multiple Models

You can test multiple models and compare their performance:

1. Add models to your models.yml configuration
2. Run tests for each model
3. Use the calculation tools to compare results

## Project Structure

```
├── api/
│   ├── calculate_total_score.py  # Calculate and analyze model scores
│   ├── test_objective.py         # Run objective tests (multiple choice)
│   ├── test_subjective.py        # Run subjective tests (open-ended)
│   └── question_types/           # Question types implementation
│       ├── base_question.py      # Base class for all question types
│       ├── calculation_question.py
│       ├── code_audit_question.py
│       └── ...
├── test_data/
│   ├── objective/                # Multiple choice questions in CSV format
│   │   ├── Blockchain_Fundamentals_benchmark.csv
│   │   ├── DAO2.csv
│   │   └── ...
│   └── subjective/               # Open-ended questions in JSON format
│       ├── Blockchain_Fundamentals_benchmark.json
│       ├── DAO.json
│       └── ...
├── models.yml                    # Model configuration
├── requirements.txt              # Python dependencies
└── Makefile                      # Commands for running tests
```

## Evaluation Methodology

The evaluation framework uses various techniques to assess model performance:

- For objective questions: Exact matching against correct answers
- For subjective questions: Combination of keyword analysis, structured evaluation, and third-party AI evaluation when configured

## Customization

- Add new questions by extending the CSV/JSON files in test_data/
- Implement new question types by extending the BaseQuestion class
- Configure evaluation parameters in the respective question type implementations