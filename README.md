---
configs:
- config_name: objective_normal
  data_files:
  - split: Tokenomist
    path:
    - "test_data/objective/Tokenomist.csv"
  - split: Fundamentals
    path:
    - "test_data/objective/Blockchain_Fundamentals_benchmark.csv"
  - split: DAO
    path:
    - "test_data/objective/DAO2.csv"
  - split: Defi
    path:
    - "test_data/objective/Defi_benchmark.csv"
  - split: MEME
    path:
    - "test_data/objective/MEME_Benchmark_modified.csv"
  - split: NFT
    path:
    - "test_data/objective/NFT_Benchmark_modified.csv"
  - split: Security
    path:
    - "test_data/objective/Security_Benchmark_modified.csv"
  - split: Smart_contract
    path:
    - "test_data/objective/SmartContracts_benchmark.csv"

- config_name: objective_infrastructure
  data_files:
  - split: Infrastructrue
    path:
    - "test_data/objective/infra_benchmark.csv"
  
- config_name: subjective_normal
  data_files:
  - split: Tokenomist
    path:
    - "test_data/subjective/Token.jsonl"
  - split: Fundamentals
    path:
    - "test_data/subjective/Blockchain_Fundamentals_benchmark.jsonl"
  - split: DAO
    path:
    - "test_data/subjective/DAO.jsonl"
  - split: Defi
    path:
    - "test_data/subjective/Defi.jsonl"
  - split: MEME
    path:
    - "test_data/subjective/MEME.jsonl"
  - split: NFT
    path:
    - "test_data/subjective/NFT.jsonl"
  - split: Security
    path:
    - "test_data/subjective/Security.jsonl"
  - split: Smart_contract
    path:
    - "test_data/subjective/smart_contract.jsonl"
- config_name: subjective_infrastructure
  data_files:
  - split: Infrastructure
    path:
    - "test_data/subjective/Infra.jsonl"

  
---

# 🔍 DMind Benchmark
A comprehensive framework for evaluating large language models (LLMs) on blockchain, cryptocurrency, and Web3 knowledge across multiple domains.

## 📊 Overview

This project provides tools to benchmark AI models on their understanding of blockchain concepts through both objective (multiple-choice) and subjective (open-ended) questions. The framework covers various domains including:

- 🧱 Blockchain Fundamentals
- 💰 DeFi (Decentralized Finance)
- 📝 Smart Contracts
- 🏛️ DAOs (Decentralized Autonomous Organizations)
- 🖼️ NFTs
- 🔒 Security
- 💹 Tokenomics
- 🎭 MEME coins
- 🌐 Blockchain Infrastructure

## ✨ Features

- 🧪 Test models on multiple-choice questions with single or multiple correct answers
- 📋 Evaluate models on open-ended questions requiring detailed explanations
- 🔄 Support for various question types including:
  - 📊 Calculation questions
  - 🔍 Code audit questions
  - 📝 Fill-in-blank questions
  - 📈 Market reasoning questions
  - 🔗 Matching questions
  - 📋 Ordering questions
  - ⚠️ Risk analysis questions
  - 🔮 Scenario analysis questions
  - ✏️ Short answer questions
  - 🧩 Strategy analysis questions
  - 🛡️ Vulnerability classification questions
- 🤖 Automated scoring and evaluation
- 📊 Calculate total scores and comparative analysis across models

## 🛠️ Installation

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your API settings in models.yml:

   ```bash
   api_base: "your_api_base"
   # Add other configuration settings as needed
   ```

## 📋 Usage

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

## 📁 Project Structure

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

## 📏 Evaluation Methodology

The evaluation framework uses various techniques to assess model performance:

- For objective questions: Exact matching against correct answers
- For subjective questions: Combination of keyword analysis, structured evaluation, and third-party AI evaluation when configured

## ⚙️ Customization

- Add new questions by extending the CSV/JSON files in test_data/
- Implement new question types by extending the BaseQuestion class
- Configure evaluation parameters in the respective question type implementations