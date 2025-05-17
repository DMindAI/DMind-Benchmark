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

| [Paper](https://arxiv.org/abs/2504.16116) | [Dataset](https://huggingface.co/datasets/DMindAI/DMind_Benchmark/tree/main/test_data) |


## Latest LLM Leaderboard In Web3

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6417e25e058f65de43201023/sQKUonttcXrlgySN7SV64.png)


## Latest Web3 LLM Benchmark Table
This table presents the performance scores (%) of State-of-the-Art (SOTA) LLMs on the DMind Benchmark across nine Web3 subdimensions: Fundamentals, Infrastructure, Smart Contract Analysis, DeFi, DAOs, NFTs, Tokenomics, Meme, and Security. Scores are normalized to 100. Higher values indicate better performance in each subdimension.

| Model                        | Fund. | Infra. | S.C.  | DeFi  | DAOs  | NFTs  | Token | Meme  | Sec.  |
|------------------------------|-------|--------|-------|-------|-------|-------|-------|-------|-------|
| Claude 3.7-Sonnet            | 89.69 | 94.97  | 89.67 | 83.06 | 73.32 | 81.80 | 24.80 | 63.70 | 71.18 |
| Claude 3.5-Sonnet            | 89.28 | 94.85  | 87.50 | 80.85 | 71.69 | 80.45 | 24.40 | 62.50 | 67.36 |
| DMind-1                      |  88.84     |    97.34    |   86.27    |   84.53    |    74.23     |  84.29   |  28.40     | 70.63      |  75.52     |
| DeepSeek R1                  | 91.55 | 97.03  | 82.83 | 82.63 | 72.78 | 79.64 | 22.80 | 69.44 | 68.40 |
| DeepSeek V3                  | 90.31 | 95.81  | 83.00 | 77.55 | 73.68 | 74.35 | 23.80 | 63.70 | 69.44 |
| Gemini 2.5 Pro (Preview-05-06)| 81.03 | 93.66  | 81.37 | 78.16 | 67.88 | 76.87 | 19.40 | 67.96 | 70.49 |
| GPT-o4-mini-high             | 91.75 | 98.57  | 87.02 | 83.26 | 74.05 | 81.07 | 23.00 | 74.63 | 64.80 |
| GPT-o3                       | 92.99 | 98.36  | 88.43 | 81.02 | 74.59 | 80.52 | 24.20 | 71.67 | 71.01 |
| GPT-o1                       | 90.31 | 98.36  | 89.31 | 83.06 | 68.24 | 69.71 | 23.40 | 51.11 | 67.45 |
| GPT-4.1                      | 88.87 | 97.55  | 87.45 | 77.35 | 73.14 | 75.60 | 22.40 | 70.19 | 69.62 |
| Grok3 beta                   | 90.72 | 96.52  | 88.08 | 81.26 | 71.87 | 80.69 | 24.00 | 73.70 | 72.35 |
| Qwen3-235B A22B              | 88.66 | 97.60  | 79.88 | 79.39 | 75.32 | 79.73 | 26.40 | 70.56 | 70.40 |

## Latest Web3 Mini LLMs Benchmark Table

This table presents the performance scores (%) of Mini LLMs on the DMind Benchmark across nine Web3 subdimensions: Fundamentals, Infrastructure, Smart Contract Analysis, DeFi, DAOs, NFTs, Tokenomics, Meme, and Security. Scores are normalized to 100. Higher values indicate better performance in each subdimension.

| Model                              | Fund. | Infra. | S.C.  | DeFi  | DAOs  | NFTs  | Token | Meme  | Sec.  |
|-------------------------------------|-------|--------|-------|-------|-------|-------|-------|-------|-------|
| Claude 3-Haiku                     | 87.13 | 96.32  | 86.08 | 75.46 | 72.05 | 83.22 | 24.40 | 63.89 | 70.57 |
| Claude 3-Opus                       | 83.51 | 91.72  | 78.82 | 77.55 | 72.23 | 77.73 | 24.60 | 69.44 | 70.75 |
| DMind-1-mini                        |   87.39    |   96.89     |    84.88   |   82.80    | 72.78      |   82.66    |   27.10    |  70.89     |   75.48    |
| DeepSeek-R1-Distill-Llama-70B       | 83.71 | 95.40  | 82.35 | 80.81 | 66.06 | 65.96 | 24.20 | 67.44 | 66.75 |
| DeepSeek-R1-Distill-Qwen-32B        | 83.51 | 92.43  | 77.25 | 76.32 | 72.05 | 75.61 | 22.40 | 70.37 | 67.10 |
| Gemini 2.5 Flash (Preview-04-17)    | 88.45 | 97.03  | 82.94 | 80.20 | 73.50 | 82.52 | 22.80 | 71.67 | 71.35 |
| Gemini 2.0 Flash (Experimental)     | 85.15 | 94.89  | 81.37 | 79.57 | 71.51 | 77.65 | 21.80 | 68.89 | 69.01 |
| GPT-o4-mini                         | 91.34 | 94.96  | 86.82 | 82.85 | 74.05 | 78.60 | 24.20 | 72.52 | 68.61 |
| GPT-o3-mini                         | 91.96 | 98.16  | 86.08 | 81.63 | 71.14 | 80.18 | 23.60 | 69.44 | 72.48 |
| GPT-o1-mini                         | 87.63 | 95.50  | 80.35 | 76.32 | 69.51 | 74.92 | 23.40 | 64.63 | 69.18 |
| GPT-4o-mini                         | 82.06 | 86.50  | 75.88 | 76.68 | 68.06 | 73.66 | 22.40 | 60.74 | 67.19 |
| Grok3 mini beta                     | 87.69 | 95.75  | 84.02 | 78.47 | 70.05 | 79.99 | 23.40 | 69.07 | 73.44 |
| Qwen3-32B                           | 84.69 | 96.50  | 78.50 | 79.50 | 66.97 | 70.70 | 25.20 | 55.63 | 66.63 |
| Qwen3-30B-A3B                       | 83.45 | 94.93  | 77.63 | 79.20 | 70.23 | 73.55 | 23.20 | 50.81 | 68.23 |
| QwQ-32B                             | 82.69 | 91.21  | 73.35 | 73.06 | 67.88 | 69.38 | 22.20 | 47.04 | 66.15 |


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
# Run tests for a specific model
make test model=claude-3-5-haiku-20241022

# Run objective tests only for a specific model
make test-objective model=claude-3-5-haiku-20241022

# Run subjective tests only for a specific model
make test-subjective model=claude-3-5-haiku-20241022

# Calculate scores for all models
make calculate

# Calculate score for a specific model
make calculate-model model=claude-3-5-haiku-20241022

# Clean all test results
make clean

# Show test results
make show

# Run the complete pipeline (clean, test, calculate, show) for a specific model
make pipeline model=claude-3-5-haiku-20241022

# Display help information
make help
```

### 🔄 Testing Multiple Models

You can test multiple models and compare their performance:

1. Add models to your models.yml configuration
2. Run tests for each model
3. Use the calculation tools to compare results

## 📁 Project Structure

```
├── api/                  # Core testing scripts
│   ├── test_objective.py       # Handles objective test questions (multiple choice)
│   ├── test_subjective.py      # Handles subjective test questions (open-ended)
│   ├── calculate_total_score.py # Calculates final scores across all datasets
│   ├── config_manager.py       # API configuration manager (handles models.yml config)
│   └── question_types/         # Question type implementation classes
│       ├── short_answer_question.py      # Short answer question handler
│       ├── scenario_analysis_question.py  # Scenario analysis question handler
│       ├── strategy_analysis_question.py  # Strategy analysis question handler
│       └── vulnerability_classification_question.py # Vulnerability classification handler
├── test_data/            # Test datasets
│   ├── objective/        # Multiple-choice question datasets (CSV format)
│   └── subjective/       # Open-ended question datasets (JSON format)
├── test_results/         # Test results storage
│   └── [model_name]/     # Model-specific result directories
│       ├── objective/    # Objective test results
│       └── subjective/   # Subjective test results
├── models.yml           # Configuration file for API keys and model settings
└── Makefile              # Test automation commands
```

## 📏 Evaluation Methodology

The evaluation framework uses various techniques to assess model performance:

- For objective questions: Exact matching against correct answers
- For subjective questions: Combination of keyword analysis, structured evaluation, and third-party AI evaluation when configured

## 🔑 Configuring API Keys and Base URLs

API keys and base URLs are configured in the `models.yml` file located in the root directory. The structure is as follows:

```yaml
api_base: "https://api.anthropic.com/v1/messages"  # Default API base URL

api_keys:
  - name: "anthropic"
    key: "your_anthropic_api_key"
  - name: "openai"
    key: "your_openai_api_key"
  - name: "google"
    key: "your_google_api_key"
  - name: "deepseek"
    key: "your_deepseek_api_key"

models:
  - name: "claude-3-5-haiku-20241022"
    model: "claude-3-5-haiku-20241022"
    provider: "anthropic"
    api_key: "anthropic"
    parameters:
      temperature: 0.7
      max_tokens: 1000
  # Add more models as needed
```

To add or modify models:
1. Add the API key to the `api_keys` section
2. Add the model configuration to the `models` section
3. The `api_key` field in the model configuration should reference a name from the `api_keys` section

## 🧠 Configuring Third-Party Evaluation Models

The system uses third-party AI models for evaluating subjective responses. This section explains how to configure these evaluation models in the `models.yml` file.

### 📝 Enhanced Models.yml Structure

For evaluation purposes, the `models.yml` file supports additional configuration sections:

```yaml
# Main API Base URL (for models being tested)
api_base: "https://api.anthropic.com/v1/messages"

# Dedicated Evaluation API Base URL (optional)
evaluation_api_base: "xxx"

api_keys:
  # Testing model API keys
  - name: "anthropic"
    key: "your_anthropic_api_key"
  
  # Evaluation model API keys
  - name: "claude_eval"
    key: "your_evaluation_api_key"
    model_name: "claude-3-7-sonnet-20250219"  # Associate specific model with this key
```

The `model_name` field in API keys is optional but allows automatic model selection when using a particular key.

### ⚙️ Configuring Dedicated Evaluation Models

You can configure specific models to be used only for evaluation purposes:

```yaml
models:
  # Models being tested
  - name: "claude-3-5-haiku-20241022"
    model: "claude-3-5-haiku-20241022"
    provider: "anthropic"
    api_key: "anthropic"
    parameters:
      temperature: 0.7
      max_tokens: 1000
  
  # Evaluation models
  - name: "claude_evaluation"
    provider: "anthropic"
    model: "claude-3-7-sonnet-20250219"
    parameters:
      temperature: 0
      max_tokens: 4000
    api_key: "claude_eval"
    api_base: "xxx"  # Optional: Override global API base
```

### 📄 Complete Example with Evaluation Configuration

Here's a complete example of a `models.yml` file with both testing and evaluation model configurations:

```yaml
# API Base URL Configuration
api_base: "https://api.anthropic.com/v1/messages"

# Evaluation API Base URL (Optional)
evaluation_api_base: "xxx"

# API Key Configuration
api_keys:
  # Testing model API keys
  - name: "anthropic"
    key: "your_anthropic_api_key"
  - name: "openai"
    key: "your_openai_api_key"
  
  # Evaluation model API keys
  - name: "claude_eval"
    key: "your_claude_evaluation_api_key"
    model_name: "claude-3-7-sonnet-20250219"
  - name: "openai_eval"
    key: "your_openai_evaluation_api_key"
    model_name: "gpt-4o"

# Model Configuration
models:
  # Testing models
  - name: "claude-3-5-haiku-20241022"
    model: "claude-3-5-haiku-20241022"
    provider: "anthropic"
    api_key: "anthropic"
    parameters:
      temperature: 0.7
      max_tokens: 1000
  
  # Evaluation models
  - name: "claude_evaluation"
    provider: "anthropic"
    model: "claude-3-7-sonnet-20250219"
    parameters:
      temperature: 0
      max_tokens: 4000
    api_key: "claude_eval"
  
  - name: "gpt4_evaluation"
    provider: "openai"
    model: "gpt-4o"
    parameters:
      temperature: 0
      max_tokens: 4000
    api_key: "openai_eval"
```

### 🔍 How Evaluation Models Are Selected

When subjective questions need to be evaluated, the system uses the following priority order:

1. First, it tries to use a model from the `models` section with the name specified in the code (e.g., "claude_evaluation")
2. If no specific model is specified, it tries models named "claude_evaluation" or "gpt4_evaluation"
3. If those aren't found, it uses the API configuration from the API key with name "claude_eval"
4. If none of the above are available, it falls back to default built-in configuration

You can specify which evaluation model to use in your code:

```python
# In your Python code, you can specify which model to use for evaluation
api_config = config_manager.get_third_party_api_config("gpt4_evaluation")
```

## 🔌 Handling Different AI Service Providers

The testing framework supports various AI service providers. The request format for each provider is handled differently in `test_objective.py` and `test_subjective.py`:

### Google (Gemini)

```python
# For Google Gemini models
api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_config['model']}:generateContent?key={Skey}"
headers = {
    'Content-Type': 'application/json'
}
data = {
    "contents": [{
        "parts": [{"text": prompt}]
    }]
}
```

### DeepSeek

```python
# For DeepSeek models (using OpenAI client)
client = OpenAI(api_key=Skey, base_url="https://api.deepseek.com")
response = client.chat.completions.create(
    model=model_config["model"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ],
    temperature=model_config["parameters"].get("temperature", 0.7),
    max_tokens=model_config["parameters"].get("max_tokens", 1000),
    stream=False
)
```

### Default (OpenAI, Anthropic, etc.)

```python
# For OpenAI, Anthropic, etc.
headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {Skey}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
}
data = {
    "model": model_config["model"],
    "messages": [{"role": "user", "content": prompt}],
    **model_config["parameters"]
}
```

To add support for a new provider:
1. Add a new condition in the `make_api_request` method in both `test_objective.py` and `test_subjective.py`
2. Implement the appropriate request format
3. Add proper response parsing logic for the new provider

## 🧵 Multi-Threading and Performance Configuration

The testing framework uses concurrent processing to speed up the evaluation of large datasets. This section explains how to configure multi-threading settings.

### 🔄 Multi-Threading Implementation

The objective testing system utilizes Python's `concurrent.futures` module with ThreadPoolExecutor to process multiple questions simultaneously:

```python
# Inside the evaluate_model method in test_objective.py
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    # Submit all questions to the thread pool
    futures = [executor.submit(process_question, (idx, row)) for idx, row in test_data.iterrows()]
    
    # Wait for all tasks to complete
    for future in concurrent.futures.as_completed(futures):
        try:
            # Get results of individual tasks
            result = future.result()
            print(f"Question {result['sample_id']} processed, score: {result['score']}")
        except Exception as exc:
            print(f"Error processing question: {exc}")
```

### ⚙️ Thread Count Configuration

You can adjust the number of parallel worker threads by modifying the `max_workers` parameter in both `test_objective.py` and `test_subjective.py`:

```python
# Default configuration: 50 parallel threads
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
```

To modify the thread count, edit this value in the files:
- For objective tests: `api/test_objective.py`
- For subjective tests: `api/test_subjective.py`

### 📊 Performance Considerations

When configuring thread count, consider the following:

1. **API Rate Limits**: Using too many threads might trigger rate limits on API services. Most providers have rate limits that could cause request failures if exceeded.

2. **System Resources**: Higher thread counts consume more system memory and could cause performance issues on machines with limited resources.

3. **Network Limitations**: More concurrent requests increase bandwidth usage, which might be a limiting factor in some environments.

4. **API Provider Guidelines**: Some API providers have guidelines about concurrent requests. Check their documentation to ensure compliance.

### 🔒 Thread Safety

The testing framework implements thread safety using the Python `threading.Lock` class to protect shared data:

```python
# Thread safety for results and scoring
results_lock = Lock()
score_lock = Lock()

# Thread-safe score update
with score_lock:
    nonlocal total_score
    total_score += question_score

# Thread-safe results update
with results_lock:
    nonlocal results
    results.append(result)
```

This ensures that concurrent threads don't interfere with each other when accessing shared data structures.

### 🔧 Configuration Recommendations

Based on different scenarios, here are some recommended thread count configurations:

- **Small Datasets (< 100 questions)**: 10-20 threads
- **Medium Datasets (100-500 questions)**: 30-50 threads
- **Large Datasets (> 500 questions)**: 50-100 threads

For API services with strict rate limits, consider lowering the thread count to avoid request failures.

## 📨 Response Handling

The framework handles response parsing for different providers:

- **Google Gemini**: Extracts answer from `response.candidates[0].content.parts[0].text`
- **OpenAI/Anthropic**: Extracts answer from `response.choices[0].message.content`
- **DeepSeek**: Uses OpenAI client and extracts answer from the response object

## 🏆 Scoring System

- **Objective tests**: Multiple-choice questions with automated scoring
  - Single-choice: 2 points for correct answers
  - Multiple-choice: 3 points for fully correct answers

- **Subjective tests**: Open-ended questions evaluated using:
  - Third-party AI evaluation (Claude-3-7-Sonnet)
  - Keyword matching as fallback

The final score for each model is calculated by combining results from all datasets, with each dataset given equal weight in the total score. 

## ⚙️ Customization

- Add new questions by extending the CSV/JSON files in test_data/
- Implement new question types by extending the BaseQuestion class
- Configure evaluation parameters in the respective question type implementations
