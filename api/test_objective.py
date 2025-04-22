import os
import json
import pandas as pd
import requests
import yaml
from typing import Dict, List, Optional
from datetime import datetime
import time
from pathlib import Path
import argparse

def load_config() -> Dict:
    """Load configuration from YAML file"""
    # Try to load from current directory first
    current_dir = Path.cwd()
    config_path = current_dir / "models.yml"
    
    # If file doesn't exist in current directory, try the original path
    if not config_path.exists():
        config_path = Path(__file__).parent.parent.parent / "app" / "core" / "config" / "models.yml"
        # If still not found, check parent directory of test
        if not config_path.exists():
            config_path = Path(__file__).parent.parent.parent / "models.yml"
    
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Map API keys to model configurations
    api_key_map = {key['name']: key['key'] for key in config['api_keys']}
    for model in config['models']:
        model['api'] = api_key_map[model['api_key']]
        del model['api_key']
    
    return config

# Load configuration
TEST_CONFIG = load_config()

class ModelTester:
    def __init__(self, config: Dict):
        self.config = config
        self.api_base = config["api_base"]
        self.models = config["models"]
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        
    def load_test_data(self, file_path: str) -> pd.DataFrame:
        """Load test data"""
        try:
            # Build complete file path
            full_path = self.test_data_dir / "objective" / file_path
            return pd.read_csv(full_path)
        except Exception as e:
            print(f"Error loading test data: {e}")
            return pd.DataFrame()
            
    def make_api_request(self, model_config: Dict, prompt: str) -> Dict:
        """Send API request"""
        Skey = model_config["api"]
        provider = model_config.get("provider", "")

        max_retries = 10  # Maximum retry attempts
        retry_delay = 15  # Retry interval (seconds)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Handle requests for different providers
                if provider.lower() == "google":
                    # Handle requests for Google Gemini models
                    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_config['model']}:generateContent?key={Skey}"
                    headers = {
                        'Content-Type': 'application/json'
                    }
                    
                    data = {
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }],
                        # "temperature": model_config["parameters"].get("temperature", 0.7),
                        # "maxOutputTokens": model_config["parameters"].get("max_tokens", 1000)
                    }
                    
                    response = requests.post(api_url, headers=headers, json=data)
                else:
                    # Default handling (OpenAI, Anthropic, etc.)
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
                    
                    response = requests.post(self.api_base, headers=headers, json=data)
                
                end_time = time.time()
                
                if response.status_code == 200:
                    try:
                        response_json = response.json()
                        return {
                            "status_code": response.status_code,
                            "response": response_json,
                            "time_taken": end_time - start_time,
                            "attempts": attempt + 1
                        }
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON response: {e}")
                        print(f"Response content: {response.text}")
                        if attempt < max_retries - 1:
                            print(f"Will retry in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                else:
                    print(f"API request failed, status code: {response.status_code}")
                    print(f"Response content: {response.text}")
                    if attempt < max_retries - 1:
                        print(f"Will retry in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                
            except Exception as e:
                print(f"Error during API request: {e}")
                if attempt < max_retries - 1:
                    print(f"Will retry in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
        
        # If all retries failed
        return {
            "status_code": 500,
            "response": None,
            "time_taken": end_time - start_time if 'end_time' in locals() else 0,
            "attempts": max_retries
        }
        
    def evaluate_model(self, model_config: Dict, test_data: pd.DataFrame, dataset_name: str) -> Dict:
        """Evaluate a single model"""
        results = []
        total_samples = len(test_data)
        total_score = 0
        max_score = 0
        
        for idx, row in test_data.iterrows():
            question = row["Question"]
            if "Option E" in row and pd.notna(row.get("Option E")) and len(str(row["Option E"])) >= 4:
                options = {
                    "A": row["Option A"],
                    "B": row["Option B"],
                    "C": row["Option C"],
                    "D": row["Option D"],
                    "E": row["Option E"]
                }
            else:
                options = {
                    "A": row["Option A"],
                    "B": row["Option B"],
                    "C": row["Option C"],
                    "D": row["Option D"]
                }
            correct_option = row["Correct option"]
            
            # Determine question type (single/multiple choice) and standardize answer format
            is_multiple_choice = '/' in correct_option or ',' in correct_option or len(correct_option.strip()) > 1
            if is_multiple_choice:
                # Process multiple-choice answer format
                # Remove all spaces and commas, then sort by letter
                answers = ''.join(correct_option.replace(' ', '').replace(',', '').upper())
                correct_option = '/'.join(sorted(answers))
            
            question_score = 3 if is_multiple_choice else 2
            max_score += question_score
            
            # Build prompt
            base_prompt = "You are a professional quiz assistant. Please carefully read the question and output only the letter of the option you think is correct. If there are multiple correct answers, please separate them with a / (example: A/B). Do not explain, do not output anything else, do not output anything else, do not output anything else, do not output anything else.\n\n"
            prompt = f"{base_prompt}Question: {question}\n\nOptions:"
            for opt, content in options.items():
                prompt += f"\n{opt}. {content}"
            
            api_result = self.make_api_request(model_config, prompt)
            
            response_content = None
            if api_result["response"] and isinstance(api_result["response"], dict):
                provider = model_config.get("provider", "").lower()
                if provider == "google":
                    # Handle Gemini response
                    try:
                        if "candidates" in api_result["response"]:
                            response_content = api_result["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
                    except (KeyError, IndexError):
                        pass
                elif "choices" in api_result["response"]:
                    try:
                        response_content = api_result["response"]["choices"][0]["message"]["content"].strip()
                    except (KeyError, IndexError):
                        pass
                elif "content" in api_result["response"]:
                    response_content = api_result["response"]["content"].strip()
                elif "response" in api_result["response"]:
                    response_content = api_result["response"]["response"].strip()
            
            # Check if the answer is correct
            is_correct = False
            partial_correct = False
            
            # Extract valid model answers (usually A, B, C, D, etc.)
            valid_answers = []
            for letter in response_content.upper():
                if letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                    valid_answers.append(letter)
                    
            # Determine if it's a multiple-choice question
            is_multiple_choice = False
            if "/" in correct_option or "," in correct_option or len(correct_option) > 1:
                is_multiple_choice = True
                # Format correct options for multiple-choice questions
                correct_options = []
                if "/" in correct_option:
                    correct_options = correct_option.split("/")
                elif "," in correct_option:
                    correct_options = [c.strip() for c in correct_option.split(",")]
                else:
                    correct_options = list(correct_option.upper())
                
                # Convert all correct options to uppercase and sort them
                correct_options = [opt.strip().upper() for opt in correct_options]
                correct_options = sorted(correct_options)
                
                # Check if the answer is completely correct or partially correct
                if set(valid_answers) == set(correct_options):
                    is_correct = True
                    partial_correct = False
                elif all(ans in correct_options for ans in valid_answers):
                    is_correct = False
                    partial_correct = True if len(valid_answers) > 0 else False
                else:
                    is_correct = False
                    partial_correct = False
                    
                # Format correct options as A/B/C format
                correct_option = "/".join(correct_options)
            else:
                # Single-choice question logic, must provide and only provide one correct answer
                if len(valid_answers) == 1 and valid_answers[0] == correct_option.upper():
                    is_correct = True
                else:
                    is_correct = False
                    
            # Define a more concise print format
            print(f"\nQuestion {row['No']}:")
            print(f"Type: {'Multiple Choice' if is_multiple_choice else 'Single Choice'}")
            print(f"Question: {question}")
            print("Options:")
            for opt_key, opt_value in options.items():
                print(f"{opt_key}. {opt_value}")
            print(f"Correct Answer: {correct_option}")
            print(f"Model Answer: {''.join(valid_answers)}")
            print(f"Is Correct: {'Yes' if is_correct else 'No'}")
            print("-" * 50)
            
            total_score += (3 if is_correct and is_multiple_choice else 2 if is_correct else 1 if partial_correct else 0)
            
            result = {
                "sample_id": row["No"],
                "question": question,
                "options": options,
                "correct_option": correct_option,
                "actual": response_content,
                "is_correct": is_correct,
                "partial_correct": partial_correct,
                "score": 3 if is_correct and is_multiple_choice else 2 if is_correct else 1 if partial_correct else 0,
                "time_taken": api_result["time_taken"],
                "status": "success" if api_result["status_code"] == 200 and response_content else "error"
            }
            results.append(result)
            
            # Add delay to avoid API limits
            time.sleep(1)
        
        # Calculate final score (mapped to 12.5 points)
        final_score = (total_score / max_score) * 12.5 if max_score > 0 else 0
            
        return {
            "model_name": model_config["name"],
            "dataset_name": dataset_name,
            "total_samples": total_samples,
            "total_score": total_score,
            "max_score": max_score,
            "final_score": final_score,
            "successful_samples": len([r for r in results if r["status"] == "success"]),
            "average_time": sum(r["time_taken"] for r in results) / len(results) if results else 0,
            "results": results
        }
        
    def collect_historical_results(self, model_name: str) -> List[Dict]:
        """Collect all historical test results for a specified model
        Args:
            model_name: Model name
        Returns:
            List[Dict]: List of all historical test results
        """
        historical_results = []
        model_dir = self.results_dir / model_name / "objective"
        
        if not model_dir.exists():
            return historical_results
            
        # Iterate through all JSON files in the model directory
        for file in model_dir.glob("*.json"):
            if file.name.startswith("all_results_"):
                continue  # Skip summary files
                
            try:
                with open(file, "r") as f:
                    result = json.load(f)
                    historical_results.append(result)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
                
        return historical_results
        
    def run_tests(self, model_name: Optional[str] = None, generate_summary: bool = True):
        """Run tests
        Args:
            model_name: Optional, specify the name of the model to test. If None, all models will be tested
            generate_summary: Whether to generate summary results files
        """
        # List of test datasets
        test_datasets = [
            # "Blockchain_Fundamentals_benchmark.csv",
            # "Security_Benchmark_modified.csv",
            # "DAO2.csv",
            # "SmartContracts_benchmark.csv",
            # "Defi_benchmark.csv",
            # "MEME_Benchmark_modified.csv",
            "infra_benchmark.csv",
            # "Tokenomist.csv",
            # "NFT_Benchmark_modified.csv"
        ]
        
        model_results = {}  # Used to store all results for each model
        
        for dataset in test_datasets:
            test_data = self.load_test_data(dataset)
            if test_data.empty:
                print(f"No test data available for {dataset}")
                continue
                
            if model_name:
                # Test specified model
                model_config = next((m for m in self.models if m["name"] == model_name), None)
                if not model_config:
                    print(f"Model {model_name} not found in configuration")
                    return
                    
                # Create model-specific results directory
                model_results_dir = self.results_dir / model_config["name"] / "objective"
                model_results_dir.mkdir(parents=True, exist_ok=True)
                    
                print(f"Testing model {model_config['name']} on dataset {dataset}")
                results = self.evaluate_model(model_config, test_data, dataset)
                
                # Save single dataset results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = model_results_dir / f"{dataset.replace('.csv', '')}_{timestamp}.json"
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Test results saved to {results_file}")
                
                # Update model results
                if model_config["name"] not in model_results:
                    model_results[model_config["name"]] = []
                model_results[model_config["name"]].append(results)
            else:
                # Test all models
                for model_config in self.models:
                    # Create model-specific results directory
                    model_results_dir = self.results_dir / model_config["name"] / "objective"
                    model_results_dir.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Testing model {model_config['name']} on dataset {dataset}")
                    results = self.evaluate_model(model_config, test_data, dataset)
                    
                    # Save single model and dataset results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_file = model_results_dir / f"{dataset.replace('.csv', '')}_{timestamp}.json"
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"Test results saved to {results_file}")
                    
                    # Update model results
                    if model_config["name"] not in model_results:
                        model_results[model_config["name"]] = []
                    model_results[model_config["name"]].append(results)
        
        # Update summary results for each model
        if generate_summary:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for model_name, results in model_results.items():
                model_results_dir = self.results_dir / model_name / "objective"
                # Collect historical results
                historical_results = self.collect_historical_results(model_name)
                # Merge current results and historical results
                all_model_results = historical_results + results
                # Calculate total score
                total_final_score = sum(result["final_score"] for result in all_model_results)
                # Add total score to summary results
                summary_results = {
                    "model_name": model_name,
                    "total_final_score": total_final_score,
                    "dataset_results": all_model_results
                }
                # Update or create all_results file
                model_all_results_file = model_results_dir / "all_results.json"
                with open(model_all_results_file, "w") as f:
                    json.dump(summary_results, f, indent=2)
                print(f"Updated all results for {model_name} in {model_all_results_file}")
                print(f"Total final score for {model_name}: {total_final_score}")
                print(f"Total number of test results: {len(all_model_results)}")

def main():
    parser = argparse.ArgumentParser(description='Run model tests')
    parser.add_argument('--model', type=str, help='Name of the model to test. If not specified, all models will be tested.')
    parser.add_argument('--no-summary', action='store_true', help='Do not generate summary results files')
    args = parser.parse_args()
    
    tester = ModelTester(TEST_CONFIG)
    tester.run_tests(args.model, not args.no_summary)

if __name__ == "__main__":
    main() 