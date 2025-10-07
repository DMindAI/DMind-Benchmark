import os
import json
import pandas as pd
import requests
import yaml
from typing import Dict, List, Optional
from datetime import datetime
import time
from pathlib import Path
from openai import OpenAI
import argparse
import concurrent.futures
from threading import Lock
import random

def load_config() -> Dict:
    """Load configuration from YAML file"""
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
        self.test_data_dir = Path(__file__).parent.parent.parent.parent / "DMind_review" / "new_dataset"
        
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

        max_retries = 30  # Maximum retry attempts
        retry_delay = 5  # Retry interval (seconds)
        
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
                elif provider.lower() == "openai":
                    try:
                        base_url = model_config.get("base_url", "https://api.openai.com/v1")
                        print(Skey)
                        client = OpenAI(
                            base_url=base_url,
                            api_key=Skey,
                        )
                        # client = OpenAI()
                        
                        extra_headers = model_config.get("extra_headers", {})
                        extra_body = model_config.get("extra_body", {})
                        
                        completion = client.chat.completions.create(
                            extra_headers=extra_headers,
                            extra_body=extra_body,
                            model=model_config["model"],
                            # input=prompt,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            temperature=model_config.get("parameters", {}).get("temperature", 0.7),
                        )
                        
                        response_json = {
                            "choices": [
                                {
                                    "message": {
                                        "content": completion.choices[0].message.content
                                    }
                                }
                            ]
                        }
                        
                        end_time = time.time()
                        return {
                            "status_code": 200,
                            "response": response_json,
                            "time_taken": end_time - start_time,
                            "attempts": attempt + 1
                        }
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            end_time = time.time()
                            return {
                                "status_code": 500,
                                "response": {"error": str(e)},
                                "time_taken": end_time - start_time,
                                "attempts": attempt + 1
                            }
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
                        # "top_k": 20,
                        # "top_p": 0.95,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        # "max_tokens": 4096,
                        # "stream": "false"
                        # **model_config["parameters"]
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
        
        results_lock = Lock()
        score_lock = Lock()
        
        def process_question(row_tuple):
            idx, row = row_tuple
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
            original_correct_option = row["Correct option"]
            
            # Shuffle options to randomize their order
            random.seed(int(time.time() * 1000) % 10000 + row["No"])
            
            # Create mapping between original option keys and their content
            option_contents = list(options.items())
            
            # Shuffle the options
            random.shuffle(option_contents)
            
            # Create new options dictionary with shuffled content
            shuffled_options = {}
            original_to_new_mapping = {}  # Maps original keys to shuffled keys
            new_to_original_mapping = {}  # Maps new keys to original keys
            
            for new_key, (original_key, content) in zip(sorted(options.keys()), option_contents):
                shuffled_options[new_key] = content
                original_to_new_mapping[original_key] = new_key
                new_to_original_mapping[new_key] = original_key
            # shuffled_options = options.copy()
            # original_to_new_mapping = {k: k for k in options.keys()}
            # new_to_original_mapping = {k: k for k in options.keys()}
            
            # Map the correct option to the new shuffled position
            # Handle different formats of correct options (single, multiple with / or ,)
            if "/" in original_correct_option or "," in original_correct_option or len(original_correct_option) > 1:
                # Multiple correct options case
                correct_options = []
                
                # Handle special format starting with //
                if original_correct_option.startswith("//"):
                    original_correct_option = original_correct_option[2:]  # Remove // prefix
                
                if "/" in original_correct_option:
                    correct_list = original_correct_option.split("/")
                elif "," in original_correct_option:
                    correct_list = [c.strip() for c in original_correct_option.split(",")]
                else:
                    correct_list = list(original_correct_option.upper())
                
                # Convert each original correct option to its new position
                for opt in correct_list:
                    if opt.strip().upper() in original_to_new_mapping:
                        correct_options.append(original_to_new_mapping[opt.strip().upper()])
                
                # Format back to original format
                correct_option = "/".join(sorted(correct_options))
                if original_correct_option.startswith("//"):
                    correct_option = "//" + correct_option
            else:
                # Single correct option case
                correct_option = original_to_new_mapping.get(original_correct_option.upper(), original_correct_option)
            
            # Build prompt with shuffled options
            base_prompt = """
<Role>\nYou are a professional quiz assistant.\n\n<Task>\nYour task is to answer multiple-choice questions in the following format:\n1. Read the question carefully\n2. Output only the letter(s) of the correct option(s) (A, B, C, or D)\n3. If there are multiple correct answers, separate them with slashes (e.g., A/B)\n4. Do not explain your choice\n
5. Do not output any other content,Do not output any other content
"""
            prompt = f"{base_prompt}Question: {question}\n\nOptions:"
            for opt, content in shuffled_options.items():
                prompt += f"\n{opt}. {content}"
            
            api_result = self.make_api_request(model_config, prompt)
            print(f"Question {row['No']} API request completed")
            
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
            invalid_response = False
            seen_options = set()
            print(f"Response content: {response_content}")
            if response_content != None:
                response_content = response_content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
                if "</think>\n" in response_content:
                    response_content = response_content.split("</think>\n")[1]

                for letter in response_content.upper():
                    if letter in ["A", "B", "C", "D", "E"]:
                        # Check for duplicate options
                        if letter in seen_options:
                            print(f"Detected duplicate option: {letter}")
                            invalid_response = True
                            break
                        seen_options.add(letter)
                        valid_answers.append(letter)
                    elif letter.isalpha() and letter not in ["A", "B", "C", "D", "E"]:
                        print(f"Detected invalid option: {letter}")
                        invalid_response = True
                        break
            
                # Check if number of options exceeds 5
                if len(valid_answers) > 5:
                    print(f"Number of options exceeds limit: {len(valid_answers)} > 5")
                    invalid_response = True

            else:
                invalid_response = True 

            # If response is invalid, need to resend request
            retry_count = 0
            if invalid_response:
                print(f"Model returned invalid response: {response_content}")
                print("Resending request...")
                
                # Maximum retries: 30
                max_retries = 30
                
                while invalid_response and retry_count < max_retries:
                    retry_count += 1
                    print(f"Question {row['No']} retry {retry_count}/{max_retries}...")
                    
                    # Add additional prompts emphasizing ABCDE only
                    retry_prompt = prompt + f"\n\nWarning: Your previous answer '{response_content}' has incorrect format. Please strictly follow these requirements:\n1. Use only the option letters A, B, C, D, E\n2. Do not repeat any options\n3. For multiple answers, separate with / (e.g., A/B)\n4. Do not output any explanations or other content\n5. Total number of options should not exceed 5"
                    api_result = self.make_api_request(model_config, retry_prompt)
                    
                    if api_result["response"] and isinstance(api_result["response"], dict):
                        provider = model_config.get("provider", "").lower()
                        if provider == "google":
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
                    
                    # Re-validate response
                    valid_answers = []
                    invalid_response = False
                    seen_options = set()

                    if response_content != None:
                        for letter in response_content.upper():
                            if letter in ["A", "B", "C", "D", "E"]:
                                if letter in seen_options:
                                    print(f"Still detected duplicate option after retry: {letter}")
                                    invalid_response = True
                                    break
                                seen_options.add(letter)
                                valid_answers.append(letter)
                            elif letter.isalpha() and letter not in ["A", "B", "C", "D", "E"]:
                                print(f"Still detected invalid option after retry: {letter}")
                                invalid_response = True
                                break
                    else:
                        invalid_response = True
                    if len(valid_answers) > 5:
                        print(f"Number of options still exceeds limit after retry: {len(valid_answers)} > 5")
                        invalid_response = True
                    
                    if not invalid_response:
                        print(f"Question {row['No']} retry successful, received valid response: {valid_answers}")
                    
                    # Avoid frequent API requests
                    time.sleep(1)
                
                # If still invalid after retries, mark as error
                if invalid_response:
                    print(f"Question {row['No']} still invalid after {max_retries} retries, marking as error")
                    is_correct = False
                    partial_correct = False
                    
                    # Record detailed information about this failed request
                    print(f"Request content: {prompt}")
                    print(f"Model name: {model_config['name']}")
                    print(f"Dataset: {dataset_name}")
                    print(f"Question ID: {row['No']}")
            
            # Determine if it's a multiple-choice question
            is_multiple_choice = False
            if "/" in correct_option or "," in correct_option or len(correct_option) > 1:
                is_multiple_choice = True
                # Format correct options for multiple-choice questions
                correct_options = []
                
                # Handle special format starting with //
                if correct_option.startswith("//"):
                    correct_option = correct_option[2:]  # Remove // prefix
                
                if "/" in correct_option:
                    correct_options = correct_option.split("/")
                elif "," in correct_option:
                    correct_options = [c.strip() for c in correct_option.split(",")]
                else:
                    correct_options = list(correct_option.upper())
                
                # Convert all correct options to uppercase and sort them
                correct_options = [opt.strip().upper() for opt in correct_options if opt.strip()]
                correct_options = sorted(correct_options)
                
                # Compare model answers and correct answers
                valid_answers_set = set(valid_answers)
                correct_options_set = set(correct_options)
                
                # Check if answer is completely correct
                if valid_answers_set == correct_options_set:
                    is_correct = True
                    partial_correct = False
                # Check if answer is partially correct (all model answers are correct but incomplete)
                elif valid_answers_set.issubset(correct_options_set) and len(valid_answers) > 0:
                    # Answer is incomplete but no errors, considered partially correct
                    is_correct = False
                    partial_correct = True
                # Check if there are incorrect answers
                else:
                    is_correct = False
                    partial_correct = False
                
                # Format correct options as A/B/C format
                correct_option = "/".join(correct_options)
            else:
                # Single-choice question logic, must provide and only provide one correct answer
                if len(valid_answers) == 1 and valid_answers[0] == correct_option.upper():
                    is_correct = True
                    partial_correct = False
                else:
                    is_correct = False
                    partial_correct = False
            
            # Define a more concise print format
            print(f"\nQuestion {row['No']}:")
            print(f"Type: {'Multiple Choice' if is_multiple_choice else 'Single Choice'}")
            print(f"Question: {question}")
            print("Original Options:")
            for opt_key, opt_value in options.items():
                print(f"{opt_key}. {opt_value}")
            print("Shuffled Options:")
            for opt_key, opt_value in shuffled_options.items():
                print(f"{opt_key}. {opt_value}")
            print(f"Original Correct Answer: {original_correct_option}")
            print(f"Shuffled Correct Answer: {correct_option}")
            print(f"Model Answer: {''.join(valid_answers)}")
            print(f"Response Valid: {'Yes' if not invalid_response else 'No'}")
            print(f"Retry Count: {retry_count}")
            print(f"Is Correct: {'Yes' if is_correct else 'No'}")
            print("-" * 50)
            
            # Calculate score
            question_score = 3 if is_correct and is_multiple_choice else 2 if is_correct else 1 if partial_correct else 0
            
            # Update max_score (maximum possible score)
            with score_lock:
                nonlocal max_score
                # Add the maximum possible score for this question type
                max_score += 3 if is_multiple_choice else 2
            
            # Thread-safe increase of total_score
            with score_lock:
                nonlocal total_score
                total_score += question_score
            
            result = {
                "sample_id": row["No"],
                "question": question,
                "options": options,
                "correct_option": correct_option,
                "actual": response_content,
                "valid_response": not invalid_response,
                "retry_count": retry_count,
                "is_correct": is_correct,
                "partial_correct": partial_correct,
                "score": question_score,
                "time_taken": api_result["time_taken"],
                "status": "success" if api_result["status_code"] == 200 and response_content and not invalid_response else "error"
            }
            
            # Thread-safe addition of results
            with results_lock:
                nonlocal results
                results.append(result)
            
            return result
        
        # Use ThreadPoolExecutor for multithreaded processing
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            # Submit all questions to the thread pool
            futures = [executor.submit(process_question, (idx, row)) for idx, row in test_data.iterrows()]
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    # Get the result of a single task
                    result = future.result()
                    print(f"Question {result['sample_id']} processing completed, score: {result['score']}")
                except Exception as exc:
                    print(f"Error processing question: {exc}")
        
        total_time = time.time() - start_time
        print(f"All questions processed in parallel, total time: {total_time:.2f} seconds")
        
        # Sort results by question ID
        results.sort(key=lambda x: x['sample_id'])
        
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
            "results": results,
            "total_processing_time": total_time
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
            "Blockchain_Fundamentals.csv",
            "Security.csv",
            "DAO.csv",
            "SmartContract.csv",
            "Defi.csv",
            "MEME.csv",
            "Infra.csv",
            "Tokenomist.csv",
            "NFT.csv"
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
        

def main():
    parser = argparse.ArgumentParser(description='Run model tests')
    parser.add_argument('--model', type=str, help='Name of the model to test. If not specified, all models will be tested.')
    parser.add_argument('--no-summary', action='store_true', help='Do not generate summary results files')
    args = parser.parse_args()
    
    tester = ModelTester(TEST_CONFIG)
    tester.run_tests(args.model, not args.no_summary)

if __name__ == "__main__":
    main() 