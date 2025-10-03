import os
import json
import pandas as pd
import requests
import yaml
from typing import Dict, List, Optional, Type, Any
from datetime import datetime
import time
from pathlib import Path
import argparse
from openai import OpenAI
from question_types.base_question import BaseQuestion
from question_types.strategy_analysis_question import StrategyAnalysisQuestion
from question_types.matching_question import MatchingQuestion
from question_types.ordering_question import OrderingQuestion
from question_types.calculation_question import CalculationQuestion
from question_types.fill_in_blank_question import FillInBlankQuestion
from question_types.market_reasoning_question import MarketReasoningQuestion
from question_types.short_answer_question import ShortAnswerQuestion
from question_types.risk_analysis_question import RiskAnalysisQuestion
from question_types.scenario_analysis_question import ScenarioAnalysisQuestion
from question_types.vulnerability_classification_question import VulnerabilityClassificationQuestion
from question_types.code_audit_question import CodeAuditQuestion
import concurrent.futures

# Question type mapping
QUESTION_TYPES = {
    "strategy_analysis": StrategyAnalysisQuestion,
    "matching": MatchingQuestion,
    "ordering": OrderingQuestion,
    "calculation": CalculationQuestion,
    "fill_in_blank": FillInBlankQuestion,
    "market_reasoning": MarketReasoningQuestion,
    "short_answer": ShortAnswerQuestion,
    "risk_analysis": RiskAnalysisQuestion,
    "scenario_analysis": ScenarioAnalysisQuestion,
    "vulnerability_classification": VulnerabilityClassificationQuestion,
    "code_audit": CodeAuditQuestion
}

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
    
    api_key_map = {key['name']: key['key'] for key in config['api_keys']}
    for model in config['models']:
        model['api'] = api_key_map[model['api_key']]
        del model['api_key']
    
    return config

class SubjectiveModelTester:
    def __init__(self, config: Dict):
        self.config = config
        self.api_base = config["api_base"]
        self.models = config["models"]
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        
    def load_test_data(self, file_path: str) -> Dict:
        """Load subjective test data"""
        try:
            # Build complete file path
            full_path = self.test_data_dir / "subjective_converted" / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading test data: {e}")
            return {}
            
    def make_api_request(self, model_config: Dict, prompt: str) -> Dict:
        """Send API request"""
        Skey = model_config["api"]

        max_retries = 30  # Maximum retry attempts
        retry_delay = 10  # Retry interval (seconds)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                provider = model_config.get("provider", "").lower()
                
                if provider == "google":
                    # Handle requests for Google Gemini models
                    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_config['model']}:generateContent?key={Skey}"
                    headers = {
                        'Content-Type': 'application/json'
                    }
                    
                    data = {
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }]
                    
                    }
                    
                    # Output request content
                    print("\n" + "="*50)
                    print("Request content:")
                    print(f"URL: {api_url}")
                    print(f"Headers: {json.dumps(headers, indent=2, ensure_ascii=False)}")
                    print(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    print("="*50 + "\n")
                    
                    response = requests.post(api_url, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        response_json = response.json()
                elif provider.lower() == "openai":
                    # 处理OpenAI请求
                    try:
                        # 初始化OpenAI客户端
                        base_url = model_config.get("base_url", "https://api.openai.com/v1")
                        print(Skey)
                        client = OpenAI(
                            base_url=base_url,
                            api_key=Skey,
                        )
                        # client = OpenAI()
                        
                        # 准备额外头部和参数
                        extra_headers = model_config.get("extra_headers", {})
                        extra_body = model_config.get("extra_body", {})
                        
                        # 创建完成请求
                        response = client.chat.completions.create(
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
                        
                        response.choices[0].message.content = response.choices[0].message.content.split("</think>\n")[1]
                        response_json = {
                            "id": response.id,
                            "choices": [
                                {
                                    "message": {
                                        "content": response.choices[0].message.content,
                                        "role": response.choices[0].message.role
                                    },
                                    "index": 0,
                                    "finish_reason": response.choices[0].finish_reason
                                }
                            ],
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            }
                        }
                        response_status = 200
                    except Exception as e:
                        print(f"OpenAI API call error: {e}")
                        if attempt < max_retries - 1:
                            print(f"Will retry in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                elif provider == "deepseek":
                    # Handle DeepSeek model requests using OpenAI client
                    print("\n" + "="*50)
                    print("Request content:")
                    print(f"DeepSeek API Request: model={model_config['model']}")
                    print(f"prompt: {prompt[:100]}...")
                    print("="*50 + "\n")
                    
                    try:
                        client = OpenAI(api_key=Skey, base_url=model_config["base_url"])
                        
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
                        
                        # Convert OpenAI response object to dictionary
                        response.choices[0].message.content = response.choices[0].message.content.split("</think>\n")[1]
                        response_json = {
                            "id": response.id,
                            "choices": [
                                {
                                    "message": {
                                        "content": response.choices[0].message.content,
                                        "role": response.choices[0].message.role
                                    },
                                    "index": 0,
                                    "finish_reason": response.choices[0].finish_reason
                                }
                            ],
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            }
                        }
                        
                        response_status = 200
                    except Exception as e:
                        print(f"DeepSeek API call error: {e}")
                        if attempt < max_retries - 1:
                            print(f"Will retry in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                        response_json = None
                        response_status = 500
                else:
                    # Default handling (OpenAI, Anthropic, etc.)
                    headers = {
                        'Accept': 'application/json',
                        'Authorization': f'Bearer {Skey}',
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }

                    prompt_enforce = """

"""
                    
                    data = {
                        "model": model_config["model"],
                        "messages": [{"role": "user", "content": prompt + prompt_enforce}],
                        "stream": False,
                        "temperature": 0.7,
                        "max_tokens": 4096,
                        **model_config["parameters"]
                    }
                    
                    # Output request content
                    print("\n" + "="*50)
                    print("Request content:")
                    print(f"URL: {self.api_base}")
                    print(f"Headers: {json.dumps(headers, indent=2, ensure_ascii=False)}")
                    print(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    print("="*50 + "\n")
                    
                    response = requests.post(self.api_base, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        response_json = response.json()
                    else:
                        response_json = None
                
                end_time = time.time()
                
                # Output response content
                print("\n" + "="*50)
                print("Response content:")
                
                if provider == "deepseek":
                    print(f"Status Code: {response_status}")
                    if response_json:
                        print(f"Response: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
                else:
                    print(f"Status Code: {response.status_code}")
                    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False) if response.status_code == 200 else response.text}")
                
                print("="*50 + "\n")
                
                if (provider == "deepseek" and response_status == 200) or (provider != "deepseek" and response.status_code == 200):
                    if response_json:
                        return {
                            "status_code": 200,
                            "response": response_json,
                            "time_taken": end_time - start_time,
                            "attempts": attempt + 1
                        }
                    else:
                        if attempt < max_retries - 1:
                            print(f"Failed to parse response, will retry in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                else:
                    print(f"API request failed")
                    if provider != "deepseek":
                        print(f"Status code: {response.status_code}")
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
        
    def evaluate_model(self, model_config: Dict, test_data: List[Dict], dataset_name: str) -> Dict:
        """
        Evaluate model's performance on test data
        
        Args:
            model_config: Model configuration
            test_data: List of test data
            dataset_name: Dataset name
            
        Returns:
            Dict: Evaluation results
        """
        results = []
        total_score = 0
        total_possible = 0
        
        for question_data in test_data:
            question_type = question_data.get("question_type", "")
            
            # Get corresponding question type class
            question_class = QUESTION_TYPES.get(question_type)
            if not question_class:
                print(f"Unknown question type: {question_type}")
                continue
                
            # Create question instance
            question = question_class(question_data)
            
            # Build prompt
            prompt = question.build_prompt()
            
            # Call model API
            print(f"Prompt: {prompt}")
            api_result = self.make_api_request(model_config, prompt)
            
            # Extract model response
            model_response = ""
            if api_result["status_code"] == 200:
                provider = model_config.get("provider", "").lower()
                if provider == "google":
                    # Handle Gemini response
                    try:
                        if "candidates" in api_result["response"]:
                            model_response = api_result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            model_response = "Unable to extract model response"
                    except (KeyError, IndexError):
                        model_response = "Unable to extract model response"
                elif provider == "deepseek":
                    # Handle DeepSeek response
                    try:
                        model_response = api_result["response"]["choices"][0]["message"]["content"]
                    except (KeyError, IndexError):
                        model_response = "Unable to extract model response"
                else:
                    # Handle standard response
                    try:
                        model_response = api_result["response"]["choices"][0]["message"]["content"]
                    except (KeyError, IndexError):
                        model_response = "Unable to extract model response"
            
            # Evaluate answer
            evaluation_result = question.evaluate_response(model_response)
            
            # Record results
            result = {
                "question_type": question_type,
                "prompt": prompt,
                "model_response": model_response,
                "api_result": api_result,
                **evaluation_result
            }
            
            # Add specific question type result fields
            for field in question.get_result_fields():
                if field in evaluation_result:
                    result[field] = evaluation_result[field]
            
            results.append(result)
            
            # Update total score
            total_score += evaluation_result.get("score", 0)
            total_possible += evaluation_result.get("total_possible", 0)
        
        # Calculate average score
        average_score = total_score / total_possible if total_possible > 0 else 0
        
        return {
            "model_name": model_config["name"],
            "dataset_name": dataset_name,
            "total_score": total_score,
            "total_possible": total_possible,
            "average_score": average_score,
            "results": results
        }

    def evaluate_and_save(self, model_config, test_data, dataset, timestamp):
        """线程任务：评测并保存结果"""
        model_results_dir = self.results_dir / model_config["name"] / "subjective"
        model_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Testing model {model_config['name']} on dataset {dataset}")
        results = self.evaluate_model(model_config, test_data, dataset)
        results_file = model_results_dir / f"{dataset.replace('.json', '')}_{timestamp}.json"
        with open(results_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Test results saved to {results_file}")

    def run_tests(self, model_name: Optional[str] = None, max_workers: int = 30):
        """多线程运行主入口"""
        test_datasets = [
            "blockchain-fundamental.json",
            "dao.json",
            "defi.json",
            "infrastructure.json",
            "meme.json",
            "nft.json",
            "tokenomics.json",
            "security.json",
            "smart-contract.json"
        ]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tasks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            if model_name:
                model_config = next((m for m in self.models if m["name"] == model_name), None)
                if not model_config:
                    print(f"Model {model_name} not found in configuration")
                    return
                for dataset in test_datasets:
                    test_data = self.load_test_data(dataset)
                    if not test_data:
                        print(f"No test data available for {dataset}")
                        continue
                    tasks.append(executor.submit(self.evaluate_and_save, model_config, test_data, dataset, timestamp))
            else:
                for model_config in self.models:
                    for dataset in test_datasets:
                        test_data = self.load_test_data(dataset)
                        if not test_data:
                            print(f"No test data available for {dataset}")
                            continue
                        tasks.append(executor.submit(self.evaluate_and_save, model_config, test_data, dataset, timestamp))
            for future in concurrent.futures.as_completed(tasks):
                try:
                    future.result()
                except Exception as exc:
                    print(f"{exc}")

def main():
    parser = argparse.ArgumentParser(description='Run subjective model tests')
    parser.add_argument('--model', type=str, help='Name of the model to test. If not specified, all models will be tested.')
    parser.add_argument('--threads', type=int, default=30, help='Number of threads to use for parallel testing.')
    args = parser.parse_args()
    
    config = load_config()
    tester = SubjectiveModelTester(config)
    tester.run_tests(args.model, max_workers=args.threads)

if __name__ == "__main__":
    main() 