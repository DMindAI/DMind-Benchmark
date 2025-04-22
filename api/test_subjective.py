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
            full_path = self.test_data_dir / "subjective" / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading test data: {e}")
            return {}
            
    def make_api_request(self, model_config: Dict, prompt: str) -> Dict:
        """Send API request"""
        Skey = model_config["api"]

        max_retries = 10  # Maximum retry attempts
        retry_delay = 15  # Retry interval (seconds)
        
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
                elif provider == "deepseek":
                    # Handle DeepSeek model requests using OpenAI client
                    print("\n" + "="*50)
                    print("Request content:")
                    print(f"DeepSeek API Request: model={model_config['model']}")
                    print(f"prompt: {prompt[:100]}...")
                    print("="*50 + "\n")
                    
                    try:
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
                        
                        # Convert OpenAI response object to dictionary
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
                    
                    data = {
                        "model": model_config["model"],
                        "messages": [{"role": "user", "content": prompt}],
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
                            print(f"解析响应失败，将在 {retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                else:
                    print(f"API请求失败")
                    if provider != "deepseek":
                        print(f"状态码: {response.status_code}")
                        print(f"响应内容: {response.text}")
                    if attempt < max_retries - 1:
                        print(f"将在 {retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                
            except Exception as e:
                print(f"发送API请求时出错: {e}")
                if attempt < max_retries - 1:
                    print(f"将在 {retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
        
        # 如果所有重试都失败了
        return {
            "status_code": 500,
            "response": None,
            "time_taken": end_time - start_time if 'end_time' in locals() else 0,
            "attempts": max_retries
        }
        
    def evaluate_model(self, model_config: Dict, test_data: List[Dict], dataset_name: str) -> Dict:
        """
        评估模型在测试数据上的表现
        
        Args:
            model_config: 模型配置
            test_data: 测试数据列表
            dataset_name: 数据集名称
            
        Returns:
            Dict: 评估结果
        """
        results = []
        total_score = 0
        total_possible = 0
        
        for question_data in test_data:
            question_type = question_data.get("question_type", "")
            
            # 获取对应的题目类型类
            question_class = QUESTION_TYPES.get(question_type)
            if not question_class:
                print(f"未知的题目类型: {question_type}")
                continue
                
            # 创建题目实例
            question = question_class(question_data)
            
            # 构建提示词
            prompt = question.build_prompt()
            
            # 调用模型API
            api_result = self.make_api_request(model_config, prompt)
            
            # 提取模型回答
            model_response = ""
            if api_result["status_code"] == 200:
                provider = model_config.get("provider", "").lower()
                if provider == "google":
                    # 处理 Gemini 响应
                    try:
                        if "candidates" in api_result["response"]:
                            model_response = api_result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            model_response = "无法提取模型回答"
                    except (KeyError, IndexError):
                        model_response = "无法提取模型回答"
                elif provider == "deepseek":
                    # 处理 DeepSeek 响应
                    try:
                        model_response = api_result["response"]["choices"][0]["message"]["content"]
                    except (KeyError, IndexError):
                        model_response = "无法提取模型回答"
                else:
                    # 处理标准响应
                    try:
                        model_response = api_result["response"]["choices"][0]["message"]["content"]
                    except (KeyError, IndexError):
                        model_response = "无法提取模型回答"
            
            # 评估回答
            evaluation_result = question.evaluate_response(model_response)
            
            # 记录结果
            result = {
                "question_type": question_type,
                "prompt": prompt,
                "model_response": model_response,
                "api_result": api_result,
                **evaluation_result
            }
            
            # 添加特定题目类型的结果字段
            for field in question.get_result_fields():
                if field in evaluation_result:
                    result[field] = evaluation_result[field]
            
            results.append(result)
            
            # 更新总分
            total_score += evaluation_result.get("score", 0)
            total_possible += evaluation_result.get("total_possible", 0)
        
        # 计算平均分
        average_score = total_score / total_possible if total_possible > 0 else 0
        
        return {
            "model_name": model_config["name"],
            "dataset_name": dataset_name,
            "total_score": total_score,
            "total_possible": total_possible,
            "average_score": average_score,
            "results": results
        }
        
    def run_tests(self, model_name: Optional[str] = None):
        """运行主观题测试
        Args:
            model_name: 可选，指定要测试的模型名称。如果为None，则测试所有模型
        """
        # 测试数据集列表
        test_datasets = [
            # "Blockchain_Fundamentals_benchmark.json",
            # "DAO.json",
            # "Defi.json",
            "Infra.json",
            "MEME.json",
            "NFT.json",
            "Token.json",
            "Security.json",
            "smart_contract.json"
        ]
        
        for dataset in test_datasets:
            test_data = self.load_test_data(dataset)
            if not test_data:
                print(f"No test data available for {dataset}")
                continue
                
            if model_name:
                # 测试指定模型
                model_config = next((m for m in self.models if m["name"] == model_name), None)
                if not model_config:
                    print(f"Model {model_name} not found in configuration")
                    return
                    
                # 创建模型专属的主观题结果目录
                model_results_dir = self.results_dir / model_config["name"] / "subjective"
                model_results_dir.mkdir(parents=True, exist_ok=True)
                    
                print(f"Testing model {model_config['name']} on dataset {dataset}")
                results = self.evaluate_model(model_config, test_data, dataset)
                
                # 保存结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = model_results_dir / f"{dataset.replace('.json', '')}_{timestamp}.json"
                with open(results_file, "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Test results saved to {results_file}")
            else:
                # 测试所有模型
                for model_config in self.models:
                    # 创建模型专属的主观题结果目录
                    model_results_dir = self.results_dir / model_config["name"] / "subjective"
                    model_results_dir.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Testing model {model_config['name']} on dataset {dataset}")
                    results = self.evaluate_model(model_config, test_data, dataset)
                    
                    # 保存结果
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_file = model_results_dir / f"{dataset.replace('.json', '')}_{timestamp}.json"
                    with open(results_file, "w", encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"Test results saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Run subjective model tests')
    parser.add_argument('--model', type=str, help='Name of the model to test. If not specified, all models will be tested.')
    args = parser.parse_args()
    
    config = load_config()
    tester = SubjectiveModelTester(config)
    tester.run_tests(args.model)

if __name__ == "__main__":
    main() 