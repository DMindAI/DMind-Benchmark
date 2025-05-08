from typing import Dict, List, Any, Optional
import requests
import json
import time
import logging
import os
import subprocess
import tempfile
from .base_question import BaseQuestion
from utils.config_manager import config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scenario_analysis_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ScenarioAnalysisQuestion")

class ScenarioAnalysisQuestion(BaseQuestion):
    """Scenario analysis class for handling scenario-based questions"""
    
    def __init__(self, question_data: Dict[str, Any]):
        """
        Initialize scenario analysis question
        
        Args:
            question_data: Dictionary containing scenario analysis question data
        """
        super().__init__(question_data)
        self.question_type = "scenario_analysis"
        self.scenario = question_data.get("scenario", "")
        self.requirements = question_data.get("requirements", [])
        self.scoring_criteria = question_data.get("scoring_criteria", [])
        self.reference_solution = question_data.get("reference_solution", "")
        
        # Calculate total_possible from scoring criteria
        total_points = 0
        for criterion in self.scoring_criteria:
            total_points += criterion.get("points", 0)
        self.total_possible = question_data.get("total_possible", total_points)
        
        # 从配置管理器获取API配置
        api_config = config_manager.get_third_party_api_config()
        self.third_party_api_key = api_config["api_key"]
        self.third_party_api_base = api_config["api_base"]
        self.third_party_model = api_config["model"]
        self.max_retries = 10  # Maximum retry attempts
        self.retry_delay = 2  # Retry interval (seconds)
        
        logger.info(f"Initializing scenario analysis question: {len(self.scenario)} characters")
        logger.info(f"Using API key: {self.third_party_api_key[:5]}...")
        logger.info(f"Using API endpoint: {self.third_party_api_base}")
        
    def build_prompt(self) -> str:
        """
        Build scenario analysis question prompt
        
        Returns:
            str: Built prompt
        """
        prompt = "Please analyze the following scenario and provide a comprehensive solution:\n\n"
        prompt += f"Scenario:\n{self.scenario}\n\n"
        
        if self.requirements:
            prompt += "Requirements:\n"
            for i, req in enumerate(self.requirements, 1):
                prompt += f"{i}. {req}\n"
            prompt += "\n"
        
        prompt += "Please provide a detailed analysis and solution for this scenario."
        logger.info(f"Prompt building completed, length: {len(prompt)}")
        return prompt
    
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        Evaluate model's answer to scenario analysis question
        
        Args:
            response: Model's answer
            
        Returns:
            Dict[str, Any]: Evaluation results, including score and detailed information
        """
        logger.info(f"Starting answer evaluation, answer length: {len(response)}")
        
        # Try to use third-party AI for evaluation
        logger.info("Attempting to use third-party AI for evaluation...")
        third_party_evaluation = self._evaluate_with_third_party_ai(response)
        
        # If third-party AI evaluation fails, use keyword matching method
        if not third_party_evaluation:
            logger.info("Third-party AI evaluation failed, using keyword matching method...")
            return self._evaluate_with_keywords(response)
        
        logger.info(f"Evaluation completed, total score: {third_party_evaluation.get('score', 0)}")
        return third_party_evaluation
    
    def _evaluate_with_third_party_ai(self, response_text: str) -> Dict[str, Any]:
        """
        Use third-party AI to evaluate the answer
        
        Args:
            response_text: Model's answer
            
        Returns:
            Dict[str, Any]: Evaluation results, None if evaluation fails
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                if retry_count > 0:
                    logger.info(f"Retry {retry_count} for third-party AI evaluation...")
                    time.sleep(self.retry_delay)  # Wait for a while before retrying
                
                logger.info("Starting to build third-party AI evaluation prompt...")
                
                # Build evaluation criteria prompt
                criteria_prompt = "Please evaluate the response according to the following criteria:\n"
                for criterion in self.scoring_criteria:
                    criterion_name = criterion.get("criterion", "")
                    max_points = criterion.get("points", 0)
                    description = criterion.get("description", "")
                    criteria_prompt += f"- {criterion_name} ({max_points} points): {description}\n"
                
                # Build complete evaluation prompt
                evaluation_prompt = f"""
                You are a professional scenario analysis evaluator. Please evaluate the quality of this analysis.
                
                Original scenario:
                {self.scenario}
                
                Requirements:
                {', '.join(self.requirements)}
                
                Reference solution:
                {self.reference_solution}
                
                Model's Answer: {response_text}
                
                {criteria_prompt}
                
                For each scoring criterion, evaluate how well the answer performed and assign a score.
                
                Output the evaluation results in the following JSON format:
                {{
                    "criterion_scores": [
                        {{
                            "criterion": "Criterion name",
                            "score": score,
                            "max_points": maximum points,
                            "explanation": "Scoring rationale"
                        }},
                        ...
                    ],
                    "total_score": total score,
                    "total_possible": {self.total_possible},
                    "overall_feedback": "Overall evaluation"
                }}
                
                Only output the evaluation results in JSON format, without any other content.
                """
                
                logger.info("Starting to call third-party AI API...")
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {self.third_party_api_key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": self.third_party_model,
                    "messages": [{"role": "user", "content": evaluation_prompt}],
                    "max_tokens": 4000,
                    "temperature": 0
                }
                
                start_time = time.time()
                
                try:
                    # Try to use requests to send request
                    response_obj = requests.post(self.third_party_api_base, headers=headers, json=data)
                    end_time = time.time()
                    
                    logger.info(f"API call completed, time taken: {end_time - start_time:.2f} seconds, status code: {response_obj.status_code}")
                    
                    if response_obj.status_code != 200:
                        error_msg = f"API call failed, status code: {response_obj.status_code}, trying to use curl as fallback"
                        logger.warning(error_msg)
                        raise Exception(error_msg)
                    
                    response_data = response_obj.json()
                    
                except Exception as e:
                    # If requests fails, try using curl
                    logger.info(f"Using requests to call API failed: {str(e)}, trying to use curl...")
                    
                    # Write data to temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                        json.dump(data, temp_file)
                        temp_file_path = temp_file.name
                    
                    # Build curl command
                    curl_cmd = [
                        'curl', '-s', self.third_party_api_base,
                        '-H', f'Authorization: Bearer {self.third_party_api_key}',
                        '-H', 'Content-Type: application/json',
                        '-H', 'Accept: application/json',
                        '-H', 'User-Agent: Apifox/1.0.0 (https://apifox.com)',
                        '-X', 'POST',
                        '-d', f'@{temp_file_path}'
                    ]
                    
                    # Execute curl command
                    try:
                        curl_result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
                        end_time = time.time()
                        logger.info(f"curl API call completed, time taken: {end_time - start_time:.2f} seconds")
                        
                        # Parse response
                        try:
                            response_data = json.loads(curl_result.stdout)
                            
                            # Create an object similar to requests.Response
                            class CurlResponse:
                                def __init__(self, data, status_code=200):
                                    self.data = data
                                    self.status_code = status_code
                                
                                def json(self):
                                    return self.data
                            
                            response_obj = CurlResponse(response_data)
                            
                        except json.JSONDecodeError as je:
                            logger.error(f"Failed to parse curl response: {str(je)}")
                            logger.error(f"curl response: {curl_result.stdout[:200]}")
                            logger.error(f"curl error: {curl_result.stderr}")
                            raise je
                        
                        # Delete temporary file
                        os.unlink(temp_file_path)
                        
                    except subprocess.CalledProcessError as ce:
                        logger.error(f"Failed to execute curl command: {str(ce)}")
                        logger.error(f"curl error output: {ce.stderr}")
                        # Delete temporary file
                        os.unlink(temp_file_path)
                        raise ce
                
                logger.info(f"API response data: {json.dumps(response_data)[:200]}...")
                
                if "choices" not in response_data or not response_data["choices"]:
                    error_msg = "API response does not contain choices field"
                    logger.error(error_msg)
                    last_error = Exception(error_msg)
                    retry_count += 1
                    continue
                
                evaluation_text = response_data["choices"][0]["message"]["content"]
                logger.info(f"Evaluation text length: {len(evaluation_text)}")
                
                # Try to extract JSON from evaluation text
                try:
                    # Find start and end positions of JSON string
                    json_start = evaluation_text.find("{")
                    json_end = evaluation_text.rfind("}") + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = evaluation_text[json_start:json_end]
                        logger.info(f"Extracted JSON length: {len(json_str)}")
                        
                        evaluation_result = json.loads(json_str)
                        
                        # Check if the returned total score is 0 (might be an error in scoring)
                        total_score = evaluation_result.get('total_score', 0)
                        if total_score == 0 and retry_count == 0:
                            # First attempt got 0 points, log a warning and continue
                            logger.warning("API returned a total score of 0, this might be a scoring error. Checking scoring criteria...")
                            
                            # Check scores for each criterion
                            criterion_scores = evaluation_result.get('criterion_scores', [])
                            all_zeros = all(item.get('score', 0) == 0 for item in criterion_scores)
                            
                            if all_zeros and len(criterion_scores) > 0:
                                logger.warning("All scoring criteria are 0 points, might be an API scoring error. Will retry...")
                                raise ValueError("API returned all-zero scores, might be a scoring error")
                        
                        logger.info(f"JSON parsing successful, total score: {total_score}")
                        
                        # Add debugging information
                        evaluation_result["debug_info"] = {
                            "evaluation_method": "third_party_ai",
                            "api_response_time": end_time - start_time,
                            "retry_count": retry_count
                        }
                        
                        # Change total_score to score
                        if "total_score" in evaluation_result:
                            evaluation_result["score"] = evaluation_result.pop("total_score")
                        
                        return evaluation_result
                    else:
                        logger.error("Cannot find JSON in API response")
                        last_error = Exception("Cannot find JSON in API response")
                        retry_count += 1
                        continue
                    
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parsing failed: {str(e)}"
                    logger.error(error_msg)
                    last_error = e
                    retry_count += 1
                    continue
                    
            except Exception as e:
                error_msg = f"Error occurred during evaluation: {str(e)}"
                logger.error(error_msg)
                last_error = e
                retry_count += 1
                continue
        
        if last_error:
            logger.error(f"Evaluation failed, last error: {str(last_error)}")
        
        return None
    
    def _evaluate_with_keywords(self, response: str) -> Dict[str, Any]:
        """
        Use keyword matching method to evaluate the answer
        
        Args:
            response: Model's answer
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("Starting to use keyword matching method to evaluate the answer...")
        
        # Initialize results
        total_score = 0
        criterion_scores = []
        
        # Check for reference solution keywords in the response
        if self.reference_solution:
            # Simple content analysis
            reference_words = set(self.reference_solution.lower().split())
            response_words = set(response.lower().split())
            common_words = reference_words.intersection(response_words)
            
            # Calculate similarity percentage
            similarity = len(common_words) / len(reference_words) if len(reference_words) > 0 else 0
            logger.info(f"Content similarity: {similarity:.2%} ({len(common_words)}/{len(reference_words)} words in common)")
        else:
            similarity = 0.5  # Default similarity if no reference solution
        
        # Evaluate based on scoring criteria
        for criterion in self.scoring_criteria:
            criterion_name = criterion.get("criterion", "")
            max_points = criterion.get("points", 0)
            
            # Basic scoring - assign scores based on similarity and response length
            response_length_factor = min(1.0, len(response) / 1000)  # Normalize by expected length
            
            # Combine similarity and length factor for scoring
            score = ((similarity * 0.7) + (response_length_factor * 0.3)) * max_points
            
            logger.info(f"{criterion_name} score: {score:.2f}/{max_points}")
            
            # Add criterion score to results
            criterion_scores.append({
                "criterion": criterion_name,
                "score": score,
                "max_points": max_points,
                "explanation": f"Score based on content similarity ({similarity:.2%}) and response length."
            })
            
            # Add to total score
            total_score += score
        
        logger.info(f"Keyword matching evaluation completed, total score: {total_score:.2f}/{self.total_possible}")
        
        # Build debugging information
        debug_info = {
            "evaluation_method": "keyword_matching",
            "content_similarity": similarity,
            "response_length": len(response),
            "reference_length": len(self.reference_solution) if self.reference_solution else 0
        }
        
        # Build final results
        evaluation_result = {
            "score": total_score,
            "total_possible": self.total_possible,
            "overall_feedback": f"Evaluation based on content similarity with reference solution ({similarity:.2%}).",
            "criterion_scores": criterion_scores,
            "debug_info": debug_info
        }
        
        return evaluation_result
    
    def get_result_fields(self) -> List[str]:
        """
        Get fields to include in results
        
        Returns:
            List[str]: Field list
        """
        return ["score", "total_possible", "criterion_scores", "debug_info"] 