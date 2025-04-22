from typing import Dict, List, Any, Optional
import requests
import json
import time
import logging
import os
from .base_question import BaseQuestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("strategy_analysis_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StrategyAnalysisQuestion")

class StrategyAnalysisQuestion(BaseQuestion):
    """Strategy analysis class, used to handle strategy analysis type questions"""
    
    def __init__(self, question_data: Dict[str, Any]):
        """
        Initialize strategy analysis question
        
        Args:
            question_data: Dictionary containing strategy analysis question data
        """
        super().__init__(question_data)
        self.question_type = "strategy_analysis"
        self.scenario = question_data.get("scenario", "")
        self.instructions = question_data.get("instructions", "")
        self.scoring_criteria = question_data.get("scoring_criteria", [])
        self.total_possible = question_data.get("total_possible", 10)
        self.keywords = question_data.get("keywords", {})  # List of keywords for each scoring criterion
        
        # Get API key from environment variable, use default if it doesn't exist
        self.third_party_api_key = os.environ.get("CLAUDE_API_KEY", "sk-sjkpMQ7WsWk5jUShcqhK4RSe3GEooupy8jsy7xQkbg6eQaaX")
        self.third_party_api_base = "https://api.claude-plus.top/v1/chat/completions"
        self.max_retries = 10  # Maximum retry attempts
        self.retry_delay = 2  # Retry interval (seconds)
        logger.info(f"Initializing strategy analysis question: {self.scenario[:50]}...")
        logger.info(f"Using API key: {self.third_party_api_key[:5]}...")
        
    def build_prompt(self) -> str:
        """
        Build prompt for strategy analysis question
        
        Returns:
            str: Built prompt
        """
        prompt = f"Scenario: {self.scenario}\n\n"
        prompt += f"Task: {self.instructions}\n\n"
        prompt += "Please provide detailed analysis and strategy recommendations."
        logger.info(f"Prompt building completed, length: {len(prompt)}")
        return prompt
    
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        Evaluate model's answer to strategy analysis question
        
        Args:
            response: Model's answer
            
        Returns:
            Dict[str, Any]: Evaluation results, including score and detailed information
        """
        logger.info(f"Starting answer evaluation, answer length: {len(response)}")
        
        # Use third-party AI for evaluation
        logger.info("Attempting to use third-party AI for evaluation...")
        third_party_evaluation = self._evaluate_with_third_party_ai(response)
        
        # Third-party AI evaluation will always return a result (success or keyword fallback)
        logger.info(f"Evaluation completed, total score: {third_party_evaluation.get('score', 0)}")
        return third_party_evaluation
    
    def _evaluate_with_third_party_ai(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Use third-party AI (Claude-3-7-Sonnet-20250219) to evaluate the answer
        
        Args:
            response: Model's answer
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results, returns None if evaluation fails
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                if retry_count > 0:
                    logger.info(f"Retry {retry_count} for third-party AI evaluation...")
                    time.sleep(self.retry_delay)  # Wait for a while before retrying
                
                logger.info("Starting to build third-party AI evaluation prompt...")
                # Build scoring criteria prompt
                criteria_prompt = "Please evaluate the quality of the answer according to the following scoring criteria:\n\n"
                for criterion in self.scoring_criteria:
                    criterion_name = criterion.get("criterion", "")
                    max_points = criterion.get("points", 0)
                    key_points = criterion.get("key_points", [])
                    
                    criteria_prompt += f"Criterion: {criterion_name} (Maximum: {max_points} points)\n"
                    criteria_prompt += "Key points:\n"
                    for point in key_points:
                        criteria_prompt += f"- {point}\n"
                    criteria_prompt += "\n"
                
                # Build complete evaluation prompt
                evaluation_prompt = f"""
                You are a professional evaluation expert, please evaluate the quality of the answer according to the following scoring criteria.
                
                Scenario: {self.scenario}
                
                Task: {self.instructions}
                
                Answer: {response}
                
                {criteria_prompt}
                
                Please provide a score of 0-10 for each scoring criterion, and explain your scoring rationale.
                Finally, please output the evaluation results in JSON format as follows:
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
                
                logger.info(f"Evaluation prompt building completed, length: {len(evaluation_prompt)}")
                
                # Call Claude API
                logger.info("Starting to call Claude API...")
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {self.third_party_api_key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": "claude-3-7-sonnet-20250219",
                    "messages": [{"role": "user", "content": evaluation_prompt}],
                    "max_tokens": 4000,
                    "temperature": 0
                }
                
                start_time = time.time()
                response_obj = requests.post(self.third_party_api_base, headers=headers, json=data)
                end_time = time.time()
                
                logger.info(f"API call completed, time taken: {end_time - start_time:.2f} seconds, status code: {response_obj.status_code}")
                
                if response_obj.status_code == 200:
                    response_data = response_obj.json()
                    logger.info(f"API response data: {json.dumps(response_data)[:200]}...")
                    
                    # Get answer from choices
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        evaluation_text = response_data["choices"][0]["message"]["content"]
                        logger.info(f"API return text length: {len(evaluation_text)}")
                        
                        # Extract JSON part
                        json_start = evaluation_text.find("{")
                        json_end = evaluation_text.rfind("}") + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            try:
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
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parsing failed: {str(e)}")
                                last_error = f"JSON parsing failed: {str(e)}"
                                # Continue to next retry
                        else:
                            logger.error("Cannot find JSON in API response")
                            last_error = "Cannot find JSON in API response"
                    else:
                        logger.error("API response does not contain choices field")
                        last_error = "API response format incorrect"
                else:
                    error_message = "Unknown error"
                    try:
                        error_data = response_obj.json()
                        if "error" in error_data:
                            error_message = error_data["error"].get("message", "Unknown error")
                            error_type = error_data["error"].get("type", "Unknown type")
                            logger.error(f"API call failed: {error_message} (type: {error_type})")
                    except:
                        logger.error(f"API call failed: {response_obj.text[:200]}...")
                    
                    last_error = f"API call failed: {response_obj.status_code} - {error_message}"
                    
                    # If it's an authentication error, try using a backup API key
                    if "Token not provided" in error_message or "authentication" in error_message.lower():
                        logger.warning("Authentication error detected, trying to use backup API key...")
                        # Here you can add logic for backup API key
                        # self.third_party_api_key = "Backup API key"
            
            except Exception as e:
                logger.error(f"Third-party AI evaluation failed: {str(e)}", exc_info=True)
                last_error = str(e)
            
            retry_count += 1
            if retry_count < self.max_retries:
                logger.info(f"Will retry in {self.retry_delay} seconds for {retry_count + 1}th attempt...")
        
        logger.error(f"Third-party AI evaluation failed, retried {retry_count} times, last error: {last_error}")
        # Return keyword matching result instead of None, ensure valid score even if retry fails
        return self._evaluate_with_keywords(response)
    
    def _evaluate_with_keywords(self, response: str) -> Dict[str, Any]:
        """
        Use keyword matching method to evaluate the answer (original evaluation logic)
        
        Args:
            response: Model's answer
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("Starting to use keyword matching method to evaluate the answer...")
        # Initialize results
        total_score = 0
        criterion_scores = []
        keyword_matches = {}
        
        # Evaluate each scoring criterion
        for criterion in self.scoring_criteria:
            criterion_name = criterion.get("criterion", "")
            max_points = criterion.get("points", 0)
            key_points = criterion.get("key_points", [])
            
            logger.info(f"Evaluating criterion: {criterion_name}, maximum points: {max_points}")
            
            # Get keyword list for this criterion
            criterion_keywords = self.keywords.get(criterion_name, [])
            
            # Calculate keyword match percentage
            keyword_score = 0
            matched_keywords = []
            
            if criterion_keywords:
                for keyword in criterion_keywords:
                    if keyword.lower() in response.lower():
                        keyword_score += 1
                        matched_keywords.append(keyword)
                
                # Keyword score accounts for 70% of total score
                keyword_score = (keyword_score / len(criterion_keywords)) * max_points * 0.7
                logger.info(f"Keyword match: {len(matched_keywords)}/{len(criterion_keywords)}, score: {keyword_score:.2f}")
            else:
                # If no keywords, evaluate based on key points
                key_points_score = 0
                for point in key_points:
                    if point.lower() in response.lower():
                        key_points_score += 1
                
                # Key points score accounts for 70% of total score
                keyword_score = (key_points_score / len(key_points)) * max_points * 0.7
                logger.info(f"Key point match: {key_points_score}/{len(key_points)}, score: {keyword_score:.2f}")
            
            # Calculate content quality score (accounts for 30% of total score)
            content_score = 0
            if len(response) > 100:  # Ensure answer has enough length
                content_score = max_points * 0.3
                logger.info(f"Content quality score: {content_score:.2f}")
            
            # Calculate total score for this criterion
            criterion_total_score = keyword_score + content_score
            logger.info(f"Criterion total score: {criterion_total_score:.2f}")
            
            # Add to results
            criterion_scores.append({
                "criterion": criterion_name,
                "score": criterion_total_score,
                "max_points": max_points,
                "matched_keywords": matched_keywords,
                "keyword_score": keyword_score,
                "content_score": content_score
            })
            
            total_score += criterion_total_score
        
        logger.info(f"Keyword matching evaluation completed, total score: {total_score:.2f}")
        
        # Build detailed debugging information
        debug_info = {
            "criterion_scores": criterion_scores,
            "total_score": total_score,
            "response_length": len(response),
            "evaluation_method": "keyword_matching"
        }
        
        return {
            "score": total_score,
            "total_possible": self.total_possible,
            "criterion_scores": criterion_scores,
            "debug_info": debug_info
        }
    
    def get_result_fields(self) -> List[str]:
        """
        Get fields to include in results
        
        Returns:
            List[str]: Field list
        """
        return ["score", "total_possible", "criterion_scores", "debug_info"] 