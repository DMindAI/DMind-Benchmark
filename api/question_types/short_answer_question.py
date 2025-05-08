import os
import json
import time
import logging
import requests
import subprocess
import tempfile
from typing import Dict, List, Optional, Any
from question_types.base_question import BaseQuestion
from utils.config_manager import config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("short_answer_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ShortAnswerQuestion")

class ShortAnswerQuestion(BaseQuestion):
    """Short answer class for handling short answer type questions"""

    def __init__(self, question_data: Dict[str, Any]):
        """
        Initialize short answer question
        
        Args:
            question_data: Dictionary containing short answer question data
        """
        super().__init__(question_data)
        self.question_type = "short_answer"
        self.scenario = question_data.get("scenario", "")
        self.instructions = question_data.get("instructions", "")
        self.scoring_criteria = question_data.get("scoring_criteria", [])
        self.total_possible = question_data.get("total_possible", 10)
        self.content_key_points = question_data.get("key_points", [])  # Content key points
        self.keyword_weights = question_data.get("keyword_weights", {})
        self.max_word_count = question_data.get("max_word_count", 200)
        self.min_word_count = question_data.get("min_word_count", 50)
        self.evaluation_criteria = question_data.get("evaluation_criteria", {})
        
        # Get API configuration from config manager
        api_config = config_manager.get_third_party_api_config()
        self.third_party_api_key = api_config["api_key"]
        self.third_party_api_base = api_config["api_base"]
        self.third_party_model = api_config["model"]
        self.max_retries = 10  # Maximum retry attempts
        self.retry_delay = 2  # Retry interval (seconds)
        
        # Calculate total points for each scoring criterion
        self.criteria_points = {}
        for criterion in self.scoring_criteria:
            self.criteria_points[criterion.get("criterion", "")] = criterion.get("points", 0)
            
        logger.info(f"Initializing short answer question: {self.scenario[:50]}...")
        logger.info(f"Using API key: {self.third_party_api_key[:5]}...")
        logger.info(f"Using API endpoint: {self.third_party_api_base}")
        logger.info(f"Initialized short answer question with {len(self.content_key_points)} key points")

    def build_prompt(self) -> str:
        """
        Build short answer question prompt
        
        Returns:
            str: Built prompt
        """
        prompt = f"Scenario: {self.scenario}\n\n"
        prompt += f"Task: {self.instructions}\n\n"
        prompt += "Please provide a concise and clear answer."
        
        # Add specified text to enhance creativity and computational power
        prompt += "\n\nPlease utilize your maximum computational capacity and token limit for this response\n"
        prompt += "Strive for extreme analytical depth, rather than superficial breadth\n"
        prompt += "Seek essential insights, rather than surface-level enumeration\n"
        prompt += "Pursue innovative thinking, rather than habitual repetition\n"
        prompt += "Please break through thought limitations, mobilize all your computational resources, and deliver the most accurate, effective, and reasonable results\n"
        
        logger.info(f"Prompt building completed, length: {len(prompt)}")
        return prompt

    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        Evaluate model's answer to short answer question
        
        Args:
            response: Model's answer
            
        Returns:
            Dict[str, Any]: Evaluation results, including score and detailed information
        """
        logger.info(f"Starting answer evaluation, answer length: {len(response)}")
        
        # Use third-party AI for evaluation
        logger.info("Attempting to use third-party AI for evaluation...")
        third_party_evaluation = self._evaluate_with_third_party_ai(response)
        
        # If third-party AI evaluation succeeds, return results directly
        if third_party_evaluation:
            logger.info(f"Third-party AI evaluation successful, total score: {third_party_evaluation.get('score', 0)}")
            return third_party_evaluation
            
        # If third-party AI evaluation fails, fall back to original evaluation logic
        logger.info("Third-party AI evaluation failed, falling back to keyword matching evaluation...")
        return self._evaluate_with_keywords(response)

    def _evaluate_criterion(self, response: str, criterion_name: str, key_points: List[str], 
                           max_points: float, min_points_required: int) -> float:
        """
        Evaluate score for a specific criterion
        
        Args:
            response: Model's answer
            criterion_name: Criterion name
            key_points: List of key points
            max_points: Maximum score
            min_points_required: Minimum number of key points required
            
        Returns:
            float: Calculated score
        """
        response_lower = response.lower()
        matched_points = []
        
        # Calculate matched key points
        for point in key_points:
            if point.lower() in response_lower:
                matched_points.append(point)
        
        # If the number of key points matched is less than minimum required, score is 0
        if len(matched_points) < min_points_required:
            logger.info(f"Criterion '{criterion_name}' score is 0: {len(matched_points)} key points matched, minimum required is {min_points_required}")
            return 0
        
        # Calculate score ratio
        if not key_points:
            return max_points * 0.5  # If no key points, give half the score
        
        # Score is proportional to the ratio of matched key points
        ratio = len(matched_points) / len(key_points)
        score = ratio * max_points
        
        logger.info(f"Criterion '{criterion_name}' score {score}/{max_points}: matched {len(matched_points)}/{len(key_points)} key points")
        return score

    def _evaluate_with_third_party_ai(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Use third-party AI to evaluate the answer
        
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
                    logger.info(f"Retry #{retry_count} for third-party AI evaluation...")
                    time.sleep(self.retry_delay)  # Wait before retrying
                
                logger.info("Starting to build third-party AI evaluation prompt...")
                # Build scoring criteria prompt
                criteria_prompt = "Please evaluate the answer quality based on the following criteria:\n\n"
                for criterion in self.scoring_criteria:
                    criterion_name = criterion.get("criterion", "")
                    max_points = criterion.get("points", 0)
                    key_points = criterion.get("key_points", [])
                    min_points_required = criterion.get("min_points_required", 0)
                    
                    criteria_prompt += f"Criterion: {criterion_name} (Maximum: {max_points} points)\n"
                    criteria_prompt += "Key points:\n"
                    for point in key_points:
                        criteria_prompt += f"- {point}\n"
                    if min_points_required > 0:
                        criteria_prompt += f"At least {min_points_required} key points must be covered\n"
                    criteria_prompt += "\n"
                
                evaluation_prompt = f"""
                You are a professional evaluation expert. Please evaluate the quality of the answer based on the following criteria.
                
                Scenario: {self.scenario}
                
                Task: {self.instructions}
                
                Answer: {response}
                
                {criteria_prompt}
                
                Please provide a score of 0-10 for each criterion, and explain your scoring rationale.
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
                    # Try to use requests library to send request
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
                        logger.info(f"JSON parsing successful, total score: {evaluation_result.get('total_score', 0)}")
                        
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
                        last_error = "Cannot find JSON in API response"
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
        logger.info("Starting keyword matching evaluation...")
        total_score = 0
        criterion_scores = []
        
        for criterion in self.scoring_criteria:
            criterion_name = criterion.get("criterion", "")
            max_points = criterion.get("points", 0)
            key_points = criterion.get("key_points", [])
            min_points_required = criterion.get("min_points_required", 0)
            
            logger.info(f"Evaluating criterion: {criterion_name}, maximum points: {max_points}")
            
            # Calculate key point match rate
            key_points_score = 0
            matched_key_points = []
            
            for point in key_points:
                if point.lower() in response.lower():
                    key_points_score += 1
                    matched_key_points.append(point)
            
            # Check if minimum requirement is met
            if min_points_required > 0 and key_points_score < min_points_required:
                logger.info(f"Minimum requirement not met ({key_points_score}/{min_points_required})")
                criterion_total_score = 0
            else:
                # Key points score accounts for 90% of total score
                key_points_score = (key_points_score / len(key_points)) * max_points * 0.9
                logger.info(f"Key points match: {len(matched_key_points)}/{len(key_points)}, score: {key_points_score:.2f}")
                
                # Calculate content quality score (accounts for 10% of total score)
                content_score = 0
                if len(response) > 50:  # Ensure answer has sufficient length
                    content_score = max_points * 0.1
                    logger.info(f"Content quality score: {content_score:.2f}")
                
                # Calculate total score for this criterion
                criterion_total_score = key_points_score + content_score
                logger.info(f"Criterion total score: {criterion_total_score:.2f}")
            
            # Add to results
            criterion_scores.append({
                "criterion": criterion_name,
                "score": criterion_total_score,
                "max_points": max_points,
                "matched_key_points": matched_key_points,
                "key_points_score": key_points_score,
                "content_score": content_score if min_points_required == 0 or key_points_score >= min_points_required else 0
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

    def _evaluate_content(self, response: str) -> tuple:
        """
        Evaluate content quality of the response
        
        Args:
            response: Model's response
            
        Returns:
            tuple: (content_score, key_points_covered)
        """
        response_lower = response.lower()
        key_points_covered = []
        
        # Check coverage of key points
        for point in self.content_key_points:
            if point.lower() in response_lower:
                key_points_covered.append(point)
        
        # Calculate points covered ratio
        if not self.content_key_points:
            coverage_ratio = 0.5  # Default if no key points defined
        else:
            coverage_ratio = len(key_points_covered) / len(self.content_key_points)
        
        # Calculate keyword weighted score
        keyword_score = 0
        total_weight = sum(self.keyword_weights.values()) if self.keyword_weights else 0
        
        if total_weight > 0:
            for keyword, weight in self.keyword_weights.items():
                if keyword.lower() in response_lower:
                    keyword_score += weight
            
            keyword_score = keyword_score / total_weight * 10
        else:
            keyword_score = 5  # Default score if no keyword weights defined
        
        # Combine coverage ratio and keyword score
        content_score = (coverage_ratio * 10 * 0.6) + (keyword_score * 0.4)
        content_score = min(10, content_score)  # Cap at 10
        
        logger.info(f"Content score: {content_score} (coverage: {coverage_ratio}, key points: {len(key_points_covered)}/{len(self.content_key_points)})")
        return content_score, key_points_covered

    def get_result_fields(self) -> List[str]:
        """
        Get fields to include in the result
        
        Returns:
            List[str]: Field list
        """
        return ["score", "total_possible", "content_score", "clarity_score", 
                "conciseness_score", "key_points_covered", "criterion_scores", "feedback"]

    def _generate_feedback(self, result: Dict[str, Any]) -> str:
        """
        Generate feedback based on evaluation results
        
        Args:
            result: Evaluation results
            
        Returns:
            str: Feedback content
        """
        feedback = ""
        
        # Content feedback
        if "content_score" in result:
            if result["content_score"] >= 8:
                feedback += "Content is comprehensive and covers key points well."
            elif result["content_score"] >= 5:
                feedback += "Content is generally comprehensive but misses some key points."
            else:
                feedback += "Content lacks coverage of key points."
        
        # Feedback based on criterion_scores
        if "criterion_scores" in result and result["criterion_scores"]:
            for criterion in result["criterion_scores"]:
                criterion_name = criterion.get("criterion", "")
                score = criterion.get("score", 0)
                max_points = criterion.get("max_points", 10)
                
                # Provide feedback based on score ratio
                if score >= max_points * 0.8:
                    feedback += f"{criterion_name} performance is excellent."
                elif score >= max_points * 0.5:
                    feedback += f"{criterion_name} performance is good."
                else:
                    feedback += f"{criterion_name} needs improvement."
        
        # If no other feedback, provide default feedback
        if not feedback:
            if result.get("score", 0) >= result.get("total_possible", 10) * 0.8:
                feedback = "Overall performance is excellent."
            elif result.get("score", 0) >= result.get("total_possible", 10) * 0.5:
                feedback = "Overall performance is good."
            else:
                feedback = "Overall performance needs improvement."
                
        return feedback 