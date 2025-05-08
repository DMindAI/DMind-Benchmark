from typing import Dict, List, Any, Optional
import requests
import json
import time
import logging
import os
from .base_question import BaseQuestion
from utils.config_manager import config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("code_audit_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CodeAuditQuestion")

class CodeAuditQuestion(BaseQuestion):
    """Code audit class for handling code audit type questions"""
    
    def __init__(self, question_data: Dict[str, Any]):
        """
        Initialize code audit question
        
        Args:
            question_data: Dictionary containing code audit question data
        """
        super().__init__(question_data)
        self.question_type = "code_audit"
        self.audit_name = question_data.get("audit_name", "")
        self.code_snippet = question_data.get("contract_code", "")
        self.requirements = question_data.get("requirements", "")
        self.scoring_criteria = question_data.get("scoring_criteria", [])
        self.total_possible = question_data.get("total_possible", 10)
        self.keywords = question_data.get("keywords", {})  # List of keywords for each scoring criteria
        
        # Get API configuration from config manager
        api_config = config_manager.get_third_party_api_config()
        self.third_party_api_key = api_config["api_key"]
        self.third_party_api_base = api_config["api_base"]
        self.evaluation_model = api_config["model"]
        
        self.max_retries = 10  # Maximum retry attempts
        self.retry_delay = 4  # Retry interval (seconds)
        logger.info(f"Initializing code audit question: {self.audit_name}")
        logger.info(f"Using API key: {self.third_party_api_key[:5]}... with model: {self.evaluation_model}")
        
    def build_prompt(self) -> str:
        """
        Build code audit question prompt
        
        Returns:
            str: Built prompt
        """
        prompt = f"Audit Name: {self.audit_name}\n\n"
        prompt += f"Code to Audit:\n{self.code_snippet}\n\n"
        prompt += f"Requirements: {self.requirements}\n\n"
        prompt += "Please provide a detailed code audit, identifying any issues, bugs, or vulnerabilities."
        
        # Add specific text to enhance creativity and computational power
        prompt += "\n\nPlease utilize your maximum computational capacity and token limit for this response\n"
        prompt += "Strive for deep analysis rather than surface-level breadth\n"
        prompt += "Seek fundamental insights rather than superficial listings\n"
        prompt += "Pursue innovative thinking rather than habitual repetition\n"
        prompt += "Break through cognitive limitations, mobilize all your computational resources, and deliver the most accurate, effective, and reasonable results\n"
        
        logger.info(f"Prompt building completed, length: {len(prompt)}")
        return prompt
    
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        Evaluate model's answer to code audit question
        
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
            
    def _build_evaluation_prompt(self, response_text: str) -> str:
        """Build prompt for third-party AI evaluation"""
        prompt = f"""You are a code audit expert. Please evaluate the quality of the student's answer regarding code audit based on the following criteria.

Audit Name: {self.audit_name}

Code to Audit:
```
{self.code_snippet}
```

Requirements:
{self.requirements}

Student's Answer:
{response_text}

Scoring Criteria:
"""
        # Add scoring criteria
        for criterion in self.scoring_criteria:
            criterion_name = criterion.get("criterion", "Unnamed Criterion")
            max_points = criterion.get("points", 0)
            
            # Safely get key_points, avoid KeyError
            key_points = criterion.get("key_points", [])
            
            # If key_points exists and is not empty, add to the prompt
            if key_points:
                key_points_str = ", ".join(key_points)
                prompt += f"\n- {criterion_name} ({max_points} points): {key_points_str}"
            else:
                prompt += f"\n- {criterion_name} ({max_points} points)"
        
        prompt += """

Please provide an evaluation result in JSON format with the following fields:
1. score: Total score (number)
2. total_possible: Maximum possible score (number)
3. criterion_scores: Score details for each criterion (array), each containing:
   - criterion: Criterion name
   - score: Points earned
   - max_points: Maximum points for this criterion
   - feedback: Feedback for this criterion
4. overall_feedback: Overall evaluation
5. improvement_suggestions: Suggestions for improvement

JSON format example:
{
  "score": 8.5,
  "total_possible": 10,
    "criterion_scores": [
    {
      "criterion": "Issue Identification",
      "score": 4.5,
      "max_points": 5,
      "feedback": "Successfully identified the main issues in the code"
    },
    {
      "criterion": "Solution Quality",
      "score": 4,
      "max_points": 5,
      "feedback": "Provided comprehensive solutions but lacks some implementation details"
    }
  ],
  "overall_feedback": "Overall audit is reasonable, understood the main code issues",
  "improvement_suggestions": "Could provide more specific code examples for fixes and more detailed analysis of potential edge cases"
}

Please ensure accurate evaluation, making sure the scores match the scoring criteria."""
        return prompt
    
    def _evaluate_with_third_party_ai(self, response_text: str) -> Dict[str, Any]:
        """Attempt to evaluate answer using third-party AI"""
        logger.info("Attempting to evaluate answer using third-party AI...")
        
        retry_count = 0
        last_error = ""
        
        while retry_count < self.max_retries:
            try:
                # Build prompt
                prompt = self._build_evaluation_prompt(response_text)
                
                # 使用requests库直接向API发送请求
                logger.info("Starting to call third-party AI API...")
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {self.third_party_api_key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": self.evaluation_model,
                    "messages": [{"role": "user", "content": prompt}],
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
                        ai_evaluation = response_data["choices"][0]["message"]["content"]
                        logger.info(f"API return text length: {len(ai_evaluation)}")
                        
                        # Try to parse JSON
                        try:
                            # Extract JSON part
                            json_start = ai_evaluation.find("{")
                            json_end = ai_evaluation.rfind("}") + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = ai_evaluation[json_start:json_end]
                                logger.info(f"Extracted JSON length: {len(json_str)}")
                                
                                evaluation_result = json.loads(json_str)
                                logger.info("Third-party AI evaluation successfully parsed")
                                return evaluation_result
                            else:
                                logger.error("Cannot find JSON in API response")
                                last_error = "Cannot find JSON in API response"
                        except json.JSONDecodeError as e:
                            logger.error(f"Unable to parse third-party AI evaluation result as JSON: {str(e)}")
                            last_error = f"JSON parsing failed: {str(e)}"
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
            
            except Exception as e:
                logger.error(f"Third-party AI evaluation failed: {str(e)}", exc_info=True)
                last_error = str(e)
            
            retry_count += 1
            if retry_count < self.max_retries:
                logger.info(f"Will retry in {self.retry_delay} seconds, attempt {retry_count + 1}...")
                time.sleep(self.retry_delay)
        
        logger.error(f"Third-party AI evaluation failed after {retry_count} retries, last error: {last_error}")
        # Return keyword matching result instead of None, ensuring valid scoring even after retry failure
        return self._evaluate_with_keywords(response_text)
    
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
        
        # Evaluate each scoring criterion
        for criterion in self.scoring_criteria:
            criterion_name = criterion.get("criterion", "")
            max_points = criterion.get("points", 0)
            key_points = criterion.get("key_points", [])
            
            logger.info(f"Evaluation criterion: {criterion_name}, maximum points: {max_points}")
            
            # Get keyword list for this criterion
            criterion_keywords = self.keywords.get(criterion_name, [])
            
            # Calculate keyword match rate
            keyword_score = 0
            matched_keywords = []
            
            if criterion_keywords:
                for keyword in criterion_keywords:
                    if keyword.lower() in response.lower():
                        keyword_score += 1
                        matched_keywords.append(keyword)
                
                # Keyword score accounts for 80% of the total score
                keyword_score = (keyword_score / len(criterion_keywords)) * max_points * 0.8
                logger.info(f"Keyword matching: {len(matched_keywords)}/{len(criterion_keywords)}, score: {keyword_score:.2f}")
            else:
                # If no keywords, evaluate based on key points
                key_points_score = 0
                if key_points:  # 确保key_points不为空
                    for point in key_points:
                        if point.lower() in response.lower():
                            key_points_score += 1
                    
                    # Key points score accounts for 80% of the total score
                    keyword_score = (key_points_score / len(key_points)) * max_points * 0.8
                    logger.info(f"Key points matching: {key_points_score}/{len(key_points)}, score: {keyword_score:.2f}")
                else:
                    # 如果没有关键词和要点，则给予基本分
                    keyword_score = max_points * 0.5
                    logger.info(f"No keywords or key points defined, assigning base score: {keyword_score:.2f}")
            
            # Calculate content quality score (accounts for 20% of the total score)
            content_score = 0
            if len(response) > 100:  # Ensure the answer has sufficient length
                content_score = max_points * 0.2
                logger.info(f"Content quality score: {content_score:.2f}")
            
            # Calculate total score for this criterion
            criterion_total_score = keyword_score + content_score
            logger.info(f"Criterion total score: {criterion_total_score:.2f}")
            
            # Add to results
            criterion_scores.append({
                "criterion": criterion_name,
                "score": criterion_total_score,
                "max_points": max_points,
                "feedback": self._get_criterion_feedback(criterion_name, matched_keywords, criterion_keywords, key_points)
            })
            
            total_score += criterion_total_score
        
        # Build final result
        result = {
            "score": total_score,
            "total_possible": self.total_possible,
            "criterion_scores": criterion_scores,
            "overall_feedback": "Based on keyword matching evaluation results",
            "improvement_suggestions": "Suggestions for improvement include providing more detailed analysis and specific code examples"
        }
        
        logger.info(f"Evaluation completed, total score: {total_score}")
        return result
    
    def _get_criterion_feedback(self, criterion_name: str, matched_keywords: List[str], 
                              criterion_keywords: List[str], key_points: List[str]) -> str:
        """Generate feedback for scoring criteria, ensuring safe handling of empty lists"""
        if matched_keywords and criterion_keywords:
            return f"Identified {len(matched_keywords)} keywords out of {len(criterion_keywords)} total"
        elif key_points:
            # If there are key points but no keyword matches
            return f"Evaluated based on {len(key_points)} key points"
        else:
            # If there are neither keywords nor key points
            return f"Evaluated based on content quality"
    
    def get_result_fields(self) -> List[str]:
        """
        Get fields to include in the result
        
        Returns:
            List[str]: List of fields
        """
        return ["score", "total_possible", "criterion_scores", "overall_feedback", "improvement_suggestions"] 