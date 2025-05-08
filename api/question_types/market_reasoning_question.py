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
        logging.FileHandler("market_reasoning_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketReasoningQuestion")

class MarketReasoningQuestion(BaseQuestion):
    """Market reasoning question class for evaluating analysis of market scenarios and trends"""
    
    def __init__(self, question_data: Dict[str, Any]):
        """
        Initialize market reasoning question
        
        Args:
            question_data: Dictionary containing question data
        """
        super().__init__(question_data)
        self.question_type = "market_reasoning"
        self.market_data = question_data.get("market_data", {})
        self.key_factors = question_data.get("key_factors", [])
        self.expected_insights = question_data.get("expected_insights", [])
        self.scenario = question_data.get("scenario", "")
        logger.info(f"Initialized market reasoning question with {len(self.key_factors)} key factors and {len(self.expected_insights)} expected insights")
        
        # Get API configuration from config manager
        api_config = config_manager.get_third_party_api_config()
        self.third_party_api_key = api_config["api_key"]
        self.third_party_api_base = api_config["api_base"]
        self.evaluation_model = api_config["model"]
        
        self.max_retries = 10  # Maximum retry attempts
        self.retry_delay = 2  # Retry interval (seconds)
        logger.info(f"Initializing market reasoning question, scenario length: {len(self.scenario)}")
        logger.info(f"Using API key: {self.third_party_api_key[:5]}... with model: {self.evaluation_model}")
        
    def build_prompt(self) -> str:
        """
        Build market reasoning question prompt
        
        Returns:
            str: Built prompt
        """
        prompt = f"Market Scenario: {self.scenario}\n\n"
        
        if self.market_data:
            prompt += "Market Data:\n"
            for key, value in self.market_data.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        prompt += "Please analyze the market scenario and provide:\n"
        prompt += "1. Key market trends and their implications\n"
        prompt += "2. Main factors influencing the market\n"
        prompt += "3. Potential opportunities and threats\n"
        prompt += "4. Recommended market strategy based on the analysis\n\n"
        
        logger.info(f"Prompt built with length: {len(prompt)}")
        return prompt
    
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        Evaluate model's answer to market reasoning question
        
        Args:
            response: Model's response to evaluate
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info(f"Evaluating market reasoning response of length: {len(response)}")
        
        # Initialize result structure
        result = {
            "score": 0,
            "max_score": 10,
            "factor_coverage": 0,
            "insight_coverage": 0,
            "reasoning_quality": 0,
            "strategy_quality": 0,
            "feedback": ""
        }
        
        # Evaluate coverage of key factors (30% of total score)
        factor_coverage = self._evaluate_factor_coverage(response)
        result["factor_coverage"] = factor_coverage
        
        # Evaluate coverage of expected insights (30% of total score)
        insight_coverage = self._evaluate_insight_coverage(response)
        result["insight_coverage"] = insight_coverage
        
        # Evaluate quality of reasoning (20% of total score)
        reasoning_quality = self._evaluate_reasoning_quality(response)
        result["reasoning_quality"] = reasoning_quality
        
        # Evaluate quality of strategy recommendations (20% of total score)
        strategy_quality = self._evaluate_strategy_quality(response)
        result["strategy_quality"] = strategy_quality
        
        # Calculate overall score
        result["score"] = (
            factor_coverage * 3 +
            insight_coverage * 3 +
            reasoning_quality * 2 +
            strategy_quality * 2
        ) / 10
        
        # Generate feedback
        result["feedback"] = self._generate_feedback(result)
        
        logger.info(f"Evaluation completed. Final score: {result['score']}/{result['max_score']}")
        return result
    
    def _evaluate_factor_coverage(self, response: str) -> float:
        """
        Evaluate coverage of key factors in the response
        
        Args:
            response: Model's response
            
        Returns:
            float: Factor coverage score (0-10)
        """
        if not self.key_factors:
            return 5  # Default score if no key factors defined
        
        response_lower = response.lower()
        covered_factors = 0
        
        for factor in self.key_factors:
            if factor.lower() in response_lower:
                covered_factors += 1
        
        coverage_ratio = covered_factors / len(self.key_factors)
        score = min(10, coverage_ratio * 10)
        
        logger.info(f"Factor coverage: {covered_factors}/{len(self.key_factors)} factors mentioned, score: {score}")
        return score
    
    def _evaluate_insight_coverage(self, response: str) -> float:
        """
        Evaluate coverage of expected insights in the response
        
        Args:
            response: Model's response
            
        Returns:
            float: Insight coverage score (0-10)
        """
        if not self.expected_insights:
            return 5  # Default score if no expected insights defined
        
        response_lower = response.lower()
        covered_insights = 0
        
        for insight in self.expected_insights:
            if insight.lower() in response_lower:
                covered_insights += 1
        
        coverage_ratio = covered_insights / len(self.expected_insights)
        score = min(10, coverage_ratio * 10)
        
        logger.info(f"Insight coverage: {covered_insights}/{len(self.expected_insights)} insights mentioned, score: {score}")
        return score
    
    def _evaluate_reasoning_quality(self, response: str) -> float:
        """
        Evaluate quality of reasoning in the response
        
        Args:
            response: Model's response
            
        Returns:
            float: Reasoning quality score (0-10)
        """
        # Simple evaluation based on response length and structure
        # In a real implementation, this would use more sophisticated NLP techniques
        
        # Check for reasoning indicators
        reasoning_indicators = [
            "because", "due to", "as a result", "therefore", "consequently",
            "implies", "suggests", "indicates", "leads to", "results in"
        ]
        
        indicator_count = sum(response.lower().count(indicator) for indicator in reasoning_indicators)
        
        # Normalize by response length
        normalized_count = min(10, indicator_count * 100 / len(response)) if response else 0
        
        # Check for response structure (paragraphs, sections)
        paragraphs = [p for p in response.split("\n\n") if p.strip()]
        structure_score = min(10, len(paragraphs) * 2)
        
        # Combine scores
        score = (normalized_count * 0.6) + (structure_score * 0.4)
        
        logger.info(f"Reasoning quality score: {score} (indicator count: {indicator_count}, paragraphs: {len(paragraphs)})")
        return score
    
    def _evaluate_strategy_quality(self, response: str) -> float:
        """
        Evaluate quality of strategy recommendations in the response
        
        Args:
            response: Model's response
            
        Returns:
            float: Strategy quality score (0-10)
        """
        # Check for strategy section
        strategy_section = ""
        
        response_lower = response.lower()
        strategy_keywords = ["strategy", "recommendation", "approach", "action plan"]
        
        for keyword in strategy_keywords:
            if keyword in response_lower:
                # Find paragraph containing strategy keyword
                paragraphs = response.split("\n\n")
                for paragraph in paragraphs:
                    if keyword in paragraph.lower():
                        strategy_section = paragraph
                        break
                
                if strategy_section:
                    break
        
        if not strategy_section:
            logger.info("No clear strategy section found in response")
            return 3  # Low score if no strategy section found
        
        # Evaluate strategy specificity
        specificity_indicators = [
            "specifically", "particular", "exact", "precise",
            "detailed", "concrete", "clear", "defined"
        ]
        
        specificity_count = sum(strategy_section.lower().count(indicator) for indicator in specificity_indicators)
        
        # Evaluate strategy actionability
        action_indicators = [
            "implement", "execute", "perform", "conduct", "undertake",
            "carry out", "do", "act", "proceed", "move forward"
        ]
        
        action_count = sum(strategy_section.lower().count(indicator) for indicator in action_indicators)
        
        # Combine scores
        specificity_score = min(10, specificity_count * 2)
        action_score = min(10, action_count * 2)
        length_score = min(10, len(strategy_section) / 50)  # Normalize by expected length
        
        score = (specificity_score * 0.4) + (action_score * 0.4) + (length_score * 0.2)
        
        logger.info(f"Strategy quality score: {score} (specificity: {specificity_score}, actionability: {action_score}, length: {length_score})")
        return score
    
    def _generate_feedback(self, result: Dict[str, Any]) -> str:
        """
        Generate feedback based on evaluation results
        
        Args:
            result: Evaluation results
            
        Returns:
            str: Feedback
        """
        feedback = ""
        
        # Factor coverage feedback
        if result["factor_coverage"] >= 8:
            feedback += "Excellent coverage of key market factors. "
        elif result["factor_coverage"] >= 5:
            feedback += "Good coverage of market factors, but some important factors were missed. "
        else:
            feedback += "Insufficient coverage of key market factors. "
        
        # Insight coverage feedback
        if result["insight_coverage"] >= 8:
            feedback += "Comprehensive market insights identified. "
        elif result["insight_coverage"] >= 5:
            feedback += "Some market insights identified, but analysis could be more comprehensive. "
        else:
            feedback += "Few expected market insights were identified. "
        
        # Reasoning quality feedback
        if result["reasoning_quality"] >= 8:
            feedback += "Strong reasoning and analysis of market dynamics. "
        elif result["reasoning_quality"] >= 5:
            feedback += "Adequate reasoning, but connections between factors could be more explicit. "
        else:
            feedback += "Reasoning lacks depth and clarity. "
        
        # Strategy quality feedback
        if result["strategy_quality"] >= 8:
            feedback += "Strategic recommendations are specific, actionable, and well-aligned with the analysis."
        elif result["strategy_quality"] >= 5:
            feedback += "Strategic recommendations are present but could be more specific and actionable."
        else:
            feedback += "Strategic recommendations lack specificity and actionability."
        
        return feedback
    
    def get_result_fields(self) -> List[str]:
        """
        Get fields to include in the result
        
        Returns:
            List[str]: List of field names
        """
        return ["score", "max_score", "factor_coverage", "insight_coverage", "reasoning_quality", "strategy_quality", "feedback"]
    
    def _evaluate_with_third_party_ai(self, response_text: str) -> Dict[str, Any]:
        """
        Use third-party AI (Claude-3-7-Sonnet-20250219) to evaluate the answer
        
        Args:
            response_text: Model's answer
            
        Returns:
            Dict[str, Any]: Evaluation results, if evaluation fails returns keyword matching evaluation results
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
                You are a professional market analysis evaluation expert, please evaluate the quality of the answer according to the following scoring criteria.
                
                Scenario: {self.scenario}
                
                Task: {self.instructions}
                
                Answer: {response_text}
                
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
                    "model": self.evaluation_model,
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
                                    "retry_count": retry_count,
                                    "time_taken": end_time - start_time
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
                "explanation": f"Keyword matching found {len(matched_keywords)} keywords or key points relevant to this criterion."
            })
            
            # Add keyword matches to debug info
            keyword_matches[criterion_name] = matched_keywords
            
            total_score += criterion_total_score
        
        logger.info(f"Keyword matching evaluation completed, total score: {total_score:.2f}")
        
        # Build detailed debugging information
        debug_info = {
            "criterion_scores": criterion_scores,
            "keyword_matches": keyword_matches,
            "evaluation_method": "keyword_matching"
        }
        
        # Build final results
        evaluation_result = {
            "score": total_score,
            "total_possible": self.total_possible,
            "overall_feedback": "Scored based on keyword matching and content quality.",
            "criterion_scores": criterion_scores,
            "debug_info": debug_info
        }
    
        return evaluation_result 