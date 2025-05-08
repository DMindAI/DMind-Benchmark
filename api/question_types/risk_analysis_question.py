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
        logging.FileHandler("risk_analysis_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RiskAnalysisQuestion")

class RiskAnalysisQuestion(BaseQuestion):
    """Risk analysis question class for evaluating risk assessment capabilities"""
    
    def __init__(self, question_data: Dict[str, Any]):
        """
        Initialize risk analysis question
        
        Args:
            question_data: Dictionary containing risk analysis question data
        """
        super().__init__(question_data)
        self.question_type = "risk_analysis"
        self.scenario = question_data.get("scenario", "")
        self.risk_factors = question_data.get("risk_factors", [])
        self.expected_threats = question_data.get("expected_threats", [])
        self.expected_vulnerabilities = question_data.get("expected_vulnerabilities", [])
        self.expected_countermeasures = question_data.get("expected_countermeasures", [])
        self.risk_weights = question_data.get("risk_weights", {"threats": 0.3, "vulnerabilities": 0.3, "countermeasures": 0.4})
        
        logger.info(f"Initialized risk analysis question with {len(self.risk_factors)} risk factors, " 
                   f"{len(self.expected_threats)} expected threats, "
                   f"{len(self.expected_vulnerabilities)} expected vulnerabilities, "
                   f"{len(self.expected_countermeasures)} expected countermeasures")
        
        # Calculate total_possible from scoring criteria
        total_points = 0
        for criterion in self.scoring_criteria:
            total_points += criterion.get("points", 0)
        self.total_possible = question_data.get("total_possible", total_points)
        
        # Get API configuration from config manager
        api_config = config_manager.get_third_party_api_config()
        self.third_party_api_key = api_config["api_key"]
        self.third_party_api_base = api_config["api_base"]
        self.evaluation_model = api_config["model"]
        
        self.max_retries = 5  # Maximum retry attempts
        self.retry_delay = 2  # Retry interval (seconds)
        logger.info(f"Initializing risk analysis question: {len(self.scenario)} characters")
        logger.info(f"Using API key: {self.third_party_api_key[:5]}... with model: {self.evaluation_model}")
        
    def build_prompt(self) -> str:
        """
        Build risk analysis question prompt
        
        Returns:
            str: Built prompt
        """
        prompt = f""
        
        if self.scenario:
            prompt += f"Scenario:\n{self.scenario}\n\n"
        
        if self.risk_factors:
            prompt += "Consider the following risk factors in your analysis:\n"
            for i, factor in enumerate(self.risk_factors):
                prompt += f"{i+1}. {factor}\n"
            prompt += "\n"
        
        prompt += ("For the above scenario, provide a comprehensive risk analysis that includes:\n"
                  "1. Key threats: Identify potential threats relevant to this scenario\n"
                  "2. Vulnerabilities: Analyze weak points that could be exploited\n"
                  "3. Countermeasures: Suggest effective controls or measures to mitigate risks\n"
                  "4. Risk assessment: Provide an overall risk assessment with priority levels\n\n"
                  "Organize your analysis into clear sections for each component.")
        
        logger.info(f"Prompt built with length: {len(prompt)}")
        return prompt
    
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        Evaluate model's answer to risk analysis question
        
        Args:
            response: Model's response to evaluate
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info(f"Evaluating risk analysis response of length: {len(response)}")
        
        result = {
            "score": 0,
            "max_score": 10,
            "threat_score": 0,
            "vulnerability_score": 0,
            "countermeasure_score": 0,
            "identified_threats": [],
            "identified_vulnerabilities": [],
            "identified_countermeasures": [],
            "missed_threats": [],
            "missed_vulnerabilities": [],
            "missed_countermeasures": [],
            "feedback": ""
        }
        
        # Evaluate threats identified
        threat_score, identified_threats, missed_threats = self._evaluate_component(
            response, self.expected_threats, "threats"
        )
        result["threat_score"] = threat_score
        result["identified_threats"] = identified_threats
        result["missed_threats"] = missed_threats
        
        # Evaluate vulnerabilities identified
        vulnerability_score, identified_vulnerabilities, missed_vulnerabilities = self._evaluate_component(
            response, self.expected_vulnerabilities, "vulnerabilities"
        )
        result["vulnerability_score"] = vulnerability_score
        result["identified_vulnerabilities"] = identified_vulnerabilities
        result["missed_vulnerabilities"] = missed_vulnerabilities
        
        # Evaluate countermeasures proposed
        countermeasure_score, identified_countermeasures, missed_countermeasures = self._evaluate_component(
            response, self.expected_countermeasures, "countermeasures"
        )
        result["countermeasure_score"] = countermeasure_score
        result["identified_countermeasures"] = identified_countermeasures
        result["missed_countermeasures"] = missed_countermeasures
        
        # Calculate weighted overall score
        result["score"] = (
            threat_score * self.risk_weights["threats"] +
            vulnerability_score * self.risk_weights["vulnerabilities"] +
            countermeasure_score * self.risk_weights["countermeasures"]
        )
        
        # Generate feedback
        result["feedback"] = self._generate_feedback(result)
        
        logger.info(f"Risk analysis evaluation completed. Final score: {result['score']}/{result['max_score']}")
        return result
    
    def _evaluate_component(self, response: str, expected_items: List[str], component_type: str) -> tuple:
        """
        Evaluate a specific component of the risk analysis
        
        Args:
            response: Model's response
            expected_items: List of expected items for this component
            component_type: Type of component (threats, vulnerabilities, countermeasures)
            
        Returns:
            tuple: (score, identified_items, missed_items)
        """
        if not expected_items:
            logger.warning(f"No expected {component_type} defined, returning default score")
            return 5.0, [], []  # Default middle score if no expected items
        
        response_lower = response.lower()
        
        identified_items = []
        missed_items = []
        
        # Check which items were identified
        for item in expected_items:
            if item.lower() in response_lower:
                identified_items.append(item)
            else:
                missed_items.append(item)
        
        # Calculate coverage ratio
        coverage_ratio = len(identified_items) / len(expected_items)
        
        # Score is out of 10
        component_score = coverage_ratio * 10
        
        logger.info(f"{component_type.capitalize()} evaluation: {len(identified_items)}/{len(expected_items)} items identified, score: {component_score}")
        return component_score, identified_items, missed_items
    
    def _generate_feedback(self, result: Dict[str, Any]) -> str:
        """
        Generate feedback based on evaluation results
        
        Args:
            result: Evaluation results
            
        Returns:
            str: Feedback
        """
        feedback = ""
        
        # Threat analysis feedback
        if result["threat_score"] >= 8:
            feedback += "Excellent threat identification with comprehensive coverage. "
        elif result["threat_score"] >= 5:
            feedback += "Good threat analysis, but some important threats were missed. "
        else:
            feedback += "Insufficient threat identification. Key threats missing include: " + ", ".join(result["missed_threats"][:3]) + ". "
        
        # Vulnerability analysis feedback
        if result["vulnerability_score"] >= 8:
            feedback += "Strong vulnerability assessment with thorough analysis. "
        elif result["vulnerability_score"] >= 5:
            feedback += "Adequate vulnerability analysis, but lacks depth in some areas. "
        else:
            feedback += "Weak vulnerability assessment. Important vulnerabilities missing include: " + ", ".join(result["missed_vulnerabilities"][:3]) + ". "
        
        # Countermeasure feedback
        if result["countermeasure_score"] >= 8:
            feedback += "Comprehensive countermeasures proposed with effective risk mitigation strategies. "
        elif result["countermeasure_score"] >= 5:
            feedback += "Reasonable countermeasures suggested, but some key controls were overlooked. "
        else:
            feedback += "Insufficient countermeasures proposed. Important missing controls include: " + ", ".join(result["missed_countermeasures"][:3]) + ". "
        
        # Overall feedback
        if result["score"] >= 8:
            feedback += "Overall, this is a strong risk analysis that effectively addresses the scenario."
        elif result["score"] >= 5:
            feedback += "Overall, this is a satisfactory risk analysis but with room for improvement in coverage and depth."
        else:
            feedback += "Overall, this risk analysis requires significant improvement in identifying threats, vulnerabilities, and appropriate countermeasures."
        
        return feedback
    
    def get_result_fields(self) -> List[str]:
        """
        Get fields to include in the result
        
        Returns:
            List[str]: List of field names
        """
        return [
            "score", "max_score", 
            "threat_score", "vulnerability_score", "countermeasure_score",
            "identified_threats", "identified_vulnerabilities", "identified_countermeasures",
            "missed_threats", "missed_vulnerabilities", "missed_countermeasures",
            "feedback"
        ]
    
    def _evaluate_with_third_party_ai(self, response_text: str) -> Dict[str, Any]:
        """
        Use third-party AI to evaluate the answer
        
        Args:
            response_text: Model's answer
            
        Returns:
            Dict[str, Any]: Evaluation results, if evaluation fails returns None
        """
        # Implementation would be here
        # This is just a placeholder for the real implementation
        pass 