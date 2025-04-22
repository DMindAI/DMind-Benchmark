from typing import Dict, Any
from abc import ABC, abstractmethod

class BaseQuestion(ABC):
    """Base question class, defines the interface that all question types must implement"""
    
    def __init__(self, question_data: Dict[str, Any]):
        self.question_data = question_data
        self.question_type = question_data.get("question_type", "")
        self.instructions = question_data.get("instructions", "")
        self.scoring_criteria = question_data.get("scoring_criteria", {})
    
    @abstractmethod
    def build_prompt(self) -> str:
        """Build prompt"""
        pass
    
    @abstractmethod
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Evaluate response"""
        pass
    
    @abstractmethod
    def get_result_fields(self) -> Dict[str, Any]:
        """Get result fields"""
        pass 