from typing import Dict, List, Any, Optional
from .base_question import BaseQuestion

class FillInBlankQuestion(BaseQuestion):
    """Fill-in-the-blank question class, used to handle fill-in-the-blank type questions"""
    
    def __init__(self, question_data: Dict[str, Any]):
        """
        Initialize fill-in-the-blank question
        
        Args:
            question_data: Dictionary containing fill-in-the-blank question data
        """
        super().__init__(question_data)
        self.question_type = "fill_in_blank"
        self.instructions = question_data.get("instructions", "")
        self.context = question_data.get("context", "")
        self.blanks = question_data.get("blanks", [])
        self.scoring = question_data.get("scoring", {})
        
    def build_prompt(self) -> str:
        """
        Build fill-in-the-blank question prompt
        
        Returns:
            str: Built prompt
        """
        prompt = f"{self.instructions}\n\n{self.context}\n\n"
        prompt += "Please output answers for all blanks in order, in the following format:\n"
        prompt += "#1#: [answer1]\n"
        prompt += "#2#: [answer2]\n"
        prompt += "#3#: [answer3]\n"
        prompt += "...\n\n"
        prompt += "Only output the answers, no additional explanation needed."
        return prompt
    
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        Evaluate model's answer to fill-in-the-blank question
        
        Args:
            response: Model's answer
            
        Returns:
            Dict[str, Any]: Evaluation results, including score and detailed information
        """
        # Parse the model's answer
        model_answers = self._parse_response(response)
        
        # Calculate number of correct answers
        correct_count = 0
        results = []
        
        for blank in self.blanks:
            blank_id = blank.get("id")
            correct_answer = blank.get("answer")
            answer_type = blank.get("type", "text")
            
            model_answer = model_answers.get(str(blank_id))
            
            # Check if the answer is correct
            is_correct = False
            if model_answer is not None:
                if answer_type == "number":
                    try:
                        # For numeric types, try to convert to float for comparison
                        model_value = float(model_answer)
                        correct_value = float(correct_answer)
                        is_correct = abs(model_value - correct_value) < 0.0001  # Use small error margin
                    except ValueError:
                        is_correct = False
                else:
                    # For text types, compare directly
                    is_correct = str(model_answer).strip().lower() == str(correct_answer).strip().lower()
            
            if is_correct:
                correct_count += 1
            
            results.append({
                "blank_id": blank_id,
                "correct_answer": correct_answer,
                "model_answer": model_answer,
                "is_correct": is_correct
            })
        
        # Calculate score
        points_per_correct = self.scoring.get("points_per_correct", 1)
        score = correct_count * points_per_correct
        
        # Build detailed debug information
        debug_info = {
            "model_answers": model_answers,
            "results": results,
            "correct_count": correct_count,
            "score": score
        }
        
        # Build more detailed results
        detailed_results = {
            "score": score,
            "total_possible": self.scoring.get("total_possible", len(self.blanks)),
            "correct_count": correct_count,
            "total_blanks": len(self.blanks),
            "model_answers": model_answers,
            "correct_answers": {str(blank.get("id")): blank.get("answer") for blank in self.blanks},
            "blank_details": results,
            "debug_info": debug_info
        }
        
        return detailed_results
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse the model's answer, extract fill-in-the-blank answers
        
        Args:
            response: Model's answer
            
        Returns:
            Dict[str, str]: Parsed answers, keys are blank IDs, values are answers
        """
        # Here we need to parse based on the model's output format
        # Assuming the model outputs answers in the format "#1#: 100"
        answers = {}
        
        # Try to extract blank IDs and answers from the response
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to match the "#number#: answer" format
            import re
            match = re.match(r'#(\d+)#:\s*(.+)', line)
            if match:
                blank_id = match.group(1)
                answer = match.group(2).strip()
                answers[blank_id] = answer
                
        return answers
    
    def get_result_fields(self) -> List[str]:
        """
        Get fields to include in results
        
        Returns:
            List[str]: Field list
        """
        return ["score", "total_possible", "debug_info"] 