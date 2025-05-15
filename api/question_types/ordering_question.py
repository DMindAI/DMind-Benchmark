from typing import Dict, Any, List
import json
from .base_question import BaseQuestion

class OrderingQuestion(BaseQuestion):
    """Ordering question class"""
    
    def __init__(self, question_data: Dict[str, Any]):
        super().__init__(question_data)
        self.steps = question_data.get("steps", [])
        self.correct_order = question_data.get("correct_order", [])
        self.scoring = question_data.get("scoring", {
            "method": "sequence_comparison",
            "points_per_correct_position": 1,
            "total_possible": len(self.steps)  # 1 point for each correct position
        })
    
    def build_prompt(self) -> str:
        """Build ordering question prompt"""
        steps_text = "\n".join([f"{step['id']}. {step['text']}" for step in self.steps])
        
        return f"""

<Role>
You are a professional blockchain expert.
</Role>

<Task>
Please arrange the following steps in the correct order.
</Task>

<Step list>
{steps_text}
</Step list>

<Instructions>
{self.instructions}

Please output the correct order of the steps, with each step ID on a separate line, arranged in the correct sequence.
Only output the step numbers, do not output any other content.
Only output the step numbers, do not output any other content.
Only output the step numbers, do not output any other content.
Only output the step numbers, do not output any other content.
</Instructions>
If your ordering is ABCDE, please output as follows:
A
B
C
D
E

Do not explain, do not output anything else.
"""
    
    def evaluate_response(self, response: str) -> Dict:
        """Evaluate the model's answer"""
        try:
            # 移除思考过程，只保留回答部分
            # 优先处理更精确的</think>\n格式
            if "</think>\n" in response:
                response = response.split("</think>\n")[-1].strip()
            # 如果没有找到，尝试处理</think>格式
            elif "</think>" in response:
                response = response.split("</think>")[-1].strip()
                
            # 处理可能包含的箭头或其他格式
            response = response.replace("→", "\n").replace("->", "\n")
            
            # Parse the model's answer
            lines = response.strip().split('\n')
            model_order = []
            
            # Extract ordering result
            for line in lines:
                if line.strip() and not line.startswith(('Example', 'format')):  # Ignore example format markers
                    model_order.append(line.strip())
            
            # Calculate ordering score
            position_score = 0
            for i, step_id in enumerate(model_order):
                if i < len(self.correct_order) and step_id == self.correct_order[i]:
                    position_score += self.scoring["points_per_correct_position"]
            
            # Debug information
            print("\n=== Scoring Details ===")
            print(f"Model ordering: {model_order}")
            print(f"Correct ordering: {self.correct_order}")
            print(f"Score: {position_score}")
            print("===============\n")
            
            return {
                "score": position_score,
                "total_possible": self.scoring["total_possible"],
                "model_order": model_order,
                "correct_order": self.correct_order
            }
        except Exception as e:
            print(f"Error while evaluating answer: {e}")
            return {
                "score": 0,
                "total_possible": self.scoring["total_possible"],
                "model_order": [],
                "error": str(e)
            }
    
    def get_result_fields(self) -> Dict[str, Any]:
        """Get ordering question result fields"""
        return {
            "question_type": "ordering",
            "steps": self.steps,
            "correct_order": self.correct_order,
            "scoring": self.scoring
        } 