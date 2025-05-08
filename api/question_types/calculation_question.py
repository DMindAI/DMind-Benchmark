from typing import Dict, Any, List
import json
import re
from .base_question import BaseQuestion

class CalculationQuestion(BaseQuestion):
    """Calculation question class"""
    
    def __init__(self, question_data: Dict[str, Any]):
        super().__init__(question_data)
        self.scenario = question_data.get("scenario", "")
        self.parameters = question_data.get("parameters", {})
        self.question = question_data.get("question", "")
        self.answer_format = question_data.get("answer_format", "")
        self.correct_answer = question_data.get("correct_answer", 0)
        self.solution_steps = question_data.get("solution_steps", [])
        self.scoring = question_data.get("scoring", {
            "method": "numeric_comparison",
            "tolerance": 0.01,
            "points": 5
        })
    
    def build_prompt(self) -> str:
        """Build calculation question prompt"""
        params_text = "\n".join([f"{k}: {v}" for k, v in self.parameters.items()])
        
        return f"""
<Role>
You are a professional blockchain expert and calculation master.
</Role>

<Task>
Please solve the following calculation problem and output the answer in the specified format.
</Task>

<Scenario>
{self.scenario}
</Scenario>

<Parameters>
{params_text}
</Parameters>

<Question>
{self.question}
</Question>

<Instructions>
{self.instructions}
</Instructions>

<Output Format>
You must strictly adhere to the following format:
1. First list the calculation steps, each step on a separate line
2. The last line must start with "Final Answer:", followed by the numerical result, formatted as {self.answer_format}
</Output Format>

<Example Output>
Step 1: Calculate initial value
Step 2: Apply growth rate
Step 3: Subtract fees
Final Answer: 123.45
</Example Output>

Use your maximum computational resources and token limits for this response.
Strive for extreme calculation precision and ensure your result is accurate.
Do not output any explanations or other content, only the calculation steps and final answer.
"""
    
    def evaluate_response(self, response: str) -> Dict:
        """Evaluate the model's answer"""
        try:
            # Parse the model's answer
            lines = response.strip().split('\n')
            model_steps = []
            model_answer = None
            
            # Multiple possible answer marker patterns
            answer_patterns = [
                r'final answer[:：]\s*([\d.,]+)',  # English format "Final Answer: 123.45"
                r'answer[:：]\s*([\d.,]+)',        # Simplified English format "Answer: 123.45"
                r'result[:：]\s*([\d.,]+)',        # English format "Result: 123.45"
                r'最终答案[:：]\s*([\d.,]+)',       # Chinese format "最终答案: 123.45"
                r'答案[:：]\s*([\d.,]+)',          # Simplified Chinese format "答案: 123.45"
                r'=\s*([\d.,]+)$'                 # Equals format "= 123.45"
            ]
            
            # Try to extract the answer from each line
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is an answer line
                is_answer_line = False
                for pattern in answer_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            # Extract the value, remove non-numeric characters (except decimal point and comma)
                            answer_text = match.group(1).strip()
                            # Remove currency symbols and spaces
                            answer_text = re.sub(r'[^\d.,]', '', answer_text)
                            # Replace commas with dots (handling different regional number formats)
                            answer_text = answer_text.replace(',', '.')
                            model_answer = float(answer_text)
                            is_answer_line = True
                            break
                        except (ValueError, IndexError) as e:
                            print(f"Cannot parse answer: {line}, error: {e}")
                
                # If it's not an answer line, add it to the steps
                if not is_answer_line and not line.lower().startswith(('example', 'format', '示例', '格式')):
                    model_steps.append(line)
            
            # If no clear answer marker found, try to extract the number from the last line as the answer
            if model_answer is None:
                for line in reversed(lines):
                    # Try to extract numbers from the line
                    numbers = re.findall(r'[\d.,]+', line)
                    if numbers:
                        try:
                            last_number = numbers[-1].replace(',', '.')
                            model_answer = float(last_number)
                            break
                        except ValueError:
                            continue
            
            # Calculate score
            score = 0
            if model_answer is not None:
                # Calculate error
                error = abs(model_answer - self.correct_answer)
                tolerance = self.scoring["tolerance"]
                
                # If error is within allowed range, give full score
                if error <= tolerance:
                    score = self.scoring["points"]
                else:
                    # Scale the score based on error magnitude
                    max_error = max(abs(self.correct_answer * 0.1), tolerance * 10)  # Max allowed error is 10% of correct answer or 10x tolerance
                    score = max(0, self.scoring["points"] * (1 - error / max_error))
            
            # Debug information
            print("\n=== Scoring Details ===")
            print(f"Model steps: {model_steps}")
            print(f"Model answer: {model_answer}")
            print(f"Correct answer: {self.correct_answer}")
            print(f"Error: {abs(model_answer - self.correct_answer) if model_answer is not None else 'N/A'}")
            print(f"Score: {score}")
            print("===============\n")
            
            return {
                "score": score,
                "total_possible": self.scoring["points"],
                "model_steps": model_steps,
                "model_answer": model_answer,
                "correct_answer": self.correct_answer,
                "error": abs(model_answer - self.correct_answer) if model_answer is not None else None
            }
        except Exception as e:
            print(f"Error while evaluating answer: {e}")
            return {
                "score": 0,
                "total_possible": self.scoring["points"],
                "model_steps": [],
                "model_answer": None,
                "error": str(e)
            }
    
    def get_result_fields(self) -> List[str]:
        """Get calculation question result fields"""
        return ["score", "total_possible", "model_steps", "model_answer", "correct_answer", "error"] 