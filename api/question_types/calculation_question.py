from typing import Dict, Any, List
import json
from .base_question import BaseQuestion

class CalculationQuestion(BaseQuestion):
    """计算题类"""
    
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
        """构建计算题提示词"""
        # 构建参数说明
        params_text = "\n".join([f"{k}: {v}" for k, v in self.parameters.items()])
        
        return f"""作为一个区块链领域的专家，请解决以下计算问题。

场景: {self.scenario}

参数:
{params_text}

问题: {self.question}

{self.instructions}

请按照以下格式输出答案：
1. 计算步骤（每行一个步骤）
2. 最终答案（{self.answer_format}）

示例输出格式：
步骤1: ...
步骤2: ...
...
答案: 123.45

不要解释，不要输出其他任何内容。
"""
    
    def evaluate_response(self, response: str) -> Dict:
        """评估模型的回答"""
        try:
            # 解析模型的回答
            lines = response.strip().split('\n')
            model_steps = []
            model_answer = None
            
            # 分离步骤和答案
            for line in lines:
                if line.lower().startswith(('答案:', 'answer:')):
                    try:
                        # 提取数值
                        answer_text = line.split(':')[1].strip()
                        # 移除货币符号和空格
                        answer_text = answer_text.replace('$', '').replace('¥', '').strip()
                        model_answer = float(answer_text)
                    except (ValueError, IndexError):
                        print(f"无法解析答案: {line}")
                elif line.strip() and not line.startswith(('示例', '格式')):
                    model_steps.append(line.strip())
            
            # 计算得分
            score = 0
            if model_answer is not None:
                # 计算误差
                error = abs(model_answer - self.correct_answer)
                tolerance = self.scoring["tolerance"]
                
                # 如果误差在允许范围内，给予满分
                if error <= tolerance:
                    score = self.scoring["points"]
                else:
                    # 根据误差大小按比例扣分
                    max_error = max(abs(self.correct_answer * 0.1), tolerance * 10)  # 最大允许误差为正确答案的10%或容差的10倍
                    score = max(0, self.scoring["points"] * (1 - error / max_error))
            
            # 调试信息
            print("\n=== 评分详情 ===")
            print(f"模型步骤: {model_steps}")
            print(f"模型答案: {model_answer}")
            print(f"正确答案: {self.correct_answer}")
            print(f"误差: {abs(model_answer - self.correct_answer) if model_answer is not None else 'N/A'}")
            print(f"得分: {score}")
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
            print(f"评估回答时出错: {e}")
            return {
                "score": 0,
                "total_possible": self.scoring["points"],
                "model_steps": [],
                "model_answer": None,
                "error": str(e)
            }
    
    def get_result_fields(self) -> Dict[str, Any]:
        """获取计算题结果字段"""
        return {
            "question_type": "calculation",
            "scenario": self.scenario,
            "parameters": self.parameters,
            "question": self.question,
            "answer_format": self.answer_format,
            "correct_answer": self.correct_answer,
            "solution_steps": self.solution_steps,
            "scoring": self.scoring
        } 