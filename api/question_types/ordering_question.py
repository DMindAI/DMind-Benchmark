from typing import Dict, Any, List
import json
from .base_question import BaseQuestion

class OrderingQuestion(BaseQuestion):
    """排序题类"""
    
    def __init__(self, question_data: Dict[str, Any]):
        super().__init__(question_data)
        self.steps = question_data.get("steps", [])
        self.correct_order = question_data.get("correct_order", [])
        self.scoring = question_data.get("scoring", {
            "method": "sequence_comparison",
            "points_per_correct_position": 1,
            "total_possible": len(self.steps)  # 每个正确位置1分
        })
    
    def build_prompt(self) -> str:
        """构建排序题提示词"""
        steps_text = "\n".join([f"{step['id']}. {step['text']}" for step in self.steps])
        
        return f"""作为一个区块链领域的专家，请将以下步骤按照正确的顺序排序。

步骤列表:
{steps_text}

{self.instructions}

请按照以下格式输出排序结果（每行一个步骤ID，按正确顺序排列）：

示例输出格式：
A
B
C
D
E

不要解释，不要输出其他任何内容。
"""
    
    def evaluate_response(self, response: str) -> Dict:
        """评估模型的回答"""
        try:
            # 解析模型的回答
            lines = response.strip().split('\n')
            model_order = []
            
            # 提取排序结果
            for line in lines:
                if line.strip() and not line.startswith(('示例', '格式')):  # 忽略示例格式标记
                    model_order.append(line.strip())
            
            # 计算排序得分
            position_score = 0
            for i, step_id in enumerate(model_order):
                if i < len(self.correct_order) and step_id == self.correct_order[i]:
                    position_score += self.scoring["points_per_correct_position"]
            
            # 调试信息
            print("\n=== 评分详情 ===")
            print(f"模型排序: {model_order}")
            print(f"正确排序: {self.correct_order}")
            print(f"得分: {position_score}")
            print("===============\n")
            
            return {
                "score": position_score,
                "total_possible": self.scoring["total_possible"],
                "model_order": model_order,
                "correct_order": self.correct_order
            }
        except Exception as e:
            print(f"评估回答时出错: {e}")
            return {
                "score": 0,
                "total_possible": self.scoring["total_possible"],
                "model_order": [],
                "error": str(e)
            }
    
    def get_result_fields(self) -> Dict[str, Any]:
        """获取排序题结果字段"""
        return {
            "question_type": "ordering",
            "steps": self.steps,
            "correct_order": self.correct_order,
            "scoring": self.scoring
        } 