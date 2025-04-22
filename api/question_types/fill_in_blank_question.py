from typing import Dict, List, Any, Optional
from .base_question import BaseQuestion

class FillInBlankQuestion(BaseQuestion):
    """填空题类，用于处理填空题类型的题目"""
    
    def __init__(self, question_data: Dict[str, Any]):
        """
        初始化填空题
        
        Args:
            question_data: 包含填空题数据的字典
        """
        super().__init__(question_data)
        self.question_type = "fill_in_blank"
        self.instructions = question_data.get("instructions", "")
        self.context = question_data.get("context", "")
        self.blanks = question_data.get("blanks", [])
        self.scoring = question_data.get("scoring", {})
        
    def build_prompt(self) -> str:
        """
        构建填空题的提示
        
        Returns:
            str: 构建好的提示
        """
        prompt = f"{self.instructions}\n\n{self.context}\n\n"
        prompt += "请按顺序输出所有填空的答案，格式如下：\n"
        prompt += "#1#: [答案1]\n"
        prompt += "#2#: [答案2]\n"
        prompt += "#3#: [答案3]\n"
        prompt += "...\n\n"
        prompt += "只需输出答案，无需其他解释。"
        return prompt
    
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        评估模型对填空题的回答
        
        Args:
            response: 模型的回答
            
        Returns:
            Dict[str, Any]: 评估结果，包含分数和详细信息
        """
        # 解析模型的回答
        model_answers = self._parse_response(response)
        
        # 计算正确数量
        correct_count = 0
        results = []
        
        for blank in self.blanks:
            blank_id = blank.get("id")
            correct_answer = blank.get("answer")
            answer_type = blank.get("type", "text")
            
            model_answer = model_answers.get(str(blank_id))
            
            # 检查答案是否正确
            is_correct = False
            if model_answer is not None:
                if answer_type == "number":
                    try:
                        # 对于数字类型，尝试转换为浮点数进行比较
                        model_value = float(model_answer)
                        correct_value = float(correct_answer)
                        is_correct = abs(model_value - correct_value) < 0.0001  # 使用小误差范围
                    except ValueError:
                        is_correct = False
                else:
                    # 对于文本类型，直接比较
                    is_correct = str(model_answer).strip().lower() == str(correct_answer).strip().lower()
            
            if is_correct:
                correct_count += 1
            
            results.append({
                "blank_id": blank_id,
                "correct_answer": correct_answer,
                "model_answer": model_answer,
                "is_correct": is_correct
            })
        
        # 计算分数
        points_per_correct = self.scoring.get("points_per_correct", 1)
        score = correct_count * points_per_correct
        
        # 构建详细的调试信息
        debug_info = {
            "model_answers": model_answers,
            "results": results,
            "correct_count": correct_count,
            "score": score
        }
        
        # 构建更详细的结果
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
        解析模型的回答，提取填空答案
        
        Args:
            response: 模型的回答
            
        Returns:
            Dict[str, str]: 解析后的答案，键为填空ID，值为答案
        """
        # 这里需要根据模型的输出格式进行解析
        # 假设模型会按照 "#1#: 100" 这样的格式输出答案
        answers = {}
        
        # 尝试从回答中提取填空ID和答案
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 尝试匹配 "#数字#: 答案" 格式
            import re
            match = re.match(r'#(\d+)#:\s*(.+)', line)
            if match:
                blank_id = match.group(1)
                answer = match.group(2).strip()
                answers[blank_id] = answer
                
        return answers
    
    def get_result_fields(self) -> List[str]:
        """
        获取结果中需要包含的字段
        
        Returns:
            List[str]: 字段列表
        """
        return ["score", "total_possible", "debug_info"] 