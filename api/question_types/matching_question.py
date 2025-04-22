from typing import Dict, Any, List
import json
from .base_question import BaseQuestion

class MatchingQuestion(BaseQuestion):
    """连线题类"""
    
    def __init__(self, question_data: Dict[str, Any]):
        super().__init__(question_data)
        self.concepts = question_data.get("concepts", [])
        self.descriptions = question_data.get("descriptions", [])
        self.correct_mapping = question_data.get("correct_mapping", {})
        self.scoring = question_data.get("scoring", {
            "method": "exact_match",
            "points_per_correct": 1,
            "total_possible": len(self.concepts)
        })
    
    def build_prompt(self) -> str:
        """构建连线题提示词"""
        concepts_text = "\n".join([f"{i+1}. {concept}" for i, concept in enumerate(self.concepts)])
        descriptions_text = "\n".join([f"{chr(65+i)}. {desc}" for i, desc in enumerate(self.descriptions)])
        
        return f"""作为一个区块链领域的专家，请将以下概念与对应的描述进行匹配。

概念列表:
{concepts_text}

描述列表:
{descriptions_text}

{self.instructions}

请将每个概念与对应的描述字母进行匹配，只需输出编号对应关系，格式如下:
1 -> A
2 -> B
...

不要解释，不要输出其他任何内容。
"""
    
    def evaluate_response(self, response: str) -> Dict:
        """评估模型的回答"""
        try:
            # 解析模型的回答
            matches = {}
            model_mapping = {}  # 用于存储原始的模型答案
            lines = response.strip().split('\n')
            for line in lines:
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) == 2:
                        concept_idx = int(parts[0].strip()) - 1  # 转换为0-based索引
                        desc_letter = parts[1].strip()
                        if 0 <= concept_idx < len(self.concepts):
                            concept = self.concepts[concept_idx]
                            # 保存原始答案
                            model_mapping[desc_letter] = concept
                            # 如果字母已经存在，说明有重复匹配，记录错误
                            if desc_letter in matches:
                                print(f"警告：字母 {desc_letter} 被重复匹配")
                                continue
                            matches[desc_letter] = concept
            
            # 创建描述文本到字母的映射
            desc_to_letter = {}
            for i, desc in enumerate(self.descriptions):
                letter = chr(65 + i)  # A, B, C, ...
                desc_to_letter[desc] = letter
            
            # 计算正确匹配的数量
            correct_matches = 0
            for desc, expected_concept in self.correct_mapping.items():
                letter = desc_to_letter[desc]
                if letter in matches and matches[letter] == expected_concept:
                    correct_matches += 1
            
            # 计算得分
            score = correct_matches * self.scoring["points_per_correct"]
            
            # 调试信息
            print("\n=== 评分详情 ===")
            print(f"描述到字母映射: {desc_to_letter}")
            print(f"模型原始答案: {model_mapping}")
            print(f"处理后的答案: {matches}")
            print(f"正确答案: {self.correct_mapping}")
            print(f"正确匹配数: {correct_matches}")
            print("===============\n")
            
            return {
                "score": score,
                "total_possible": self.scoring["total_possible"],
                "correct_matches": correct_matches,
                "total_matches": len(self.correct_mapping),
                "matches": matches,
                "model_mapping": model_mapping,  # 保存原始答案
                "has_duplicate_matches": len(matches) < len(model_mapping)  # 使用原始答案长度判断是否有重复
            }
        except Exception as e:
            print(f"评估回答时出错: {e}")
            return {
                "score": 0,
                "total_possible": self.scoring["total_possible"],
                "correct_matches": 0,
                "total_matches": len(self.correct_mapping),
                "matches": {},
                "model_mapping": {},
                "error": str(e)
            }
    
    def get_result_fields(self) -> Dict[str, Any]:
        """获取连线题结果字段"""
        return {
            "question_type": "matching",
            "concepts": self.concepts,
            "descriptions": self.descriptions,
            "correct_mapping": self.correct_mapping,
            "scoring": self.scoring
        } 