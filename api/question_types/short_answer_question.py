import os
import json
import time
import logging
import requests
from typing import Dict, List, Optional, Any
from question_types.base_question import BaseQuestion

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("short_answer_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ShortAnswerQuestion")

class ShortAnswerQuestion(BaseQuestion):
    """简短回答类，用于处理简短回答类型的题目"""

    def __init__(self, question_data: Dict[str, Any]):
        """
        初始化简短回答题
        
        Args:
            question_data: 包含简短回答题数据的字典
        """
        super().__init__(question_data)
        self.question_type = "short_answer"
        self.scenario = question_data.get("scenario", "")
        self.instructions = question_data.get("instructions", "")
        self.scoring_criteria = question_data.get("scoring_criteria", [])
        self.total_possible = question_data.get("total_possible", 10)
        self.keywords = question_data.get("keywords", {})  # 每个评分标准的关键词列表
        
        # 从环境变量获取API密钥，如果不存在则使用默认值
        self.third_party_api_key ="sk-sjkpMQ7WsWk5jUShcqhK4RSe3GEooupy8jsy7xQkbg6eQaaX"
        self.third_party_api_base = "https://api.claude-plus.top/v1/chat/completions"
        self.max_retries = 10  # 最大重试次数
        self.retry_delay = 2  # 重试间隔（秒）
        logger.info(f"初始化简短回答题: {self.scenario[:50]}...")
        logger.info(f"使用API密钥: {self.third_party_api_key[:5]}...")

    def build_prompt(self) -> str:
        """
        构建简短回答题的提示
        
        Returns:
            str: 构建好的提示
        """
        prompt = f"场景：{self.scenario}\n\n"
        prompt += f"任务：{self.instructions}\n\n"
        prompt += "请提供简洁明了的回答。"
        logger.info(f"构建提示完成，长度: {len(prompt)}")
        return prompt

    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        评估模型对简短回答题的回答
        
        Args:
            response: 模型的回答
            
        Returns:
            Dict[str, Any]: 评估结果，包含分数和详细信息
        """
        logger.info(f"开始评估回答，回答长度: {len(response)}")
        
        # 使用第三方AI进行评测
        logger.info("尝试使用第三方AI进行评测...")
        third_party_evaluation = self._evaluate_with_third_party_ai(response)
        
        # 第三方AI评测总会返回结果（成功或关键词备用方案）
        logger.info(f"评测完成，总分: {third_party_evaluation.get('score', 0)}")
        return third_party_evaluation

    def _evaluate_with_third_party_ai(self, response_text: str) -> Dict[str, Any]:
        """
        使用第三方AI (Claude-3-7-Sonnet-20250219) 评估回答
        
        Args:
            response_text: 模型的回答
            
        Returns:
            Dict[str, Any]: 评估结果，如果评测失败则返回关键词匹配评测结果
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                if retry_count > 0:
                    logger.info(f"第 {retry_count} 次重试第三方AI评测...")
                    time.sleep(self.retry_delay)  # 重试前等待一段时间
                
                logger.info("开始构建第三方AI评测提示...")
                # 构建评分标准提示
                criteria_prompt = "请根据以下评分标准评估回答的质量：\n\n"
                for criterion in self.scoring_criteria:
                    criterion_name = criterion.get("criterion", "")
                    max_points = criterion.get("points", 0)
                    key_points = criterion.get("key_points", [])
                    min_points_required = criterion.get("min_points_required", 0)
                    
                    criteria_prompt += f"标准：{criterion_name}（满分：{max_points}分）\n"
                    criteria_prompt += "关键点：\n"
                    for point in key_points:
                        criteria_prompt += f"- {point}\n"
                    if min_points_required > 0:
                        criteria_prompt += f"至少需要覆盖{min_points_required}个关键点\n"
                    criteria_prompt += "\n"
                
                evaluation_prompt = f"""
                你是一个专业的评估专家，请根据以下评分标准评估回答的质量。
                
                场景：{self.scenario}
                
                任务：{self.instructions}
                
                回答：{response_text}
                
                {criteria_prompt}
                
                请为每个评分标准提供0-10的分数，并解释你的评分理由。
                最后，请以JSON格式输出评估结果，格式如下：
                {{
                    "criterion_scores": [
                        {{
                            "criterion": "标准名称",
                            "score": 分数,
                            "max_points": 满分,
                            "explanation": "评分理由"
                        }},
                        ...
                    ],
                    "total_score": 总分,
                    "total_possible": {self.total_possible},
                    "overall_feedback": "总体评价"
                }}
                
                只输出JSON格式的评估结果，不要有其他内容。
                """

                logger.info("开始调用第三方AI API...")
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {self.third_party_api_key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": "claude-3-7-sonnet-20250219",
                    "messages": [{"role": "user", "content": evaluation_prompt}],
                    "max_tokens": 4000,
                    "temperature": 0
                }
                
                start_time = time.time()
                response_obj = requests.post(self.third_party_api_base, headers=headers, json=data)
                end_time = time.time()
                
                logger.info(f"API调用完成，耗时: {end_time - start_time:.2f}秒，状态码: {response_obj.status_code}")
                
                if response_obj.status_code != 200:
                    error_msg = f"API调用失败，状态码: {response_obj.status_code}"
                    logger.error(error_msg)
                    last_error = Exception(error_msg)
                    retry_count += 1
                    continue
                
                response_data = response_obj.json()
                logger.info(f"API响应数据: {json.dumps(response_data)[:200]}...")
                
                if "choices" not in response_data or not response_data["choices"]:
                    error_msg = "API响应中没有choices字段"
                    logger.error(error_msg)
                    last_error = Exception(error_msg)
                    retry_count += 1
                    continue
                
                evaluation_text = response_data["choices"][0]["message"]["content"]
                logger.info(f"评估文本长度: {len(evaluation_text)}")
                
                # 尝试从评估文本中提取JSON
                try:
                    # 查找JSON字符串的开始和结束位置
                    json_start = evaluation_text.find("{")
                    json_end = evaluation_text.rfind("}") + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = evaluation_text[json_start:json_end]
                        logger.info(f"提取的JSON长度: {len(json_str)}")
                        
                        evaluation_result = json.loads(json_str)
                        
                        # 检查返回的总分是否为0（可能是错误的评分）
                        total_score = evaluation_result.get('total_score', 0)
                        if total_score == 0 and retry_count == 0:
                            # 第一次尝试就得到0分，记录警告并继续
                            logger.warning("API返回的总分为0，这可能是评分错误。检查评分标准...")
                            
                            # 检查各项标准分数
                            criterion_scores = evaluation_result.get('criterion_scores', [])
                            all_zeros = all(item.get('score', 0) == 0 for item in criterion_scores)
                            
                            if all_zeros and len(criterion_scores) > 0:
                                logger.warning("所有评分标准都是0分，可能是API评分错误。将重试...")
                                raise ValueError("API返回了全0评分，可能是评分错误")
                        
                        logger.info(f"JSON解析成功，总分: {total_score}")
                        
                        # 添加调试信息
                        evaluation_result["debug_info"] = {
                            "evaluation_method": "third_party_ai",
                            "api_response_time": end_time - start_time,
                            "retry_count": retry_count
                        }
                        
                        # 将total_score改为score
                        if "total_score" in evaluation_result:
                            evaluation_result["score"] = evaluation_result.pop("total_score")
                        
                        return evaluation_result
                    else:
                        logger.error("无法在API响应中找到JSON")
                        last_error = Exception("无法在API响应中找到JSON")
                        retry_count += 1
                        continue
                    
                except json.JSONDecodeError as e:
                    error_msg = f"JSON解析失败: {str(e)}"
                    logger.error(error_msg)
                    last_error = e
                    retry_count += 1
                    continue
                    
            except Exception as e:
                error_msg = f"评测过程发生错误: {str(e)}"
                logger.error(error_msg)
                last_error = e
                retry_count += 1
                continue
        
        if last_error:
            logger.error(f"评测失败，最后一次错误: {str(last_error)}")
        
        # 返回关键词匹配的结果，而不是None，确保重试失败后仍能返回有效评分
        return self._evaluate_with_keywords(response_text)

    def _evaluate_with_keywords(self, response: str) -> Dict[str, Any]:
        """
        使用关键词匹配方法评估回答
        
        Args:
            response: 模型的回答
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        logger.info("开始关键词匹配评测...")
        total_score = 0
        criterion_scores = []
        
        for criterion in self.scoring_criteria:
            criterion_name = criterion.get("criterion", "")
            max_points = criterion.get("points", 0)
            key_points = criterion.get("key_points", [])
            min_points_required = criterion.get("min_points_required", 0)
            
            logger.info(f"评估标准: {criterion_name}, 满分: {max_points}")
            
            # 计算关键点匹配度
            key_points_score = 0
            matched_key_points = []
            
            for point in key_points:
                if point.lower() in response.lower():
                    key_points_score += 1
                    matched_key_points.append(point)
            
            # 检查是否达到最小要求
            if min_points_required > 0 and key_points_score < min_points_required:
                logger.info(f"未达到最小要求 ({key_points_score}/{min_points_required})")
                criterion_total_score = 0
            else:
                # 关键点得分占总分的90%
                key_points_score = (key_points_score / len(key_points)) * max_points * 0.9
                logger.info(f"关键点匹配: {len(matched_key_points)}/{len(key_points)}, 得分: {key_points_score:.2f}")
                
                # 计算内容质量得分（占总分的10%）
                content_score = 0
                if len(response) > 50:  # 确保回答有足够的长度
                    content_score = max_points * 0.1
                    logger.info(f"内容质量得分: {content_score:.2f}")
                
                # 计算该标准的总分
                criterion_total_score = key_points_score + content_score
                logger.info(f"标准总分: {criterion_total_score:.2f}")
            
            # 添加到结果中
            criterion_scores.append({
                "criterion": criterion_name,
                "score": criterion_total_score,
                "max_points": max_points,
                "matched_key_points": matched_key_points,
                "key_points_score": key_points_score,
                "content_score": content_score if min_points_required == 0 or key_points_score >= min_points_required else 0
            })
            
            total_score += criterion_total_score
        
        logger.info(f"关键词匹配评测完成，总分: {total_score:.2f}")
        
        # 构建详细的调试信息
        debug_info = {
            "criterion_scores": criterion_scores,
            "total_score": total_score,
            "response_length": len(response),
            "evaluation_method": "keyword_matching"
        }
        
        return {
            "score": total_score,
            "total_possible": self.total_possible,
            "criterion_scores": criterion_scores,
            "debug_info": debug_info
        }

    def get_result_fields(self) -> List[str]:
        """
        获取结果中需要包含的字段
        
        Returns:
            List[str]: 字段列表
        """
        return ["score", "total_possible", "criterion_scores", "debug_info"] 