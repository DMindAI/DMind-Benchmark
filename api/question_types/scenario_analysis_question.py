from typing import Dict, List, Any, Optional
import requests
import json
import time
import logging
import os
from .base_question import BaseQuestion

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scenario_analysis_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ScenarioAnalysisQuestion")

class ScenarioAnalysisQuestion(BaseQuestion):
    """场景分析类，用于处理场景分析类型的题目"""
    
    def __init__(self, question_data: Dict[str, Any]):
        """
        初始化场景分析题
        
        Args:
            question_data: 包含场景分析题数据的字典
        """
        super().__init__(question_data)
        self.question_type = "scenario_analysis"
        self.scenario = question_data.get("scenario", "")
        self.instructions = question_data.get("instructions", "")
        self.scoring_criteria = question_data.get("scoring_criteria", [])
        self.total_possible = question_data.get("total_possible", 10)
        self.keywords = question_data.get("keywords", {})  # 每个评分标准的关键词列表
        
        # 从环境变量获取API密钥，如果不存在则使用默认值
        self.third_party_api_key = os.environ.get("CLAUDE_API_KEY", "sk-sjkpMQ7WsWk5jUShcqhK4RSe3GEooupy8jsy7xQkbg6eQaaX")
        self.third_party_api_base = "https://api.claude-plus.top/v1/chat/completions"
        self.max_retries = 10  # 最大重试次数
        self.retry_delay = 2  # 重试间隔（秒）
        logger.info(f"初始化场景分析题: {self.scenario[:50]}...")
        logger.info(f"使用API密钥: {self.third_party_api_key[:5]}...")
        
    def build_prompt(self) -> str:
        """
        构建场景分析题的提示
        
        Returns:
            str: 构建好的提示
        """
        prompt = f"场景：{self.scenario}\n\n"
        prompt += f"任务：{self.instructions}\n\n"
        prompt += "请提供详细的分析和建议。"
        logger.info(f"构建提示完成，长度: {len(prompt)}")
        return prompt
    
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """
        评估模型对情景分析题的回答
        
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
                    
                    criteria_prompt += f"标准：{criterion_name}（满分：{max_points}分）\n"
                    criteria_prompt += "关键点：\n"
                    for point in key_points:
                        criteria_prompt += f"- {point}\n"
                    criteria_prompt += "\n"
                
                # 构建完整的评测提示
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
                
                logger.info(f"评测提示构建完成，长度: {len(evaluation_prompt)}")
                
                # 调用Claude API
                logger.info("开始调用Claude API...")
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
                
                if response_obj.status_code == 200:
                    response_data = response_obj.json()
                    logger.info(f"API响应数据: {json.dumps(response_data)[:200]}...")
                    
                    # 从choices中获取回答
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        evaluation_text = response_data["choices"][0]["message"]["content"]
                        logger.info(f"API返回文本长度: {len(evaluation_text)}")
                        
                        # 提取JSON部分
                        json_start = evaluation_text.find("{")
                        json_end = evaluation_text.rfind("}") + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            try:
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
                            except json.JSONDecodeError as e:
                                logger.error(f"解析JSON失败: {str(e)}")
                                last_error = f"解析JSON失败: {str(e)}"
                                # 继续下一次重试
                        else:
                            logger.error("无法在API响应中找到JSON")
                            last_error = "无法在API响应中找到JSON"
                    else:
                        logger.error("API响应中没有choices字段")
                        last_error = "API响应格式不正确"
                else:
                    error_message = "未知错误"
                    try:
                        error_data = response_obj.json()
                        if "error" in error_data:
                            error_message = error_data["error"].get("message", "未知错误")
                            error_type = error_data["error"].get("type", "未知类型")
                            logger.error(f"API调用失败: {error_message} (类型: {error_type})")
                    except:
                        logger.error(f"API调用失败: {response_obj.text[:200]}...")
                    
                    last_error = f"API调用失败: {response_obj.status_code} - {error_message}"
                    
                    # 如果是认证错误，尝试使用备用API密钥
                    if "未提供令牌" in error_message or "authentication" in error_message.lower():
                        logger.warning("检测到认证错误，尝试使用备用API密钥...")
                        # 这里可以添加备用API密钥的逻辑
                        # self.third_party_api_key = "备用API密钥"
            
            except Exception as e:
                logger.error(f"第三方AI评测失败: {str(e)}", exc_info=True)
                last_error = str(e)
            
            retry_count += 1
            if retry_count < self.max_retries:
                logger.info(f"将在 {self.retry_delay} 秒后进行第 {retry_count + 1} 次重试...")
        
        logger.error(f"第三方AI评测失败，已重试 {retry_count} 次，最后一次错误: {last_error}")
        # 返回关键词匹配的结果，而不是None，确保重试失败后仍能返回有效评分
        return self._evaluate_with_keywords(response_text)
    
    def _evaluate_with_keywords(self, response: str) -> Dict[str, Any]:
        """
        使用关键词匹配方法评估回答（原有评测逻辑）
        
        Args:
            response: 模型的回答
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        logger.info("开始使用关键词匹配方法评估回答...")
        # 初始化结果
        total_score = 0
        criterion_scores = []
        keyword_matches = {}
        
        # 对每个评分标准进行评估
        for criterion in self.scoring_criteria:
            criterion_name = criterion.get("criterion", "")
            max_points = criterion.get("points", 0)
            key_points = criterion.get("key_points", [])
            
            logger.info(f"评估标准: {criterion_name}, 满分: {max_points}")
            
            # 获取该标准的关键词列表
            criterion_keywords = self.keywords.get(criterion_name, [])
            
            # 计算关键词匹配度
            keyword_score = 0
            matched_keywords = []
            
            if criterion_keywords:
                for keyword in criterion_keywords:
                    if keyword.lower() in response.lower():
                        keyword_score += 1
                        matched_keywords.append(keyword)
                
                # 关键词得分占总分的70%
                keyword_score = (keyword_score / len(criterion_keywords)) * max_points * 0.7
                logger.info(f"关键词匹配: {len(matched_keywords)}/{len(criterion_keywords)}, 得分: {keyword_score:.2f}")
            else:
                # 如果没有关键词，则基于关键点评估
                key_points_score = 0
                for point in key_points:
                    if point.lower() in response.lower():
                        key_points_score += 1
                
                # 关键点得分占总分的70%
                keyword_score = (key_points_score / len(key_points)) * max_points * 0.7
                logger.info(f"关键点匹配: {key_points_score}/{len(key_points)}, 得分: {keyword_score:.2f}")
            
            # 计算内容质量得分（占总分的30%）
            content_score = 0
            if len(response) > 100:  # 确保回答有足够的长度
                content_score = max_points * 0.3
                logger.info(f"内容质量得分: {content_score:.2f}")
            
            # 计算该标准的总分
            criterion_total_score = keyword_score + content_score
            logger.info(f"标准总分: {criterion_total_score:.2f}")
            
            # 添加到结果中
            criterion_scores.append({
                "criterion": criterion_name,
                "score": criterion_total_score,
                "max_points": max_points,
                "matched_keywords": matched_keywords,
                "keyword_score": keyword_score,
                "content_score": content_score
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