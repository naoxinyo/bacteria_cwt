import json
import requests
from typing import List, Dict, Any
from flask import current_app

class AIService:
    """DeepSeek AI服务封装类"""
    
    def __init__(self):
        self.api_key = current_app.config.get('DEEPSEEK_API_KEY')
        self.base_url = current_app.config.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, messages: List[Dict], max_tokens: int = 2000) -> str:
        """发送请求到DeepSeek API"""
        try:
            payload = {
                'model': 'deepseek-chat',
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': 0.7
            }
            
            response = requests.post(
                f'{self.base_url}/v1/chat/completions',
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f'API request failed: {response.status_code}')
                
        except Exception as e:
            current_app.logger.error(f'AI service error: {str(e)}')
            raise e
    
    def generate_course_content(self, course_title: str, subject: str, 
                              outline: str = '', knowledge_base: str = '') -> List[Dict]:
        """生成课程内容"""
        try:
            prompt = f"""
            作为一名专业的教学设计师，请为以下课程生成详细的教学内容：

            课程标题：{course_title}
            学科领域：{subject}
            课程大纲：{outline}
            知识库参考：{knowledge_base}

            请生成5-8个教学模块，每个模块包含：
            1. 模块标题
            2. 知识讲解内容（详细且易懂）
            3. 实训练习指导
            4. 预计学习时间（分钟）

            请以JSON格式返回，格式如下：
            [
                {{
                    "title": "模块标题",
                    "content": "详细的教学内容",
                    "type": "text",
                    "estimated_time": 30
                }}
            ]
            """
            
            messages = [
                {"role": "system", "content": "你是一名专业的教学设计师，擅长创建高质量的教学内容。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_request(messages, max_tokens=3000)
            
            # 解析JSON响应
            try:
                content_data = json.loads(response)
                return content_data
            except json.JSONDecodeError:
                # 如果JSON解析失败，返回默认结构
                return [{
                    "title": f"{course_title} - 基础内容",
                    "content": response,
                    "type": "text",
                    "estimated_time": 45
                }]
                
        except Exception as e:
            current_app.logger.error(f'Generate course content error: {str(e)}')
            raise e
    
    def generate_questions(self, course_title: str, subject: str, 
                          question_count: int = 10, question_types: List[str] = None,
                          difficulty: str = 'medium') -> List[Dict]:
        """生成考核题目"""
        try:
            if question_types is None:
                question_types = ['multiple_choice', 'short_answer']
            
            types_str = ', '.join(question_types)
            
            prompt = f"""
            作为一名专业的教育评估专家，请为以下课程生成{question_count}道考核题目：

            课程标题：{course_title}
            学科领域：{subject}
            题目类型：{types_str}
            难度等级：{difficulty}

            题目要求：
            1. 题目应该全面覆盖课程知识点
            2. 难度适中，符合{difficulty}等级
            3. 包含详细的答案和解释
            4. 如果是选择题，提供4个选项

            请以JSON格式返回，格式如下：
            [
                {{
                    "question": "题目内容",
                    "type": "multiple_choice",
                    "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
                    "answer": "A",
                    "explanation": "答案解释"
                }}
            ]
            """
            
            messages = [
                {"role": "system", "content": "你是一名专业的教育评估专家，擅长设计高质量的考核题目。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_request(messages, max_tokens=3000)
            
            try:
                questions_data = json.loads(response)
                return questions_data
            except json.JSONDecodeError:
                # 如果JSON解析失败，返回默认题目
                return [{
                    "question": f"请简述{course_title}的核心概念。",
                    "type": "short_answer",
                    "answer": "请根据课程内容回答",
                    "explanation": "这是一道开放性题目，考查学生对课程核心概念的理解。"
                }]
                
        except Exception as e:
            current_app.logger.error(f'Generate questions error: {str(e)}')
            raise e
    
    def evaluate_answer(self, question_text: str, question_type: str,
                       correct_answer: str, student_answer: str) -> Dict:
        """评估学生答案"""
        try:
            prompt = f"""
            作为一名专业的教师，请评估以下学生答案：

            题目：{question_text}
            题目类型：{question_type}
            标准答案：{correct_answer}
            学生答案：{student_answer}

            请提供：
            1. 答案是否正确（true/false）
            2. 详细的反馈和建议
            3. 如果答案错误，请指出错误之处并提供正确的解释

            请以JSON格式返回：
            {{
                "is_correct": true/false,
                "feedback": "详细反馈内容"
            }}
            """
            
            messages = [
                {"role": "system", "content": "你是一名专业的教师，擅长评估学生答案并提供建设性反馈。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_request(messages)
            
            try:
                evaluation = json.loads(response)
                return evaluation
            except json.JSONDecodeError:
                # 简单的文本匹配作为备选方案
                is_correct = student_answer.strip().lower() == correct_answer.strip().lower()
                return {
                    "is_correct": is_correct,
                    "feedback": response
                }
                
        except Exception as e:
            current_app.logger.error(f'Evaluate answer error: {str(e)}')
            # 返回默认评估
            return {
                "is_correct": False,
                "feedback": "答案评估暂时不可用，请联系教师。"
            }
    
    def analyze_student_performance(self, course_title: str, 
                                  study_records: List[Dict],
                                  exercise_records: List[Dict]) -> Dict:
        """分析学生学习表现"""
        try:
            prompt = f"""
            作为一名专业的教育数据分析师，请分析以下课程的学生学习情况：

            课程：{course_title}
            学习记录数量：{len(study_records)}
            练习记录数量：{len(exercise_records)}

            请基于提供的数据进行分析，包括：
            1. 整体学习进度和完成情况
            2. 学生表现的统计分析
            3. 发现的问题和改进建议
            4. 教学优化方向

            请以JSON格式返回分析结果：
            {{
                "content": "详细分析内容",
                "statistics": "统计数据摘要",
                "recommendations": "改进建议"
            }}
            """
            
            messages = [
                {"role": "system", "content": "你是一名专业的教育数据分析师，擅长从学习数据中提取有价值的洞察。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_request(messages, max_tokens=2000)
            
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                return {
                    "content": response,
                    "statistics": f"分析了{len(study_records)}条学习记录和{len(exercise_records)}条练习记录",
                    "recommendations": "建议持续关注学生学习进度，及时提供个性化指导。"
                }
                
        except Exception as e:
            current_app.logger.error(f'Analyze student performance error: {str(e)}')
            return {
                "content": "学习情况分析暂时不可用",
                "statistics": "统计数据处理中",
                "recommendations": "请稍后重试或联系技术支持"
            }
    
    def answer_student_question(self, question: str, context: str = '') -> str:
        """回答学生问题"""
        try:
            prompt = f"""
            作为一名专业的在线学习助手，请回答学生的问题：

            学生问题：{question}
            课程上下文：{context}

            请提供：
            1. 清晰、准确的答案
            2. 必要的解释和举例
            3. 相关的学习建议

            回答要求：
            - 语言简洁易懂
            - 内容准确可靠
            - 具有教育价值
            """
            
            messages = [
                {"role": "system", "content": "你是一名专业的在线学习助手，擅长回答学生的学习问题。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_request(messages)
            return response
            
        except Exception as e:
            current_app.logger.error(f'Answer student question error: {str(e)}')
            return "很抱歉，我暂时无法回答您的问题。请稍后重试或联系教师获得帮助。"
