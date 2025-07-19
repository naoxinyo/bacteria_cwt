from datetime import datetime
from app import db

class StudyRecord(db.Model):
    """学习记录模型"""
    __tablename__ = 'study_records'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('course_contents.id'))
    study_time = db.Column(db.Integer, default=0)  # 学习时间（分钟）
    progress = db.Column(db.Float, default=0.0)  # 学习进度（0-1）
    status = db.Column(db.String(20), default='in_progress')  # not_started, in_progress, completed
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    last_accessed = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 关联关系
    course = db.relationship('Course', backref='study_records')
    content = db.relationship('CourseContent', backref='study_records')
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'course_id': self.course_id,
            'content_id': self.content_id,
            'study_time': self.study_time,
            'progress': self.progress,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }

class ExerciseRecord(db.Model):
    """练习记录模型"""
    __tablename__ = 'exercise_records'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    exercise_id = db.Column(db.Integer, db.ForeignKey('exercises.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'))
    student_answer = db.Column(db.Text)  # 学生答案
    is_correct = db.Column(db.Boolean)
    score = db.Column(db.Float, default=0.0)  # 得分
    time_spent = db.Column(db.Integer)  # 用时（秒）
    attempt_count = db.Column(db.Integer, default=1)  # 尝试次数
    feedback = db.Column(db.Text)  # AI反馈
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 关联关系
    exercise = db.relationship('Exercise', backref='records')
    question = db.relationship('Question', backref='records')
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'exercise_id': self.exercise_id,
            'question_id': self.question_id,
            'student_answer': self.student_answer,
            'is_correct': self.is_correct,
            'score': self.score,
            'time_spent': self.time_spent,
            'attempt_count': self.attempt_count,
            'feedback': self.feedback,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None
        }

class AnalysisReport(db.Model):
    """分析报告模型"""
    __tablename__ = 'analysis_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    report_type = db.Column(db.String(50), nullable=False)  # student_progress, class_performance, course_analysis
    target_id = db.Column(db.Integer)  # 目标ID（学生ID、课程ID等）
    target_type = db.Column(db.String(50))  # student, course, class
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text)  # 报告内容，JSON格式
    statistics = db.Column(db.Text)  # 统计数据，JSON格式
    recommendations = db.Column(db.Text)  # 建议，JSON格式
    generated_by = db.Column(db.Integer, db.ForeignKey('users.id'))  # 生成者
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 关联关系
    generator = db.relationship('User', backref='generated_reports')
    
    def to_dict(self):
        return {
            'id': self.id,
            'report_type': self.report_type,
            'target_id': self.target_id,
            'target_type': self.target_type,
            'title': self.title,
            'content': self.content,
            'statistics': self.statistics,
            'recommendations': self.recommendations,
            'generated_by': self.generated_by,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
