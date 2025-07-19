from datetime import datetime
from app import db

class Course(db.Model):
    """课程模型"""
    __tablename__ = 'courses'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    subject = db.Column(db.String(100), nullable=False)  # 学科
    difficulty_level = db.Column(db.String(20), default='beginner')  # beginner, intermediate, advanced
    duration_hours = db.Column(db.Integer)  # 课程时长（小时）
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    cover_image = db.Column(db.String(255))
    is_published = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联关系
    contents = db.relationship('CourseContent', backref='course', lazy='dynamic', cascade='all, delete-orphan')
    exercises = db.relationship('Exercise', backref='course', lazy='dynamic', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'subject': self.subject,
            'difficulty_level': self.difficulty_level,
            'duration_hours': self.duration_hours,
            'teacher_id': self.teacher_id,
            'cover_image': self.cover_image,
            'is_published': self.is_published,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class CourseContent(db.Model):
    """课程内容模型"""
    __tablename__ = 'course_contents'
    
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(50), nullable=False)  # text, video, document, code
    content = db.Column(db.Text)  # 文本内容或文件路径
    order_index = db.Column(db.Integer, default=0)  # 排序
    estimated_time = db.Column(db.Integer)  # 预计学习时间（分钟）
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'course_id': self.course_id,
            'title': self.title,
            'content_type': self.content_type,
            'content': self.content,
            'order_index': self.order_index,
            'estimated_time': self.estimated_time,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Exercise(db.Model):
    """练习模型"""
    __tablename__ = 'exercises'
    
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    exercise_type = db.Column(db.String(50), nullable=False)  # quiz, coding, essay, practical
    difficulty = db.Column(db.String(20), default='medium')
    points = db.Column(db.Integer, default=100)  # 总分
    time_limit = db.Column(db.Integer)  # 时间限制（分钟）
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 关联关系
    questions = db.relationship('Question', backref='exercise', lazy='dynamic', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'course_id': self.course_id,
            'title': self.title,
            'description': self.description,
            'exercise_type': self.exercise_type,
            'difficulty': self.difficulty,
            'points': self.points,
            'time_limit': self.time_limit,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Question(db.Model):
    """题目模型"""
    __tablename__ = 'questions'
    
    id = db.Column(db.Integer, primary_key=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey('exercises.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(50), nullable=False)  # multiple_choice, true_false, short_answer, coding
    options = db.Column(db.Text)  # 选择题选项，JSON格式
    correct_answer = db.Column(db.Text)  # 正确答案
    explanation = db.Column(db.Text)  # 答案解释
    points = db.Column(db.Integer, default=10)  # 分值
    order_index = db.Column(db.Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'exercise_id': self.exercise_id,
            'question_text': self.question_text,
            'question_type': self.question_type,
            'options': self.options,
            'correct_answer': self.correct_answer,
            'explanation': self.explanation,
            'points': self.points,
            'order_index': self.order_index
        }
