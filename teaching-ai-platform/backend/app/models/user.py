from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from app import db

class User(db.Model):
    """用户基础模型"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    real_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20))
    avatar = db.Column(db.String(255))
    role = db.Column(db.String(20), nullable=False, default='student')  # admin, teacher, student
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        """设置密码"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'real_name': self.real_name,
            'phone': self.phone,
            'avatar': self.avatar,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class Teacher(db.Model):
    """教师扩展信息"""
    __tablename__ = 'teachers'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    employee_id = db.Column(db.String(50), unique=True)
    department = db.Column(db.String(100))
    title = db.Column(db.String(50))  # 职称
    specialties = db.Column(db.Text)  # 专业领域，JSON格式
    bio = db.Column(db.Text)  # 个人简介
    
    # 关联关系
    user = db.relationship('User', backref=db.backref('teacher_profile', uselist=False))
    courses = db.relationship('Course', backref='teacher', lazy='dynamic')

class Student(db.Model):
    """学生扩展信息"""
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    student_id = db.Column(db.String(50), unique=True)
    grade = db.Column(db.String(20))  # 年级
    major = db.Column(db.String(100))  # 专业
    class_name = db.Column(db.String(50))  # 班级
    enrollment_year = db.Column(db.Integer)  # 入学年份
    
    # 关联关系
    user = db.relationship('User', backref=db.backref('student_profile', uselist=False))
    study_records = db.relationship('StudyRecord', backref='student', lazy='dynamic')

class Admin(db.Model):
    """管理员扩展信息"""
    __tablename__ = 'admins'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    permissions = db.Column(db.Text)  # 权限列表，JSON格式
    
    # 关联关系
    user = db.relationship('User', backref=db.backref('admin_profile', uselist=False))
