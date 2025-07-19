from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models.user import User, Teacher
from app.models.course import Course, CourseContent, Exercise, Question
from app.models.learning import StudyRecord, ExerciseRecord, AnalysisReport
from app.services.ai_service import AIService

teacher_bp = Blueprint('teacher', __name__)

def get_current_teacher():
    """获取当前教师信息"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user or user.role != 'teacher':
        return None
    return user.teacher_profile

@teacher_bp.route('/courses', methods=['GET'])
@jwt_required()
def get_courses():
    """获取教师的课程列表"""
    try:
        teacher = get_current_teacher()
        if not teacher:
            return jsonify({'error': 'Teacher not found'}), 404
        
        courses = Course.query.filter_by(teacher_id=teacher.id).all()
        return jsonify({
            'courses': [course.to_dict() for course in courses]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@teacher_bp.route('/courses', methods=['POST'])
@jwt_required()
def create_course():
    """创建新课程"""
    try:
        teacher = get_current_teacher()
        if not teacher:
            return jsonify({'error': 'Teacher not found'}), 404
        
        data = request.get_json()
        
        # 验证必填字段
        required_fields = ['title', 'subject']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        course = Course(
            title=data['title'],
            description=data.get('description'),
            subject=data['subject'],
            difficulty_level=data.get('difficulty_level', 'beginner'),
            duration_hours=data.get('duration_hours'),
            teacher_id=teacher.id
        )
        
        db.session.add(course)
        db.session.commit()
        
        return jsonify({
            'message': 'Course created successfully',
            'course': course.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@teacher_bp.route('/courses/<int:course_id>/generate-content', methods=['POST'])
@jwt_required()
def generate_course_content():
    """AI生成课程内容"""
    try:
        teacher = get_current_teacher()
        if not teacher:
            return jsonify({'error': 'Teacher not found'}), 404
        
        course_id = request.view_args['course_id']
        course = Course.query.filter_by(id=course_id, teacher_id=teacher.id).first()
        if not course:
            return jsonify({'error': 'Course not found'}), 404
        
        data = request.get_json()
        outline = data.get('outline', '')
        knowledge_base = data.get('knowledge_base', '')
        
        # 调用AI服务生成内容
        ai_service = AIService()
        generated_content = ai_service.generate_course_content(
            course_title=course.title,
            subject=course.subject,
            outline=outline,
            knowledge_base=knowledge_base
        )
        
        # 保存生成的内容
        for i, content_data in enumerate(generated_content):
            content = CourseContent(
                course_id=course.id,
                title=content_data['title'],
                content_type=content_data.get('type', 'text'),
                content=content_data['content'],
                order_index=i,
                estimated_time=content_data.get('estimated_time', 30)
            )
            db.session.add(content)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Course content generated successfully',
            'content_count': len(generated_content)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@teacher_bp.route('/courses/<int:course_id>/generate-questions', methods=['POST'])
@jwt_required()
def generate_questions():
    """AI生成考核题目"""
    try:
        teacher = get_current_teacher()
        if not teacher:
            return jsonify({'error': 'Teacher not found'}), 404
        
        course_id = request.view_args['course_id']
        course = Course.query.filter_by(id=course_id, teacher_id=teacher.id).first()
        if not course:
            return jsonify({'error': 'Course not found'}), 404
        
        data = request.get_json()
        question_count = data.get('question_count', 10)
        question_types = data.get('question_types', ['multiple_choice'])
        difficulty = data.get('difficulty', 'medium')
        
        # 创建练习
        exercise = Exercise(
            course_id=course.id,
            title=data.get('title', f'{course.title} - 练习'),
            description=data.get('description'),
            exercise_type='quiz',
            difficulty=difficulty,
            points=question_count * 10
        )
        db.session.add(exercise)
        db.session.flush()
        
        # 调用AI服务生成题目
        ai_service = AIService()
        generated_questions = ai_service.generate_questions(
            course_title=course.title,
            subject=course.subject,
            question_count=question_count,
            question_types=question_types,
            difficulty=difficulty
        )
        
        # 保存生成的题目
        for i, question_data in enumerate(generated_questions):
            question = Question(
                exercise_id=exercise.id,
                question_text=question_data['question'],
                question_type=question_data['type'],
                options=question_data.get('options'),
                correct_answer=question_data['answer'],
                explanation=question_data.get('explanation'),
                points=10,
                order_index=i
            )
            db.session.add(question)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Questions generated successfully',
            'exercise_id': exercise.id,
            'question_count': len(generated_questions)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@teacher_bp.route('/courses/<int:course_id>/students-analysis', methods=['GET'])
@jwt_required()
def analyze_students():
    """分析学生学习情况"""
    try:
        teacher = get_current_teacher()
        if not teacher:
            return jsonify({'error': 'Teacher not found'}), 404
        
        course_id = request.view_args['course_id']
        course = Course.query.filter_by(id=course_id, teacher_id=teacher.id).first()
        if not course:
            return jsonify({'error': 'Course not found'}), 404
        
        # 获取学习记录
        study_records = StudyRecord.query.filter_by(course_id=course.id).all()
        exercise_records = ExerciseRecord.query.join(Exercise).filter(
            Exercise.course_id == course.id
        ).all()
        
        # 调用AI服务进行分析
        ai_service = AIService()
        analysis = ai_service.analyze_student_performance(
            course_title=course.title,
            study_records=[record.to_dict() for record in study_records],
            exercise_records=[record.to_dict() for record in exercise_records]
        )
        
        # 保存分析报告
        report = AnalysisReport(
            report_type='course_analysis',
            target_id=course.id,
            target_type='course',
            title=f'{course.title} - 学情分析报告',
            content=analysis.get('content'),
            statistics=analysis.get('statistics'),
            recommendations=analysis.get('recommendations'),
            generated_by=get_jwt_identity()
        )
        db.session.add(report)
        db.session.commit()
        
        return jsonify({
            'analysis': analysis,
            'report_id': report.id
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
