from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models.user import User, Student
from app.models.course import Course, CourseContent, Exercise, Question
from app.models.learning import StudyRecord, ExerciseRecord
from app.services.ai_service import AIService

student_bp = Blueprint('student', __name__)

def get_current_student():
    """获取当前学生信息"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user or user.role != 'student':
        return None
    return user.student_profile

@student_bp.route('/courses', methods=['GET'])
@jwt_required()
def get_available_courses():
    """获取可用课程列表"""
    try:
        student = get_current_student()
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        # 获取已发布的课程
        courses = Course.query.filter_by(is_published=True).all()
        
        # 获取学生的学习记录
        study_records = StudyRecord.query.filter_by(student_id=student.id).all()
        record_dict = {record.course_id: record for record in study_records}
        
        course_list = []
        for course in courses:
            course_data = course.to_dict()
            if course.id in record_dict:
                course_data['study_progress'] = record_dict[course.id].progress
                course_data['study_status'] = record_dict[course.id].status
            else:
                course_data['study_progress'] = 0.0
                course_data['study_status'] = 'not_started'
            course_list.append(course_data)
        
        return jsonify({
            'courses': course_list
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@student_bp.route('/courses/<int:course_id>/content', methods=['GET'])
@jwt_required()
def get_course_content():
    """获取课程内容"""
    try:
        student = get_current_student()
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        course_id = request.view_args['course_id']
        course = Course.query.filter_by(id=course_id, is_published=True).first()
        if not course:
            return jsonify({'error': 'Course not found'}), 404
        
        # 获取课程内容
        contents = CourseContent.query.filter_by(course_id=course.id).order_by(
            CourseContent.order_index
        ).all()
        
        # 获取学习记录
        study_records = StudyRecord.query.filter_by(
            student_id=student.id,
            course_id=course.id
        ).all()
        record_dict = {record.content_id: record for record in study_records}
        
        content_list = []
        for content in contents:
            content_data = content.to_dict()
            if content.id in record_dict:
                content_data['study_status'] = record_dict[content.id].status
                content_data['study_time'] = record_dict[content.id].study_time
            else:
                content_data['study_status'] = 'not_started'
                content_data['study_time'] = 0
            content_list.append(content_data)
        
        return jsonify({
            'course': course.to_dict(),
            'contents': content_list
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@student_bp.route('/courses/<int:course_id>/exercises', methods=['GET'])
@jwt_required()
def get_course_exercises():
    """获取课程练习"""
    try:
        student = get_current_student()
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        course_id = request.view_args['course_id']
        course = Course.query.filter_by(id=course_id, is_published=True).first()
        if not course:
            return jsonify({'error': 'Course not found'}), 404
        
        exercises = Exercise.query.filter_by(course_id=course.id).all()
        
        # 获取练习记录
        exercise_records = ExerciseRecord.query.filter_by(student_id=student.id).all()
        record_dict = {}
        for record in exercise_records:
            if record.exercise_id not in record_dict:
                record_dict[record.exercise_id] = []
            record_dict[record.exercise_id].append(record)
        
        exercise_list = []
        for exercise in exercises:
            exercise_data = exercise.to_dict()
            if exercise.id in record_dict:
                records = record_dict[exercise.id]
                exercise_data['attempt_count'] = len(records)
                exercise_data['best_score'] = max([r.score for r in records])
                exercise_data['last_attempt'] = max([r.submitted_at for r in records]).isoformat()
            else:
                exercise_data['attempt_count'] = 0
                exercise_data['best_score'] = 0
                exercise_data['last_attempt'] = None
            exercise_list.append(exercise_data)
        
        return jsonify({
            'exercises': exercise_list
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@student_bp.route('/exercises/<int:exercise_id>/questions', methods=['GET'])
@jwt_required()
def get_exercise_questions():
    """获取练习题目"""
    try:
        student = get_current_student()
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        exercise_id = request.view_args['exercise_id']
        exercise = Exercise.query.get(exercise_id)
        if not exercise:
            return jsonify({'error': 'Exercise not found'}), 404
        
        questions = Question.query.filter_by(exercise_id=exercise.id).order_by(
            Question.order_index
        ).all()
        
        # 不返回正确答案给学生
        question_list = []
        for question in questions:
            question_data = question.to_dict()
            question_data.pop('correct_answer', None)
            question_data.pop('explanation', None)
            question_list.append(question_data)
        
        return jsonify({
            'exercise': exercise.to_dict(),
            'questions': question_list
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@student_bp.route('/exercises/<int:exercise_id>/submit', methods=['POST'])
@jwt_required()
def submit_exercise():
    """提交练习答案"""
    try:
        student = get_current_student()
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        exercise_id = request.view_args['exercise_id']
        exercise = Exercise.query.get(exercise_id)
        if not exercise:
            return jsonify({'error': 'Exercise not found'}), 404
        
        data = request.get_json()
        answers = data.get('answers', {})  # {question_id: answer}
        
        questions = Question.query.filter_by(exercise_id=exercise.id).all()
        total_score = 0
        results = []
        
        # 调用AI服务进行答案评估
        ai_service = AIService()
        
        for question in questions:
            question_id = str(question.id)
            student_answer = answers.get(question_id, '')
            
            # 评估答案
            evaluation = ai_service.evaluate_answer(
                question_text=question.question_text,
                question_type=question.question_type,
                correct_answer=question.correct_answer,
                student_answer=student_answer
            )
            
            is_correct = evaluation['is_correct']
            score = question.points if is_correct else 0
            feedback = evaluation.get('feedback', '')
            
            # 保存练习记录
            record = ExerciseRecord(
                student_id=student.id,
                exercise_id=exercise.id,
                question_id=question.id,
                student_answer=student_answer,
                is_correct=is_correct,
                score=score,
                feedback=feedback
            )
            db.session.add(record)
            
            total_score += score
            results.append({
                'question_id': question.id,
                'is_correct': is_correct,
                'score': score,
                'feedback': feedback,
                'correct_answer': question.correct_answer,
                'explanation': question.explanation
            })
        
        db.session.commit()
        
        return jsonify({
            'message': 'Exercise submitted successfully',
            'total_score': total_score,
            'max_score': exercise.points,
            'percentage': (total_score / exercise.points) * 100 if exercise.points > 0 else 0,
            'results': results
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@student_bp.route('/ask-question', methods=['POST'])
@jwt_required()
def ask_question():
    """向AI助手提问"""
    try:
        student = get_current_student()
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        data = request.get_json()
        question = data.get('question', '')
        course_id = data.get('course_id')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # 获取课程上下文
        context = ''
        if course_id:
            course = Course.query.get(course_id)
            if course:
                context = f"课程：{course.title}，学科：{course.subject}"
        
        # 调用AI服务回答问题
        ai_service = AIService()
        answer = ai_service.answer_student_question(
            question=question,
            context=context
        )
        
        return jsonify({
            'question': question,
            'answer': answer
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
