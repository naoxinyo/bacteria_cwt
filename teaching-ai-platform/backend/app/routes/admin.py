from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import func, desc
from app import db
from app.models.user import User, Teacher, Student, Admin
from app.models.course import Course, Exercise
from app.models.learning import StudyRecord, ExerciseRecord, AnalysisReport

admin_bp = Blueprint('admin', __name__)

def get_current_admin():
    """获取当前管理员信息"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user or user.role != 'admin':
        return None
    return user

@admin_bp.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    """获取用户列表"""
    try:
        admin = get_current_admin()
        if not admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        role = request.args.get('role')
        search = request.args.get('search')
        
        query = User.query
        
        if role:
            query = query.filter_by(role=role)
        
        if search:
            query = query.filter(
                (User.username.contains(search)) |
                (User.real_name.contains(search)) |
                (User.email.contains(search))
            )
        
        users = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'users': [user.to_dict() for user in users.items],
            'total': users.total,
            'pages': users.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users/<int:user_id>', methods=['PUT'])
@jwt_required()
def update_user():
    """更新用户信息"""
    try:
        admin = get_current_admin()
        if not admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        user_id = request.view_args['user_id']
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # 更新基本信息
        if 'real_name' in data:
            user.real_name = data['real_name']
        if 'email' in data:
            user.email = data['email']
        if 'phone' in data:
            user.phone = data['phone']
        if 'is_active' in data:
            user.is_active = data['is_active']
        
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'User updated successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/courses', methods=['GET'])
@jwt_required()
def get_all_courses():
    """获取所有课程"""
    try:
        admin = get_current_admin()
        if not admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        subject = request.args.get('subject')
        
        query = Course.query
        
        if subject:
            query = query.filter_by(subject=subject)
        
        courses = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        course_list = []
        for course in courses.items:
            course_data = course.to_dict()
            # 添加教师信息
            if course.teacher:
                course_data['teacher_name'] = course.teacher.user.real_name
            course_list.append(course_data)
        
        return jsonify({
            'courses': course_list,
            'total': courses.total,
            'pages': courses.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/dashboard/statistics', methods=['GET'])
@jwt_required()
def get_dashboard_statistics():
    """获取仪表板统计数据"""
    try:
        admin = get_current_admin()
        if not admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        # 基础统计
        total_users = User.query.count()
        total_teachers = User.query.filter_by(role='teacher').count()
        total_students = User.query.filter_by(role='student').count()
        total_courses = Course.query.count()
        published_courses = Course.query.filter_by(is_published=True).count()
        
        # 今日活跃用户
        today = datetime.utcnow().date()
        today_active_users = User.query.filter(
            func.date(User.last_login) == today
        ).count()
        
        # 本周活跃用户
        week_ago = datetime.utcnow() - timedelta(days=7)
        week_active_users = User.query.filter(
            User.last_login >= week_ago
        ).count()
        
        # 教师使用统计
        teacher_usage = db.session.query(
            User.id,
            User.real_name,
            func.count(Course.id).label('course_count')
        ).join(Teacher).join(Course).group_by(User.id, User.real_name).all()
        
        # 学生学习统计
        student_stats = db.session.query(
            func.count(StudyRecord.id).label('total_studies'),
            func.avg(StudyRecord.progress).label('avg_progress')
        ).first()
        
        # 练习统计
        exercise_stats = db.session.query(
            func.count(ExerciseRecord.id).label('total_submissions'),
            func.avg(ExerciseRecord.score).label('avg_score')
        ).first()
        
        # 最近7天的学习趋势
        learning_trend = []
        for i in range(7):
            date = datetime.utcnow().date() - timedelta(days=i)
            count = StudyRecord.query.filter(
                func.date(StudyRecord.started_at) == date
            ).count()
            learning_trend.append({
                'date': date.isoformat(),
                'count': count
            })
        
        return jsonify({
            'basic_stats': {
                'total_users': total_users,
                'total_teachers': total_teachers,
                'total_students': total_students,
                'total_courses': total_courses,
                'published_courses': published_courses
            },
            'activity_stats': {
                'today_active_users': today_active_users,
                'week_active_users': week_active_users
            },
            'teacher_usage': [
                {
                    'teacher_id': usage.id,
                    'teacher_name': usage.real_name,
                    'course_count': usage.course_count
                }
                for usage in teacher_usage[:10]  # 前10名
            ],
            'learning_stats': {
                'total_studies': student_stats.total_studies or 0,
                'avg_progress': float(student_stats.avg_progress or 0),
                'total_submissions': exercise_stats.total_submissions or 0,
                'avg_score': float(exercise_stats.avg_score or 0)
            },
            'learning_trend': learning_trend
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/export/courses', methods=['GET'])
@jwt_required()
def export_courses():
    """导出课程资源"""
    try:
        admin = get_current_admin()
        if not admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        subject = request.args.get('subject')
        teacher_id = request.args.get('teacher_id', type=int)
        
        query = Course.query
        
        if subject:
            query = query.filter_by(subject=subject)
        if teacher_id:
            query = query.join(Teacher).filter(Teacher.id == teacher_id)
        
        courses = query.all()
        
        export_data = []
        for course in courses:
            course_data = course.to_dict()
            course_data['teacher_name'] = course.teacher.user.real_name
            course_data['content_count'] = course.contents.count()
            course_data['exercise_count'] = course.exercises.count()
            export_data.append(course_data)
        
        return jsonify({
            'courses': export_data,
            'export_time': datetime.utcnow().isoformat(),
            'total_count': len(export_data)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/reports', methods=['GET'])
@jwt_required()
def get_reports():
    """获取分析报告列表"""
    try:
        admin = get_current_admin()
        if not admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        report_type = request.args.get('report_type')
        
        query = AnalysisReport.query
        
        if report_type:
            query = query.filter_by(report_type=report_type)
        
        reports = query.order_by(desc(AnalysisReport.created_at)).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'reports': [report.to_dict() for report in reports.items],
            'total': reports.total,
            'pages': reports.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
