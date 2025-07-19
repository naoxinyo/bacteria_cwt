from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from app import db
from app.models.user import User, Teacher, Student, Admin

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """用户注册"""
    try:
        data = request.get_json()
        
        # 验证必填字段
        required_fields = ['username', 'email', 'password', 'real_name', 'role']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # 检查用户名和邮箱是否已存在
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # 创建用户
        user = User(
            username=data['username'],
            email=data['email'],
            real_name=data['real_name'],
            phone=data.get('phone'),
            role=data['role']
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.flush()  # 获取用户ID
        
        # 根据角色创建扩展信息
        if data['role'] == 'teacher':
            teacher = Teacher(
                user_id=user.id,
                employee_id=data.get('employee_id'),
                department=data.get('department'),
                title=data.get('title')
            )
            db.session.add(teacher)
        elif data['role'] == 'student':
            student = Student(
                user_id=user.id,
                student_id=data.get('student_id'),
                grade=data.get('grade'),
                major=data.get('major'),
                class_name=data.get('class_name'),
                enrollment_year=data.get('enrollment_year')
            )
            db.session.add(student)
        elif data['role'] == 'admin':
            admin = Admin(
                user_id=user.id,
                permissions=data.get('permissions', '[]')
            )
            db.session.add(admin)
        
        db.session.commit()
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password are required'}), 400
        
        # 查找用户（支持用户名或邮箱登录）
        user = User.query.filter(
            (User.username == data['username']) | (User.email == data['username'])
        ).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is disabled'}), 401
        
        # 更新最后登录时间
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # 生成JWT令牌
        access_token = create_access_token(identity=user.id)
        refresh_token = create_refresh_token(identity=user.id)
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'refresh_token': refresh_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """刷新访问令牌"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user or not user.is_active:
            return jsonify({'error': 'User not found or inactive'}), 404
        
        access_token = create_access_token(identity=current_user_id)
        
        return jsonify({
            'access_token': access_token
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """获取用户资料"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        profile = user.to_dict()
        
        # 添加角色特定信息
        if user.role == 'teacher' and user.teacher_profile:
            profile['teacher_info'] = {
                'employee_id': user.teacher_profile.employee_id,
                'department': user.teacher_profile.department,
                'title': user.teacher_profile.title,
                'specialties': user.teacher_profile.specialties,
                'bio': user.teacher_profile.bio
            }
        elif user.role == 'student' and user.student_profile:
            profile['student_info'] = {
                'student_id': user.student_profile.student_id,
                'grade': user.student_profile.grade,
                'major': user.student_profile.major,
                'class_name': user.student_profile.class_name,
                'enrollment_year': user.student_profile.enrollment_year
            }
        elif user.role == 'admin' and user.admin_profile:
            profile['admin_info'] = {
                'permissions': user.admin_profile.permissions
            }
        
        return jsonify(profile), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """用户登出"""
    # 在实际应用中，可以将令牌加入黑名单
    return jsonify({'message': 'Logout successful'}), 200
