import os
from app import create_app, db
from flask_migrate import upgrade

def deploy():
    """部署函数"""
    app = create_app(os.getenv('FLASK_CONFIG') or 'default')
    
    with app.app_context():
        # 创建数据库表
        db.create_all()
        
        # 运行数据库迁移
        upgrade()

if __name__ == '__main__':
    app = create_app(os.getenv('FLASK_CONFIG') or 'development')
    app.run(host='0.0.0.0', port=5000, debug=True)
