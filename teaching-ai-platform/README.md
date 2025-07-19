# 教学实训智能体平台

一个基于Vue3 + Flask + PostgreSQL的智能教学平台，集成DeepSeek AI能力，为教师、学生和管理员提供全方位的教学支持。

## 功能特性

### 教师端
- **智能备课设计**：基于课程大纲和知识库自动生成教学内容
- **考核内容生成**：AI自动生成多样化考核题目及参考答案
- **学情数据分析**：智能分析学生学习情况，提供教学建议

### 学生端
- **在线学习助手**：AI助手回答学习问题
- **实时练习评测**：个性化练习生成和智能纠错

### 管理端
- **用户管理**：管理员、教师、学生用户管理
- **课件资源管理**：按学科管理和导出教学资源
- **数据大屏**：教学效率和学习效果统计分析

## 技术栈

### 后端
- **框架**：Flask 2.3+
- **数据库**：PostgreSQL 14+
- **ORM**：SQLAlchemy
- **认证**：Flask-JWT-Extended
- **AI集成**：DeepSeek API

### 前端
- **框架**：Vue 3.3+
- **UI库**：Vuetify 3.x
- **状态管理**：Pinia
- **路由**：Vue Router 4.x
- **构建工具**：Vite

## 项目结构

```
teaching-ai-platform/
├── backend/                    # Flask后端
│   ├── app/
│   │   ├── models/            # 数据模型
│   │   ├── routes/            # API路由
│   │   ├── services/          # 业务逻辑
│   │   └── utils/             # 工具函数
│   ├── config.py              # 配置文件
│   ├── requirements.txt       # Python依赖
│   └── run.py                 # 启动文件
├── frontend/                   # Vue3前端
│   ├── src/
│   │   ├── components/        # 组件
│   │   ├── views/             # 页面
│   │   ├── router/            # 路由
│   │   ├── store/             # 状态管理
│   │   └── utils/             # 工具函数
│   ├── package.json           # 前端依赖
│   └── vite.config.js         # Vite配置
└── docs/                      # 文档
```

## 快速开始

### 环境要求
- Python 3.8+
- Node.js 16+
- PostgreSQL 12+
- DeepSeek API Key

### 后端设置

1. 创建虚拟环境
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入实际配置
```

4. 初始化数据库
```bash
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

5. 启动后端服务
```bash
python run.py
```

### 前端设置

1. 安装依赖
```bash
cd frontend
npm install
```

2. 启动开发服务器
```bash
npm run dev
```

### 访问应用

- 前端地址：http://localhost:3000
- 后端API：http://localhost:5000/api

## 部署

### 生产环境部署

1. 构建前端
```bash
cd frontend
npm run build
```

2. 配置生产环境变量
```bash
export FLASK_CONFIG=production
export DATABASE_URL=your-production-db-url
export DEEPSEEK_API_KEY=your-api-key
```

3. 启动生产服务
```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

## API文档

### 认证接口
- `POST /api/auth/login` - 用户登录
- `POST /api/auth/register` - 用户注册
- `GET /api/auth/profile` - 获取用户资料
- `POST /api/auth/logout` - 用户登出

### 教师接口
- `GET /api/teacher/courses` - 获取课程列表
- `POST /api/teacher/courses` - 创建课程
- `POST /api/teacher/courses/{id}/generate-content` - 生成课程内容
- `POST /api/teacher/courses/{id}/generate-questions` - 生成考核题目

### 学生接口
- `GET /api/student/courses` - 获取可用课程
- `POST /api/student/exercises/{id}/submit` - 提交练习答案
- `POST /api/student/ask-question` - 向AI助手提问

### 管理接口
- `GET /api/admin/users` - 获取用户列表
- `GET /api/admin/dashboard/statistics` - 获取统计数据
- `GET /api/admin/export/courses` - 导出课程资源

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues：[GitHub Issues](https://github.com/your-repo/issues)
- 邮箱：your-email@example.com

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 实现基础的教师、学生、管理员功能
- 集成DeepSeek AI能力
- 完成前后端分离架构
