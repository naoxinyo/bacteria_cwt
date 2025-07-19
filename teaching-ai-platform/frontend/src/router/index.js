import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/store/auth'

// 布局组件
const DefaultLayout = () => import('@/components/layouts/DefaultLayout.vue')

// 认证相关页面
const Login = () => import('@/views/auth/Login.vue')
const Register = () => import('@/views/auth/Register.vue')

// 通用页面
const Dashboard = () => import('@/views/Dashboard.vue')
const Profile = () => import('@/views/Profile.vue')

// 教师端页面
const TeacherCourses = () => import('@/views/teacher/Courses.vue')
const TeacherDesign = () => import('@/views/teacher/Design.vue')
const TeacherAnalysis = () => import('@/views/teacher/Analysis.vue')

// 学生端页面
const StudentCourses = () => import('@/views/student/Courses.vue')
const StudentExercises = () => import('@/views/student/Exercises.vue')
const StudentAssistant = () => import('@/views/student/Assistant.vue')

// 管理端页面
const AdminUsers = () => import('@/views/admin/Users.vue')
const AdminCourses = () => import('@/views/admin/Courses.vue')
const AdminOverview = () => import('@/views/admin/Overview.vue')

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: Login,
    meta: { requiresAuth: false }
  },
  {
    path: '/register',
    name: 'Register',
    component: Register,
    meta: { requiresAuth: false }
  },
  {
    path: '/',
    component: DefaultLayout,
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        redirect: '/dashboard'
      },
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: Dashboard
      },
      {
        path: 'profile',
        name: 'Profile',
        component: Profile
      },
      // 教师端路由
      {
        path: 'teacher',
        meta: { roles: ['teacher'] },
        children: [
          {
            path: 'courses',
            name: 'TeacherCourses',
            component: TeacherCourses
          },
          {
            path: 'design',
            name: 'TeacherDesign',
            component: TeacherDesign
          },
          {
            path: 'analysis',
            name: 'TeacherAnalysis',
            component: TeacherAnalysis
          }
        ]
      },
      // 学生端路由
      {
        path: 'student',
        meta: { roles: ['student'] },
        children: [
          {
            path: 'courses',
            name: 'StudentCourses',
            component: StudentCourses
          },
          {
            path: 'exercises',
            name: 'StudentExercises',
            component: StudentExercises
          },
          {
            path: 'assistant',
            name: 'StudentAssistant',
            component: StudentAssistant
          }
        ]
      },
      // 管理端路由
      {
        path: 'admin',
        meta: { roles: ['admin'] },
        children: [
          {
            path: 'users',
            name: 'AdminUsers',
            component: AdminUsers
          },
          {
            path: 'courses',
            name: 'AdminCourses',
            component: AdminCourses
          },
          {
            path: 'overview',
            name: 'AdminOverview',
            component: AdminOverview
          }
        ]
      }
    ]
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFound.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()
  
  // 检查是否需要认证
  if (to.meta.requiresAuth !== false) {
    if (!authStore.isAuthenticated) {
      // 尝试从本地存储恢复认证状态
      await authStore.checkAuth()
      
      if (!authStore.isAuthenticated) {
        next('/login')
        return
      }
    }
    
    // 检查角色权限
    if (to.meta.roles && to.meta.roles.length > 0) {
      const userRole = authStore.user?.role
      if (!to.meta.roles.includes(userRole)) {
        next('/dashboard')
        return
      }
    }
  }
  
  // 如果已登录用户访问登录页，重定向到仪表板
  if (to.name === 'Login' && authStore.isAuthenticated) {
    next('/dashboard')
    return
  }
  
  next()
})

export default router
