import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createVuetify } from 'vuetify'
import { createI18n } from 'vue-i18n'

// Vuetify
import 'vuetify/styles'
import { aliases, mdi } from 'vuetify/iconsets/mdi'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'

// 应用组件和路由
import App from './App.vue'
import router from './router'

// 样式
import './assets/main.css'

// 国际化配置
const messages = {
  zh: {
    welcome: '欢迎使用教学实训智能体平台',
    login: '登录',
    logout: '退出登录',
    dashboard: '仪表板',
    courses: '课程管理',
    students: '学生管理',
    teachers: '教师管理',
    settings: '设置'
  },
  en: {
    welcome: 'Welcome to Teaching AI Platform',
    login: 'Login',
    logout: 'Logout',
    dashboard: 'Dashboard',
    courses: 'Course Management',
    students: 'Student Management',
    teachers: 'Teacher Management',
    settings: 'Settings'
  }
}

const i18n = createI18n({
  locale: 'zh',
  fallbackLocale: 'en',
  messages
})

// Vuetify配置
const vuetify = createVuetify({
  components,
  directives,
  icons: {
    defaultSet: 'mdi',
    aliases,
    sets: {
      mdi,
    },
  },
  theme: {
    defaultTheme: 'light',
    themes: {
      light: {
        colors: {
          primary: '#1976D2',
          secondary: '#424242',
          accent: '#82B1FF',
          error: '#FF5252',
          info: '#2196F3',
          success: '#4CAF50',
          warning: '#FFC107',
        },
      },
      dark: {
        colors: {
          primary: '#2196F3',
          secondary: '#424242',
          accent: '#FF4081',
          error: '#FF5252',
          info: '#2196F3',
          success: '#4CAF50',
          warning: '#FB8C00',
        },
      },
    },
  },
})

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(vuetify)
app.use(i18n)

app.mount('#app')
