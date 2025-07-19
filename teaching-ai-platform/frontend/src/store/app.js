import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAppStore = defineStore('app', () => {
  // 全局加载状态
  const loading = ref(false)
  
  // 全局通知
  const snackbar = ref({
    show: false,
    text: '',
    color: 'info',
    timeout: 4000
  })
  
  // 主题设置
  const theme = ref(localStorage.getItem('theme') || 'light')
  
  // 语言设置
  const locale = ref(localStorage.getItem('locale') || 'zh')
  
  // 设置加载状态
  const setLoading = (status) => {
    loading.value = status
  }
  
  // 显示通知
  const showSnackbar = (text, color = 'info', timeout = 4000) => {
    snackbar.value = {
      show: true,
      text,
      color,
      timeout
    }
  }
  
  // 显示成功消息
  const showSuccess = (text) => {
    showSnackbar(text, 'success')
  }
  
  // 显示错误消息
  const showError = (text) => {
    showSnackbar(text, 'error', 6000)
  }
  
  // 显示警告消息
  const showWarning = (text) => {
    showSnackbar(text, 'warning')
  }
  
  // 显示信息消息
  const showInfo = (text) => {
    showSnackbar(text, 'info')
  }
  
  // 切换主题
  const toggleTheme = () => {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
    localStorage.setItem('theme', theme.value)
  }
  
  // 设置主题
  const setTheme = (newTheme) => {
    theme.value = newTheme
    localStorage.setItem('theme', newTheme)
  }
  
  // 设置语言
  const setLocale = (newLocale) => {
    locale.value = newLocale
    localStorage.setItem('locale', newLocale)
  }
  
  return {
    // 状态
    loading,
    snackbar,
    theme,
    locale,
    
    // 方法
    setLoading,
    showSnackbar,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    toggleTheme,
    setTheme,
    setLocale
  }
})
