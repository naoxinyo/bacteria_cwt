import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/utils/api'

export const useAuthStore = defineStore('auth', () => {
  // 状态
  const user = ref(null)
  const token = ref(localStorage.getItem('access_token'))
  const refreshToken = ref(localStorage.getItem('refresh_token'))
  const loading = ref(false)
  
  // 计算属性
  const isAuthenticated = computed(() => !!token.value && !!user.value)
  
  // 设置认证信息
  const setAuth = (authData) => {
    user.value = authData.user
    token.value = authData.access_token
    refreshToken.value = authData.refresh_token
    
    localStorage.setItem('access_token', authData.access_token)
    localStorage.setItem('refresh_token', authData.refresh_token)
    
    // 设置API默认请求头
    api.defaults.headers.common['Authorization'] = `Bearer ${authData.access_token}`
  }
  
  // 清除认证信息
  const clearAuth = () => {
    user.value = null
    token.value = null
    refreshToken.value = null
    
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    
    delete api.defaults.headers.common['Authorization']
  }
  
  // 登录
  const login = async (credentials) => {
    try {
      loading.value = true
      const response = await api.post('/auth/login', credentials)
      
      if (response.data.access_token) {
        setAuth(response.data)
        return { success: true, message: response.data.message }
      } else {
        throw new Error('登录响应格式错误')
      }
    } catch (error) {
      const message = error.response?.data?.error || error.message || '登录失败'
      return { success: false, message }
    } finally {
      loading.value = false
    }
  }
  
  // 注册
  const register = async (userData) => {
    try {
      loading.value = true
      const response = await api.post('/auth/register', userData)
      return { success: true, message: response.data.message }
    } catch (error) {
      const message = error.response?.data?.error || error.message || '注册失败'
      return { success: false, message }
    } finally {
      loading.value = false
    }
  }
  
  // 登出
  const logout = async () => {
    try {
      if (token.value) {
        await api.post('/auth/logout')
      }
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      clearAuth()
    }
  }
  
  // 刷新令牌
  const refreshAccessToken = async () => {
    try {
      if (!refreshToken.value) {
        throw new Error('No refresh token available')
      }
      
      const response = await api.post('/auth/refresh', {}, {
        headers: {
          'Authorization': `Bearer ${refreshToken.value}`
        }
      })
      
      if (response.data.access_token) {
        token.value = response.data.access_token
        localStorage.setItem('access_token', response.data.access_token)
        api.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`
        return true
      }
      
      return false
    } catch (error) {
      console.error('Token refresh failed:', error)
      clearAuth()
      return false
    }
  }
  
  // 获取用户资料
  const fetchProfile = async () => {
    try {
      const response = await api.get('/auth/profile')
      user.value = response.data
      return response.data
    } catch (error) {
      console.error('Fetch profile error:', error)
      if (error.response?.status === 401) {
        // 尝试刷新令牌
        const refreshed = await refreshAccessToken()
        if (refreshed) {
          return await fetchProfile()
        } else {
          clearAuth()
        }
      }
      throw error
    }
  }
  
  // 检查认证状态
  const checkAuth = async () => {
    if (token.value) {
      try {
        api.defaults.headers.common['Authorization'] = `Bearer ${token.value}`
        await fetchProfile()
        return true
      } catch (error) {
        console.error('Auth check failed:', error)
        clearAuth()
        return false
      }
    }
    return false
  }
  
  // 更新用户资料
  const updateProfile = async (profileData) => {
    try {
      loading.value = true
      const response = await api.put('/auth/profile', profileData)
      user.value = { ...user.value, ...response.data.user }
      return { success: true, message: '资料更新成功' }
    } catch (error) {
      const message = error.response?.data?.error || error.message || '更新失败'
      return { success: false, message }
    } finally {
      loading.value = false
    }
  }
  
  return {
    // 状态
    user,
    token,
    loading,
    
    // 计算属性
    isAuthenticated,
    
    // 方法
    login,
    register,
    logout,
    refreshAccessToken,
    fetchProfile,
    checkAuth,
    updateProfile,
    setAuth,
    clearAuth
  }
})
