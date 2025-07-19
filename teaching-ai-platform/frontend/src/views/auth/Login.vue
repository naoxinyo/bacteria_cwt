<template>
  <v-container fluid class="fill-height">
    <v-row align="center" justify="center">
      <v-col cols="12" sm="8" md="6" lg="4">
        <v-card class="elevation-12">
          <v-card-title class="text-center">
            <h2 class="primary--text">教学实训智能体平台</h2>
          </v-card-title>
          
          <v-card-text>
            <v-form ref="form" v-model="valid" @submit.prevent="handleLogin">
              <v-text-field
                v-model="form.username"
                :rules="usernameRules"
                label="用户名或邮箱"
                prepend-icon="mdi-account"
                required
                outlined
              />
              
              <v-text-field
                v-model="form.password"
                :rules="passwordRules"
                :type="showPassword ? 'text' : 'password'"
                label="密码"
                prepend-icon="mdi-lock"
                :append-icon="showPassword ? 'mdi-eye' : 'mdi-eye-off'"
                @click:append="showPassword = !showPassword"
                required
                outlined
              />
              
              <v-row>
                <v-col cols="6">
                  <v-checkbox
                    v-model="form.remember"
                    label="记住我"
                    hide-details
                  />
                </v-col>
                <v-col cols="6" class="text-right">
                  <v-btn
                    text
                    color="primary"
                    @click="forgotPassword"
                  >
                    忘记密码？
                  </v-btn>
                </v-col>
              </v-row>
            </v-form>
          </v-card-text>
          
          <v-card-actions>
            <v-spacer />
            <v-btn
              color="primary"
              :loading="loading"
              :disabled="!valid"
              large
              block
              @click="handleLogin"
            >
              登录
            </v-btn>
          </v-card-actions>
          
          <v-divider class="mx-4" />
          
          <v-card-text class="text-center">
            <span>还没有账号？</span>
            <v-btn
              text
              color="primary"
              @click="goToRegister"
            >
              立即注册
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/store/auth'
import { useAppStore } from '@/store/app'

export default {
  name: 'Login',
  setup() {
    const router = useRouter()
    const authStore = useAuthStore()
    const appStore = useAppStore()
    
    const form = reactive({
      username: '',
      password: '',
      remember: false
    })
    
    const valid = ref(false)
    const showPassword = ref(false)
    const loading = ref(false)
    
    const usernameRules = [
      v => !!v || '请输入用户名或邮箱',
      v => v.length >= 3 || '用户名至少3个字符'
    ]
    
    const passwordRules = [
      v => !!v || '请输入密码',
      v => v.length >= 6 || '密码至少6个字符'
    ]
    
    const handleLogin = async () => {
      if (!valid.value) return
      
      loading.value = true
      
      try {
        const result = await authStore.login({
          username: form.username,
          password: form.password
        })
        
        if (result.success) {
          appStore.showSuccess('登录成功')
          router.push('/dashboard')
        } else {
          appStore.showError(result.message)
        }
      } catch (error) {
        appStore.showError('登录失败，请稍后重试')
      } finally {
        loading.value = false
      }
    }
    
    const goToRegister = () => {
      router.push('/register')
    }
    
    const forgotPassword = () => {
      appStore.showInfo('请联系管理员重置密码')
    }
    
    return {
      form,
      valid,
      showPassword,
      loading,
      usernameRules,
      passwordRules,
      handleLogin,
      goToRegister,
      forgotPassword
    }
  }
}
</script>

<style scoped>
.fill-height {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.v-card {
  border-radius: 16px;
}
</style>
