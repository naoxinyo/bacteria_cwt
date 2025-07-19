<template>
  <v-app>
    <v-navigation-drawer
      v-if="isAuthenticated"
      v-model="drawer"
      app
      clipped
      :mini-variant="miniVariant"
    >
      <v-list>
        <v-list-item
          v-for="item in menuItems"
          :key="item.title"
          :to="item.to"
          link
        >
          <v-list-item-action>
            <v-icon>{{ item.icon }}</v-icon>
          </v-list-item-action>
          <v-list-item-content>
            <v-list-item-title>{{ item.title }}</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-navigation-drawer>

    <v-app-bar
      v-if="isAuthenticated"
      app
      clipped-left
      color="primary"
      dark
    >
      <v-app-bar-nav-icon @click="drawer = !drawer" />
      <v-btn icon @click="miniVariant = !miniVariant">
        <v-icon>mdi-chevron-left</v-icon>
      </v-btn>
      
      <v-toolbar-title>教学实训智能体平台</v-toolbar-title>
      
      <v-spacer />
      
      <v-menu offset-y>
        <template v-slot:activator="{ on, attrs }">
          <v-btn
            icon
            v-bind="attrs"
            v-on="on"
          >
            <v-avatar size="32">
              <v-img
                v-if="user?.avatar"
                :src="user.avatar"
                :alt="user.real_name"
              />
              <v-icon v-else>mdi-account-circle</v-icon>
            </v-avatar>
          </v-btn>
        </template>
        
        <v-list>
          <v-list-item>
            <v-list-item-content>
              <v-list-item-title>{{ user?.real_name }}</v-list-item-title>
              <v-list-item-subtitle>{{ user?.role }}</v-list-item-subtitle>
            </v-list-item-content>
          </v-list-item>
          
          <v-divider />
          
          <v-list-item @click="goToProfile">
            <v-list-item-action>
              <v-icon>mdi-account</v-icon>
            </v-list-item-action>
            <v-list-item-content>
              <v-list-item-title>个人资料</v-list-item-title>
            </v-list-item-content>
          </v-list-item>
          
          <v-list-item @click="logout">
            <v-list-item-action>
              <v-icon>mdi-logout</v-icon>
            </v-list-item-action>
            <v-list-item-content>
              <v-list-item-title>退出登录</v-list-item-title>
            </v-list-item-content>
          </v-list-item>
        </v-list>
      </v-menu>
    </v-app-bar>

    <v-main>
      <v-container fluid>
        <router-view />
      </v-container>
    </v-main>

    <v-snackbar
      v-model="snackbar.show"
      :color="snackbar.color"
      :timeout="snackbar.timeout"
    >
      {{ snackbar.text }}
      <template v-slot:action="{ attrs }">
        <v-btn
          text
          v-bind="attrs"
          @click="snackbar.show = false"
        >
          关闭
        </v-btn>
      </template>
    </v-snackbar>
  </v-app>
</template>

<script>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/store/auth'
import { useAppStore } from '@/store/app'

export default {
  name: 'App',
  setup() {
    const router = useRouter()
    const authStore = useAuthStore()
    const appStore = useAppStore()
    
    const drawer = ref(true)
    const miniVariant = ref(false)
    
    const isAuthenticated = computed(() => authStore.isAuthenticated)
    const user = computed(() => authStore.user)
    const snackbar = computed(() => appStore.snackbar)
    
    const menuItems = computed(() => {
      const role = user.value?.role
      const baseItems = [
        { title: '仪表板', icon: 'mdi-view-dashboard', to: '/dashboard' }
      ]
      
      if (role === 'teacher') {
        return [
          ...baseItems,
          { title: '课程管理', icon: 'mdi-book-multiple', to: '/teacher/courses' },
          { title: '备课设计', icon: 'mdi-pencil-box-multiple', to: '/teacher/design' },
          { title: '学情分析', icon: 'mdi-chart-line', to: '/teacher/analysis' }
        ]
      } else if (role === 'student') {
        return [
          ...baseItems,
          { title: '我的课程', icon: 'mdi-book-open-page-variant', to: '/student/courses' },
          { title: '练习中心', icon: 'mdi-clipboard-text', to: '/student/exercises' },
          { title: '学习助手', icon: 'mdi-robot', to: '/student/assistant' }
        ]
      } else if (role === 'admin') {
        return [
          ...baseItems,
          { title: '用户管理', icon: 'mdi-account-group', to: '/admin/users' },
          { title: '课程管理', icon: 'mdi-book-multiple', to: '/admin/courses' },
          { title: '数据大屏', icon: 'mdi-monitor-dashboard', to: '/admin/overview' }
        ]
      }
      
      return baseItems
    })
    
    const goToProfile = () => {
      router.push('/profile')
    }
    
    const logout = async () => {
      await authStore.logout()
      router.push('/login')
    }
    
    return {
      drawer,
      miniVariant,
      isAuthenticated,
      user,
      snackbar,
      menuItems,
      goToProfile,
      logout
    }
  }
}
</script>

<style scoped>
.v-application {
  font-family: 'Roboto', sans-serif;
}
</style>
