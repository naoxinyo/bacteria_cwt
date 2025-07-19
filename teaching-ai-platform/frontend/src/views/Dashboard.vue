<template>
  <div>
    <v-row>
      <v-col cols="12">
        <h1 class="text-h4 mb-4">
          欢迎回来，{{ user?.real_name }}！
        </h1>
      </v-col>
    </v-row>
    
    <!-- 角色特定的仪表板内容 -->
    <div v-if="user?.role === 'teacher'">
      <TeacherDashboard />
    </div>
    
    <div v-else-if="user?.role === 'student'">
      <StudentDashboard />
    </div>
    
    <div v-else-if="user?.role === 'admin'">
      <AdminDashboard />
    </div>
    
    <div v-else>
      <v-alert type="warning" class="mb-4">
        未识别的用户角色，请联系管理员
      </v-alert>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useAuthStore } from '@/store/auth'
import TeacherDashboard from '@/components/dashboard/TeacherDashboard.vue'
import StudentDashboard from '@/components/dashboard/StudentDashboard.vue'
import AdminDashboard from '@/components/dashboard/AdminDashboard.vue'

export default {
  name: 'Dashboard',
  components: {
    TeacherDashboard,
    StudentDashboard,
    AdminDashboard
  },
  setup() {
    const authStore = useAuthStore()
    
    const user = computed(() => authStore.user)
    
    return {
      user
    }
  }
}
</script>
