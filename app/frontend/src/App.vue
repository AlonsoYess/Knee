<script setup lang="ts">
import { computed, ref } from 'vue'
import { isNavigationFailure, NavigationFailureType, useRoute, useRouter } from 'vue-router'
import AppIcon from './components/AppIcon.vue'
import ApplicationStatus from './components/ApplicationStatus.vue'
import { apiMessage, closeSession, serverSession } from './state/session'
import { closeOnBackdrop } from './dialog'
import { resetDemoListPreferences } from './demoListPreferences'
import { evaluationIssue, legacyDraftsPresent } from './state/evaluations'

const route = useRoute()
const router = useRouter()
const isLogin = computed(() => route.name === 'login')
const isEvaluationSection = computed(() => route.path === '/evaluaciones' || route.path.startsWith('/evaluaciones/'))
const help = ref<HTMLDialogElement | null>(null)
const loggingOut = ref(false)
const logoutError = ref('')
const initials = computed(() => serverSession.value?.user?.displayName.split(/\s+/).filter(Boolean).slice(0, 2).map(part => part[0]).join('').toLocaleUpperCase('es') || 'U')
async function logout() {
  if (loggingOut.value) return
  logoutError.value = ''
  const failure = await router.push({ name: 'evaluations' })
  if (failure && !isNavigationFailure(failure, NavigationFailureType.duplicated)) return
  if (route.name !== 'evaluations') return
  loggingOut.value = true
  try {
    await closeSession()
    resetDemoListPreferences()
    await router.replace('/acceso')
  } catch (error) { logoutError.value = apiMessage(error) }
  finally { loggingOut.value = false }
}
</script>

<template>
  <a class="skip-link" href="#main-content">Saltar al contenido</a>
  <div class="demo-ribbon"><span class="demo-ribbon-dot" aria-hidden="true"></span>Prototipo académico · sesión real<span class="demo-ribbon-detail">Solo casos ficticios · predicción bloqueada</span></div>
  <ApplicationStatus />
  <RouterView v-if="isLogin" />
  <div v-else class="workspace">
    <header class="workspace-header">
      <RouterLink to="/evaluaciones" class="brand" aria-label="KneeOAI, mis evaluaciones"><span class="brand-mark"><AppIcon name="activity" :size="23" /></span><span>Knee<span class="brand-light">OAI</span></span></RouterLink>
      <div class="header-account"><span class="account-avatar" aria-hidden="true">{{ initials }}</span><div class="account-text"><strong>{{ serverSession?.user?.displayName }}</strong><span>{{ serverSession?.user?.role === 'administrator' ? 'Administrador' : 'Médico' }}</span></div><button class="icon-button logout-button" aria-label="Cerrar sesión" :disabled="loggingOut" @click="logout"><AppIcon name="logout" /></button></div>
    </header>
    <div class="workspace-body">
      <aside class="workspace-sidebar">
        <div><p class="nav-caption">ESPACIO DE TRABAJO</p><nav aria-label="Navegación principal"><RouterLink class="nav-item" :class="{ 'section-active': isEvaluationSection }" :aria-current="isEvaluationSection ? (route.name === 'evaluations' ? 'page' : 'location') : undefined" to="/evaluaciones"><AppIcon name="evaluations" />Mis evaluaciones</RouterLink><button class="nav-item" @click="help?.showModal()"><AppIcon name="help" />Ayuda y alcance</button></nav></div>
        <div class="sidebar-note"><AppIcon name="shield" :size="19" /><p>Prototipo académico<span>Progresión estructural a 48 meses</span></p></div>
      </aside>
      <main id="main-content" class="workspace-main"><div v-if="logoutError" id="session-action-error" class="demo-storage-alert" role="alert">{{ logoutError }}</div><div v-if="legacyDraftsPresent" id="legacy-drafts-notice" class="legacy-notice" role="status"><strong>Hay borradores locales de la versión anterior.</strong> Su copia se conserva sin modificar y no se ha enviado al servidor. Este listado muestra únicamente los casos de tu cuenta guardados en el servidor.</div><div v-if="evaluationIssue && route.name !== 'evaluations'" class="demo-storage-alert" role="alert">{{ evaluationIssue }}</div><RouterView :key="route.fullPath" /><footer class="workspace-footer"><span>KneeOAI · Prototipo</span><span>Datos e imagen en servidor · predicción bloqueada</span></footer></main>
    </div>
  </div>
  <dialog ref="help" class="app-dialog" aria-labelledby="help-title" @click="closeOnBackdrop($event, help)">
    <div class="dialog-heading"><span class="dialog-symbol"><AppIcon name="help" :size="24" /></span><button class="icon-button" aria-label="Cerrar ayuda" @click="help?.close()"><AppIcon name="x" /></button></div>
    <h2 id="help-title">Alcance de esta fase</h2><p>La sesión y los datos basales se verifican y guardan en el servidor. Usa únicamente casos ficticios. Cada cuenta accede a sus propios casos.</p><p>La carga DICOM, la referencia visual sellada y la revisión se conservan en el servidor. El centro se registra antes de mostrar el recorte y no puede editarse después. El recorte automático no admite correcciones geométricas manuales.</p><div class="inline-note">La predicción permanece bloqueada. Este flujo no entrena modelos ni accede a cohortes o prueba reservada. No se estima un grado KL ni se indica un tratamiento.</div><button class="button button-primary dialog-close" @click="help?.close()">Entendido</button>
  </dialog>
</template>

<style scoped>
.demo-storage-alert { display: flex; align-items: flex-start; gap: 11px; max-width: 1240px; margin: 0 auto 24px; padding: 16px 18px; border: 1px solid #e4b6a4; border-radius: 9px; background: #fff3ed; color: #87462f; }
.demo-storage-alert > svg { flex: none; margin-top: 2px; }
.demo-storage-alert strong { font-size: 14px; font-weight: 650; line-height: 1.6; }
.demo-storage-alert p { margin-top: 5px; font-size: 13px; line-height: 1.7; }
.legacy-notice { max-width: 1240px; margin: 0 auto 20px; padding: 15px 18px; border: 1px solid #d5e2ea; border-radius: 9px; background: #f1f7fa; color: #395c70; font-size: 13px; line-height: 1.7; }
</style>
