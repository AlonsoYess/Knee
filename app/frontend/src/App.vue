<script setup lang="ts">
import { computed, ref } from 'vue'
import { isNavigationFailure, NavigationFailureType, useRoute, useRouter } from 'vue-router'
import AppIcon from './components/AppIcon.vue'
import { leaveDemo } from './demoSession'
import { closeOnBackdrop } from './dialog'
import { resetDemoListPreferences } from './demoListPreferences'
import { demoStorageIssue } from './demo/demoEvaluationPersistence'

const route = useRoute()
const router = useRouter()
const isLogin = computed(() => route.name === 'login')
const isEvaluationSection = computed(() => route.path === '/evaluaciones' || route.path.startsWith('/evaluaciones/'))
const help = ref<HTMLDialogElement | null>(null)
async function logout() {
  const failure = await router.push({ name: 'evaluations' })
  if (failure && !isNavigationFailure(failure, NavigationFailureType.duplicated)) return
  resetDemoListPreferences()
  leaveDemo()
  await router.replace('/acceso')
}
</script>

<template>
  <a class="skip-link" href="#main-content">Saltar al contenido</a>
  <div class="demo-ribbon"><span class="demo-ribbon-dot" aria-hidden="true"></span>Demostración académica<span class="demo-ribbon-detail">Datos ficticios · modelos sin conectar</span></div>
  <RouterView v-if="isLogin" />
  <div v-else class="workspace">
    <header class="workspace-header">
      <RouterLink to="/evaluaciones" class="brand" aria-label="KneeOAI, mis evaluaciones"><span class="brand-mark"><AppIcon name="activity" :size="23" /></span><span>Knee<span class="brand-light">OAI</span></span></RouterLink>
      <div class="header-account"><span class="account-avatar" aria-hidden="true">MD</span><div class="account-text"><strong>Médico de demostración</strong><span>Cuenta ficticia</span></div><button class="icon-button logout-button" aria-label="Cerrar sesión de demostración" @click="logout"><AppIcon name="logout" /></button></div>
    </header>
    <div class="workspace-body">
      <aside class="workspace-sidebar">
        <div><p class="nav-caption">ESPACIO DE TRABAJO</p><nav aria-label="Navegación principal"><RouterLink class="nav-item" :class="{ 'section-active': isEvaluationSection }" :aria-current="isEvaluationSection ? (route.name === 'evaluations' ? 'page' : 'location') : undefined" to="/evaluaciones"><AppIcon name="evaluations" />Mis evaluaciones</RouterLink><button class="nav-item" @click="help?.showModal()"><AppIcon name="help" />Ayuda y alcance</button></nav></div>
        <div class="sidebar-note"><AppIcon name="shield" :size="19" /><p>Prototipo académico<span>Progresión estructural a 48 meses</span></p></div>
      </aside>
      <main id="main-content" class="workspace-main"><div v-if="demoStorageIssue" id="demo-storage-alert" class="demo-storage-alert" role="alert"><AppIcon name="x" :size="20" /><div><strong>No se guardaron cambios nuevos en este navegador.</strong><p>{{ demoStorageIssue.message }}</p><p v-if="demoStorageIssue.kind === 'write' || demoStorageIssue.kind === 'conflict'">Si ya había una copia guardada, se conserva.</p></div></div><RouterView :key="route.fullPath" /><footer class="workspace-footer"><span>KneeOAI · Prototipo v0.7</span><span>Diseño de interfaz · información ficticia</span></footer></main>
    </div>
  </div>
  <dialog ref="help" class="app-dialog" aria-labelledby="help-title" @click="closeOnBackdrop($event, help)">
    <div class="dialog-heading"><span class="dialog-symbol"><AppIcon name="help" :size="24" /></span><button class="icon-button" aria-label="Cerrar ayuda" @click="help?.close()"><AppIcon name="x" /></button></div>
    <h2 id="help-title">Un espacio para evaluar cada caso</h2><p>Puedes preparar datos ficticios, confirmar el lado de una imagen sintética, revisar el recorte de ejemplo y consultar el resumen actual del caso.</p><p>El resumen reúne lo guardado y señala lo pendiente. La predicción permanece bloqueada: no hay modelos conectados. Lo que guardes se conserva en este navegador. Los cambios sin guardar pueden perderse. El acceso sigue siendo una sesión simulada.</p><div class="inline-note">El alcance académico se refiere a progresión estructural a 48 meses en la población estudiada con KL inicial 2–3. La revisión simulada no acredita calidad clínica. No se estima un grado KL ni se indica un tratamiento.</div><button class="button button-primary dialog-close" @click="help?.close()">Entendido</button>
  </dialog>
</template>

<style scoped>
.demo-storage-alert { display: flex; align-items: flex-start; gap: 11px; max-width: 1240px; margin: 0 auto 24px; padding: 16px 18px; border: 1px solid #e4b6a4; border-radius: 9px; background: #fff3ed; color: #87462f; }
.demo-storage-alert > svg { flex: none; margin-top: 2px; }
.demo-storage-alert strong { font-size: 14px; font-weight: 650; line-height: 1.6; }
.demo-storage-alert p { margin-top: 5px; font-size: 13px; line-height: 1.7; }
</style>
