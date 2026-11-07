<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import { apiMessage, authenticate, refreshSession, sessionIssue } from '../state/session'
import { closeOnBackdrop } from '../dialog'

const router = useRouter()
const email = ref('')
const password = ref('')
const showPassword = ref(false)
const errors = reactive({ email: '', password: '', credentials: '' })
const recovery = ref<HTMLDialogElement | null>(null)
const emailInput = ref<HTMLInputElement | null>(null)
const passwordInput = ref<HTMLInputElement | null>(null)
const submitting = ref(false)
async function retrySession() {
  try { if ((await refreshSession()).authenticated) await router.replace('/evaluaciones') }
  catch { /* Session error is visible below. */ }
}
function clearFieldError(field: 'email' | 'password') {
  errors[field] = ''
  errors.credentials = ''
}
async function submit() {
  if (submitting.value) return
  errors.email = !email.value.trim() ? 'Escribe tu correo de acceso.' : !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value.trim()) ? 'Revisa el formato del correo.' : ''
  errors.password = !password.value ? 'Escribe tu contraseña.' : ''
  errors.credentials = ''
  if (errors.email) { emailInput.value?.focus(); return }
  if (errors.password) { passwordInput.value?.focus(); return }
  submitting.value = true
  try {
    await authenticate(email.value.trim(), password.value)
    password.value = ''
    await router.replace('/evaluaciones')
  } catch (error) { errors.credentials = apiMessage(error) }
  finally { submitting.value = false; password.value = '' }
}
</script>

<template>
  <main id="main-content" class="login-layout">
    <section class="login-story" aria-label="Presentación del prototipo">
      <a class="brand login-brand" href="/acceso" aria-label="KneeOAI"><span class="brand-mark"><AppIcon name="activity" :size="24" /></span><span>Knee<span class="brand-light">OAI</span></span></a>
      <div class="story-content"><span class="story-eyebrow">UN CASO. UN RECORRIDO CLARO.</span><h2>La evaluación comienza<br />con una buena revisión.</h2><p>Un espacio organizado para trabajar con la imagen inicial y los datos de cada rodilla.</p><div class="story-route" aria-label="Recorrido previsto"><div><span>01</span><p>Prepara el caso<small>Imagen y datos basales</small></p></div><div><span>02</span><p>Revisa la imagen<small>Rodilla y región seleccionada</small></p></div><div><span>03</span><p>Consulta la evaluación<small>Datos y revisión reunidos</small></p></div></div></div>
      <div class="story-footer"><AppIcon name="shield" :size="19" /><span>Investigación académica · KneeOAI</span></div>
    </section>
    <section class="login-content" aria-label="Acceso al prototipo">
      <div class="login-box">
        <span class="section-eyebrow">TU ESPACIO DE EVALUACIÓN</span>
        <h1 tabindex="-1">Te damos la bienvenida</h1>
        <p class="login-intro">Ingresa con tu cuenta administrada para consultar los casos ficticios guardados en el servidor.</p>
        <div v-if="sessionIssue" id="api-session-error" class="login-error" role="alert"><p>{{ sessionIssue }}</p><button type="button" class="text-button" :disabled="submitting" @click="retrySession">Reintentar conexión</button></div>
        <form class="login-form" novalidate @submit.prevent="submit">
          <div class="field"><label for="login-email">Correo de acceso</label><div class="input-with-icon"><AppIcon name="mail" :size="19" /><input id="login-email" ref="emailInput" v-model="email" name="email" type="email" autocomplete="username" placeholder="Tu correo de acceso" :disabled="submitting" :aria-invalid="Boolean(errors.email)" :aria-describedby="errors.email ? 'email-error' : undefined" @input="clearFieldError('email')" /></div><p v-if="errors.email" id="email-error" class="field-error" role="alert">{{ errors.email }}</p></div>
          <div class="field"><label for="login-password">Contraseña</label><div class="input-with-icon password-field"><AppIcon name="lock" :size="19" /><input id="login-password" ref="passwordInput" v-model="password" name="password" :type="showPassword ? 'text' : 'password'" autocomplete="current-password" :disabled="submitting" placeholder="Escribe tu contraseña" :aria-invalid="Boolean(errors.password)" :aria-describedby="errors.password ? 'password-error' : undefined" @input="clearFieldError('password')" /><button type="button" class="icon-button password-toggle" :aria-label="showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'" :aria-pressed="showPassword" @click="showPassword = !showPassword"><AppIcon :name="showPassword ? 'eye-off' : 'eye'" :size="19" /></button></div><p v-if="errors.password" id="password-error" class="field-error" role="alert">{{ errors.password }}</p></div>
          <div class="login-recovery"><button type="button" class="text-button" @click="recovery?.showModal()">¿Necesitas recuperar tu acceso?</button></div>
          <p v-if="errors.credentials" class="login-error" role="alert">{{ errors.credentials }}</p>
          <button type="submit" class="button button-primary login-submit" :disabled="submitting">{{ submitting ? 'Iniciando sesión…' : 'Iniciar sesión' }}<AppIcon name="arrow-right" :size="18" /></button>
        </form>
        <div class="demo-access"><div class="demo-access-heading"><span class="demo-access-label">ACCESO ADMINISTRADO</span><AppIcon name="shield" :size="18" /></div><p>El administrador crea tu cuenta. El prototipo no incluye una contraseña pública ni permite registrar cuentas desde esta pantalla.</p></div>
        <p class="login-footnote">Sesión real · solo casos ficticios · modelos sin conectar.</p>
      </div>
      <div class="login-version">KneeOAI · Prototipo académico</div>
    </section>
  </main>
  <dialog ref="recovery" class="app-dialog" aria-labelledby="recovery-title" @click="closeOnBackdrop($event, recovery)"><div class="dialog-heading"><span class="dialog-symbol"><AppIcon name="lock" :size="24" /></span><button class="icon-button" aria-label="Cerrar recuperación de acceso" @click="recovery?.close()"><AppIcon name="x" /></button></div><h2 id="recovery-title">Recuperar tu acceso</h2><p>Comunícate con el administrador para restablecer el acceso a tu cuenta.</p><div class="inline-note">Esta pantalla no envía correos ni cambia contraseñas. No compartas tu contraseña en el chat.</div><button class="button button-primary dialog-close" @click="recovery?.close()">Volver al acceso</button></dialog>
</template>
