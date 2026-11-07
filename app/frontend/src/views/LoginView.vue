<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import { DEMO_EMAIL, DEMO_PASSWORD, enterDemo } from '../demoSession'
import { closeOnBackdrop } from '../dialog'

const router = useRouter()
const email = ref('')
const password = ref('')
const showPassword = ref(false)
const errors = reactive({ email: '', password: '', credentials: '' })
const recovery = ref<HTMLDialogElement | null>(null)
const emailInput = ref<HTMLInputElement | null>(null)
const passwordInput = ref<HTMLInputElement | null>(null)
function fillDemo() {
  email.value = DEMO_EMAIL
  password.value = DEMO_PASSWORD
  errors.email = errors.password = errors.credentials = ''
  emailInput.value?.focus()
}
function clearFieldError(field: 'email' | 'password') {
  errors[field] = ''
  errors.credentials = ''
}
async function submit() {
  errors.email = !email.value.trim() ? 'Escribe tu correo de acceso.' : !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value.trim()) ? 'Revisa el formato del correo.' : ''
  errors.password = !password.value ? 'Escribe tu contraseña.' : ''
  errors.credentials = ''
  if (errors.email) { emailInput.value?.focus(); return }
  if (errors.password) { passwordInput.value?.focus(); return }
  if (email.value.trim().toLowerCase() !== DEMO_EMAIL || password.value !== DEMO_PASSWORD) {
    errors.credentials = 'Los datos no corresponden a la cuenta ficticia. Usa «Completar datos de ejemplo» para explorar el prototipo.'
    return
  }
  enterDemo()
  password.value = ''
  await router.replace('/evaluaciones')
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
        <p class="login-intro">Ingresa para consultar y retomar tus evaluaciones.</p>
        <form class="login-form" novalidate @submit.prevent="submit">
          <div class="field"><label for="login-email">Correo de acceso</label><div class="input-with-icon"><AppIcon name="mail" :size="19" /><input id="login-email" ref="emailInput" v-model="email" name="email" type="email" autocomplete="off" placeholder="medico@ejemplo.com" :aria-invalid="Boolean(errors.email)" :aria-describedby="errors.email ? 'email-error' : undefined" @input="clearFieldError('email')" /></div><p v-if="errors.email" id="email-error" class="field-error" role="alert">{{ errors.email }}</p></div>
          <div class="field"><label for="login-password">Contraseña</label><div class="input-with-icon password-field"><AppIcon name="lock" :size="19" /><input id="login-password" ref="passwordInput" v-model="password" name="password" :type="showPassword ? 'text' : 'password'" autocomplete="off" placeholder="Escribe tu contraseña" :aria-invalid="Boolean(errors.password)" :aria-describedby="errors.password ? 'password-error' : undefined" @input="clearFieldError('password')" /><button type="button" class="icon-button password-toggle" :aria-label="showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'" :aria-pressed="showPassword" @click="showPassword = !showPassword"><AppIcon :name="showPassword ? 'eye-off' : 'eye'" :size="19" /></button></div><p v-if="errors.password" id="password-error" class="field-error" role="alert">{{ errors.password }}</p></div>
          <div class="login-recovery"><button type="button" class="text-button" @click="recovery?.showModal()">¿Necesitas recuperar tu acceso?</button></div>
          <p v-if="errors.credentials" class="login-error" role="alert">{{ errors.credentials }}</p>
          <button type="submit" class="button button-primary login-submit">Entrar a la demostración<AppIcon name="arrow-right" :size="18" /></button>
        </form>
        <div class="demo-access"><div class="demo-access-heading"><span class="demo-access-label">PRUEBA LAS PANTALLAS</span><AppIcon name="scan" :size="18" /></div><p>Usa la cuenta ficticia para explorar el recorrido.</p><button type="button" class="text-button" @click="fillDemo">Completar datos de ejemplo<AppIcon name="arrow-up-right" :size="15" /></button></div>
        <p class="login-footnote">Acceso simulado. Usa únicamente los datos de ejemplo.</p>
      </div>
      <div class="login-version">KneeOAI · Prototipo v0.7</div>
    </section>
  </main>
  <dialog ref="recovery" class="app-dialog" aria-labelledby="recovery-title" @click="closeOnBackdrop($event, recovery)"><div class="dialog-heading"><span class="dialog-symbol"><AppIcon name="lock" :size="24" /></span><button class="icon-button" aria-label="Cerrar recuperación de acceso" @click="recovery?.close()"><AppIcon name="x" /></button></div><h2 id="recovery-title">Recuperar tu acceso</h2><p>En la primera versión, el administrador gestionará el restablecimiento de la contraseña.</p><div class="inline-note">En esta demostración puedes volver al formulario y pulsar «Completar datos de ejemplo». No se envían correos ni se modifica ninguna cuenta.</div><button class="button button-primary dialog-close" @click="recovery?.close()">Volver al acceso</button></dialog>
</template>
