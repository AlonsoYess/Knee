<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { isNavigationFailure, NavigationFailureType, useRoute, useRouter } from 'vue-router'
import { pwaController } from '../pwa/controller'
import { browserOnline, confirmServerConnection, pendingMutations, serverReachability, updateBrowserOnline } from '../state/connection'
import { apiMessage, refreshSession } from '../state/session'

const route = useRoute(), router = useRouter()
const productionPwa = import.meta.env.PROD
const pwa = ref(pwaController.getState())
const checkingConnection = ref(false), handingOff = ref(false), confirmationOpen = ref(false)
const connectionMessage = ref(''), actionMessage = ref('')
const consentDialog = ref<HTMLDialogElement | null>(null), progressDialog = ref<HTMLDialogElement | null>(null)
let unsubscribe: (() => void) | null = null
let consentAnswer: ((answer: boolean) => void) | null = null
let active = true
const edgeGuidance = computed(() => productionPwa && /Edg(?:A|iOS)?\//.test(navigator.userAgent) && !pwa.value.installed && !pwa.value.installAvailable)
const showStatus = computed(() => !browserOnline.value || serverReachability.value === 'unavailable' || Boolean(connectionMessage.value || actionMessage.value || (productionPwa && (pwa.value.error || pwa.value.installed || pwa.value.installAvailable || pwa.value.updateAvailable || edgeGuidance.value))))
watch(serverReachability, () => { connectionMessage.value = '' }, { flush: 'sync' })

function hasPageChanges(): boolean {
  // Existing basal/reference/QC guards are the single source of unsaved-edit state.
  const probe = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(probe)
  return probe.defaultPrevented
}
function hasUnsubmittedFields(): boolean {
  const selectors = route.name === 'login' ? '#login-email, #login-password' : route.name === 'crop-review' ? '#ruler-point-x, #ruler-point-y' : null
  if (!selectors) return false
  // Only test whether a field is occupied; never copy or persist its contents.
  return [...document.querySelectorAll<HTMLInputElement>(selectors)].some(field => !field.disabled && field.value.length > 0)
}
function answerConsent(answer: boolean) {
  const resolve = consentAnswer; consentAnswer = null; confirmationOpen.value = false; consentDialog.value?.close(); resolve?.(answer)
}
async function confirmUnsubmittedDiscard(): Promise<boolean> {
  confirmationOpen.value = true
  return new Promise(resolve => { consentAnswer = resolve; consentDialog.value?.showModal() })
}
async function retryConnection() {
  if (checkingConnection.value || pendingMutations.value) return
  checkingConnection.value = true; connectionMessage.value = ''
  try {
    const session = await refreshSession()
    if (!active) return
    confirmServerConnection()
    connectionMessage.value = session.authenticated ? 'El servidor está disponible. Tu sesión sigue verificada.' : 'El servidor está disponible. Inicia sesión para continuar.'
    if (session.authenticated && route.name === 'login') await router.replace('/evaluaciones')
  } catch (error) { if (active) connectionMessage.value = apiMessage(error) }
  finally { if (active) checkingConnection.value = false }
}
async function install() {
  if (!productionPwa) return
  actionMessage.value = ''
  const accepted = await pwaController.requestInstall()
  if (active && accepted && !pwaController.getState().installed) actionMessage.value = 'El navegador está completando la instalación.'
}
async function updateApplication() {
  if (!productionPwa || handingOff.value || pwa.value.updating || pendingMutations.value > 0) return
  handingOff.value = true; actionMessage.value = ''
  let approvedDraftDiscard = false
  try {
    const guardedChanges = hasPageChanges()
    if (!guardedChanges && hasUnsubmittedFields()) {
      approvedDraftDiscard = await confirmUnsubmittedDiscard()
      if (!approvedDraftDiscard) return
    }
    if (guardedChanges || (approvedDraftDiscard && route.name !== 'login')) {
      // Navigation invokes the current view's own discard dialog. A cancellation
      // keeps both the view and the waiting worker untouched.
      const failure = await router.push({ name: 'evaluations' })
      if (failure && (!isNavigationFailure(failure, NavigationFailureType.duplicated))) {
        actionMessage.value = 'Actualización aplazada. Tus cambios siguen en la pantalla.'; return
      }
    }
    const safeToActivate = () => pendingMutations.value === 0 && !hasPageChanges() && (!hasUnsubmittedFields() || approvedDraftDiscard)
    if (!safeToActivate()) { actionMessage.value = 'Termina la solicitud o revisa los cambios pendientes antes de actualizar.'; return }
    // Keep this tab still during the activation handoff; another tab is never reloaded.
    progressDialog.value?.showModal()
    const activated = await pwaController.activateUpdate(safeToActivate)
    if (active && !activated && !pwaController.getState().error) actionMessage.value = 'La actualización sigue pendiente. Puedes continuar y volver a intentarlo cuando hayas terminado.'
  } catch {
    if (active) actionMessage.value = 'No se pudo completar la actualización. Puedes seguir usando esta versión y volver a intentarlo después.'
  } finally {
    progressDialog.value?.close()
    if (active) handingOff.value = false
  }
}
onMounted(() => {
  updateBrowserOnline()
  window.addEventListener('online', updateBrowserOnline); window.addEventListener('offline', updateBrowserOnline)
  unsubscribe = pwaController.subscribe(state => { if (active) { pwa.value = { ...state }; if (state.installed) actionMessage.value = '' } })
  if (productionPwa) void pwaController.initialize().catch(() => { if (active) actionMessage.value = 'La instalación de la aplicación no está disponible por ahora. Puedes seguir usando el navegador.' })
})
onBeforeUnmount(() => { active = false; unsubscribe?.(); window.removeEventListener('online', updateBrowserOnline); window.removeEventListener('offline', updateBrowserOnline); consentAnswer?.(false); consentAnswer = null })
</script>

<template>
  <section v-if="showStatus" class="application-status" aria-label="Estado de la aplicación">
    <div v-if="serverReachability === 'unavailable'" id="server-connection-notice" class="application-notice application-notice-error" role="alert"><div><strong>No pudimos conectar con el servidor.</strong><p>Un cambio solo queda guardado cuando el servidor lo confirma. La aplicación no guarda casos nuevos sin conexión.</p></div><button id="retry-server-connection" type="button" class="button button-secondary" :disabled="checkingConnection || pendingMutations > 0" @click="retryConnection">{{ checkingConnection ? 'Comprobando servidor…' : 'Reintentar conexión' }}</button></div>
    <div v-if="!browserOnline" id="internet-connection-notice" class="application-notice" role="status"><div><strong>El navegador indica que no hay conexión a Internet.</strong><p>El servidor local puede seguir funcionando. Puedes usarlo si está disponible; cada guardado requiere su confirmación.</p></div><button v-if="serverReachability !== 'unavailable'" type="button" class="button button-secondary" :disabled="checkingConnection || pendingMutations > 0" @click="retryConnection">Comprobar servidor local</button></div>
    <p v-if="connectionMessage" id="server-connection-result" class="application-message" role="status">{{ connectionMessage }}</p>
    <div v-if="productionPwa && (pwa.installed || pwa.installAvailable || edgeGuidance)" class="application-tools"><span v-if="pwa.installed" id="pwa-installed" class="application-installed" role="status">Aplicación instalada</span><template v-else-if="pwa.installAvailable"><span>Puedes abrir KneeOAI en su propia ventana.</span><button id="pwa-install" type="button" class="button button-secondary" :disabled="pwa.installing" @click="install">{{ pwa.installing ? 'Instalando…' : 'Instalar aplicación' }}</button></template><details v-else-if="edgeGuidance" class="application-install-help"><summary>Cómo instalar en Edge</summary><p>En Edge, abre el menú de tres puntos → Aplicaciones. Si aparece «Instalar este sitio como una aplicación», selecciónalo. También puede aparecer el icono de instalación en la barra de direcciones.</p><p>Instalar la aplicación no sustituye al servidor local: debe estar iniciado para consultar o guardar tus casos.</p></details></div>
    <div v-if="productionPwa && pwa.updateAvailable" id="pwa-update-notice" class="application-notice" role="status"><div><strong>Hay una versión nueva lista.</strong><p>Guarda o descarta los cambios pendientes antes de actualizar. Solo se recargará esta pestaña o ventana.</p><p v-if="pendingMutations > 0">Hay una carga o un guardado en curso. Espera a que termine antes de actualizar.</p></div><button id="pwa-update" type="button" class="button button-primary" :disabled="pendingMutations > 0 || handingOff || pwa.updating" @click="updateApplication">{{ handingOff || pwa.updating ? 'Preparando actualización…' : 'Actualizar y recargar' }}</button></div>
    <p v-if="productionPwa && pwa.error" id="pwa-error" class="application-message application-message-error" role="alert">{{ pwa.error }}</p>
    <p v-if="actionMessage" id="pwa-message" class="application-message" role="status">{{ actionMessage }}</p>
  </section>
  <dialog ref="consentDialog" id="pwa-update-confirm" class="app-dialog" aria-labelledby="update-discard-title" @cancel.prevent="answerConsent(false)" @close="confirmationOpen && answerConsent(false)"><h2 id="update-discard-title">¿Actualizar y dejar este formulario?</h2><p>La aplicación se recargará. Los datos que escribiste aquí y todavía no enviaste no se conservarán.</p><div class="workflow-actions"><button type="button" class="button button-secondary" autofocus @click="answerConsent(false)">Seguir aquí</button><button type="button" class="button button-primary" @click="answerConsent(true)">Descartar y actualizar</button></div></dialog>
  <dialog ref="progressDialog" id="pwa-update-progress" class="app-dialog" aria-labelledby="update-progress-title" @cancel.prevent><h2 id="update-progress-title">Preparando la actualización</h2><p role="status">Estamos comprobando que sea seguro recargar esta ventana. No cierres la aplicación mientras termina.</p></dialog>
</template>

<style scoped>
.application-status { background: #f4f8fa; border-bottom: 1px solid #dce6eb; padding: 12px clamp(16px, 3vw, 40px); color: #2d5163; font-size: 13px; }
.application-notice { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 13px 16px; border: 1px solid #d4e3eb; border-radius: 9px; background: #edf6fa; margin: 7px 0; }
.application-notice p { line-height: 1.7; margin-top: 5px; }
.application-notice .button { flex: none; }
.application-notice-error { background: #fff3ed; border-color: #e7c7b8; color: #864c39; }
.application-tools { display: flex; align-items: center; gap: 15px; flex-wrap: wrap; margin: 5px 0; }
.application-installed { display: inline-block; color: #2b665b; background: #e6f3ee; padding: 8px 11px; border-radius: 7px; }
.application-message { margin: 8px 0; line-height: 1.7; }
.application-message-error { color: #8a453b; }
.application-install-help { max-width: 760px; line-height: 1.7; }
.application-install-help summary { cursor: pointer; font-weight: 600; }
.application-install-help p { margin-top: 9px; }
@media(max-width: 650px) { .application-notice { align-items: stretch; flex-direction: column; gap: 11px; } .application-notice .button { width: 100%; } }
</style>
