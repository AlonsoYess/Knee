<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { onBeforeRouteLeave, useRoute, useRouter } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import DemoCropPreview from '../components/DemoCropPreview.vue'
import { closeOnBackdrop } from '../dialog'
import {
  evaluations, createDraftInput, validateDemoDraft, createDemoImageInput,
  validateDemoImage, isEditableDemoEvaluation, createDemoReviewInput,
  saveDemoReview, type DemoReviewInput,
} from '../demo/demoEvaluationStore'
import { demoStorageIssue } from '../demo/demoEvaluationPersistence'
import { getDemoCrop, demoRejectReasons } from '../demo/demoCrops'

const route = useRoute()
const router = useRouter()
const requestedId = typeof route.params.id === 'string' ? route.params.id : ''
const existing = computed(() => evaluations.value.find(item => item.id === requestedId))
const unavailable = computed(() => !isEditableDemoEvaluation(existing.value))
const baselineIncomplete = computed(() => Boolean(existing.value && Object.keys(validateDemoDraft(createDraftInput(existing.value), true)).length))
const crop = computed(() => getDemoCrop(existing.value?.image?.sampleId ?? '', existing.value?.knee ?? null))
const imageIncomplete = computed(() => !existing.value || !crop.value || Object.keys(validateDemoImage(createDemoImageInput(existing.value), existing.value, true)).length > 0)
const form = reactive<DemoReviewInput>(existing.value ? createDemoReviewInput(existing.value) : { decision: 'pending', reason: '', reviewed: false })
const baseline = ref(JSON.stringify(form))
const decisionLocked = ref(form.decision !== 'pending')
const originalLoaded = ref(false)
const cropLoaded = ref(false)
const previewFailed = ref(false)
const errors = reactive({ reviewed: '', reason: '', decision: '' })
const saving = ref(false)
const saveError = ref('')
const message = ref('')
const enlarged = ref<HTMLDialogElement | null>(null)
const enlargedView = ref<'original' | 'crop'>('original')
const confirmation = ref<HTMLDialogElement | null>(null)
const confirmationMode = ref<'leave' | 'reopen'>('leave')
let resolveConfirmation: ((value: boolean) => void) | null = null

const steps = ['Datos basales', 'Radiografía y lado', 'Revisión del recorte', 'Resumen']
const ready = computed(() => !unavailable.value && !baselineIncomplete.value && !imageIncomplete.value)
const imagesVisible = computed(() => originalLoaded.value && cropLoaded.value && !previewFailed.value)
const dirty = computed(() => ready.value && JSON.stringify(form) !== baseline.value)
const sideLabel = computed(() => existing.value?.knee === 'right' ? 'Derecha del paciente' : existing.value?.knee === 'left' ? 'Izquierda del paciente' : 'Pendiente')
const decisionLabel = computed(() => form.decision === 'accepted' ? 'Recorte aceptado · demostración' : form.decision === 'rejected' ? 'Recorte rechazado · demostración' : 'Revisión pendiente')
const reasonLabel = computed(() => demoRejectReasons.find(reason => reason.id === form.reason)?.label ?? 'Sin motivo seleccionado')
const imageRoute = computed(() => ({ name: 'radiograph', params: { id: requestedId } }))
const editRoute = computed(() => ({ name: 'edit-evaluation', params: { id: requestedId } }))

function clearErrors() { errors.reviewed = ''; errors.reason = ''; errors.decision = ''; saveError.value = '' }
function sourceFailed() {
  previewFailed.value = true
  form.reviewed = false
  form.decision = 'pending'
  form.reason = ''
  decisionLocked.value = false
  clearErrors()
  message.value = ''
}
function reviewedChanged() { errors.reviewed = ''; errors.decision = ''; message.value = '' }
async function acceptExample() {
  clearErrors()
  if (!imagesVisible.value) { errors.reviewed = 'Espera a que se muestren correctamente la imagen y el recorte.'; return }
  if (!form.reviewed) {
    errors.reviewed = 'Confirma que has revisado la imagen y el recorte de este ejemplo.'
    await nextTick(); document.getElementById('crop-reviewed')?.focus(); return
  }
  form.decision = 'accepted'
  form.reason = ''
  decisionLocked.value = true
  message.value = 'Aceptación del ejemplo seleccionada. Guarda la revisión para conservarla.'
  await nextTick(); document.getElementById('crop-save-summary')?.focus()
}
async function rejectExample() {
  clearErrors()
  if (!form.reason || !demoRejectReasons.some(reason => reason.id === form.reason)) {
    errors.reason = 'Selecciona un motivo para rechazar el recorte de ejemplo.'
    await nextTick(); document.getElementById('crop-reason')?.focus(); return
  }
  form.decision = 'rejected'
  decisionLocked.value = true
  message.value = 'Rechazo del ejemplo seleccionado. Guarda la revisión para conservarlo.'
  await nextTick(); document.getElementById('crop-save-exit')?.focus()
}
function ask(mode: 'leave' | 'reopen'): Promise<boolean> {
  if (resolveConfirmation) return Promise.resolve(false)
  confirmationMode.value = mode
  return new Promise(resolve => { resolveConfirmation = resolve; confirmation.value?.showModal() })
}
function answer(accepted: boolean) {
  const resolve = resolveConfirmation
  resolveConfirmation = null
  confirmation.value?.close()
  resolve?.(accepted)
}
function confirmationClosed() {
  const resolve = resolveConfirmation
  resolveConfirmation = null
  resolve?.(false)
}
async function reopenReview() {
  if (!await ask('reopen')) return
  Object.assign(form, { decision: 'pending', reason: '', reviewed: false })
  decisionLocked.value = false
  clearErrors()
  message.value = 'Revisión reabierta en esta pantalla. La decisión guardada solo cambiará cuando vuelvas a guardarla.'
  await nextTick(); document.getElementById('crop-reviewed')?.focus()
}
function saveCurrent(reachSummary = false): boolean {
  try {
    saveDemoReview(form, requestedId, { reachSummary })
    baseline.value = JSON.stringify(form)
    decisionLocked.value = form.decision !== 'pending'
    return true
  } catch {
    saveError.value = demoStorageIssue.value?.message ?? 'No pudimos conservar la revisión del ejemplo. Vuelve a comprobar la imagen, el lado y la decisión.'
    return false
  }
}
async function saveReview() {
  if (!ready.value || saving.value) return
  clearErrors()
  saving.value = true
  try { if (saveCurrent()) await router.push('/evaluaciones') } finally { saving.value = false }
}
async function continueToSummary() {
  if (!ready.value || saving.value) return
  clearErrors()
  if (form.decision !== 'accepted') {
    errors.decision = form.decision === 'rejected' ? 'El recorte rechazado no permite continuar. Vuelve a la imagen o reabre la revisión del ejemplo.' : 'Selecciona una aceptación explícita del recorte antes de continuar.'
    await nextTick(); document.getElementById(form.decision === 'pending' ? 'crop-reviewed' : 'crop-reopen')?.focus(); return
  }
  if (!imagesVisible.value || !form.reviewed) { errors.reviewed = 'La imagen y el recorte deben mostrarse y estar revisados para continuar.'; return }
  saving.value = true
  try {
    if (saveCurrent(true)) {
      await router.push({ name: 'case-summary', params: { id: requestedId } })
    }
  } finally { saving.value = false }
}
function enlarge(view: 'original' | 'crop') { enlargedView.value = view; enlarged.value?.showModal() }
function beforeUnload(event: BeforeUnloadEvent) { if (dirty.value) { event.preventDefault(); event.returnValue = '' } }
onBeforeRouteLeave(async () => !dirty.value || await ask('leave'))
onMounted(() => window.addEventListener('beforeunload', beforeUnload))
onBeforeUnmount(() => { window.removeEventListener('beforeunload', beforeUnload); resolveConfirmation?.(false); resolveConfirmation = null })
</script>

<template>
  <div class="crop-page">
    <RouterLink to="/evaluaciones" class="crop-back"><AppIcon name="chevron-left" :size="16" />Volver a Mis evaluaciones</RouterLink>
    <section v-if="unavailable" class="crop-unavailable"><span class="crop-symbol"><AppIcon name="image" :size="25" /></span><h1 tabindex="-1">Revisión no disponible</h1><p>Este caso no existe en la demostración o no tiene un borrador editable asociado.</p><RouterLink to="/evaluaciones" class="button button-primary">Volver al listado</RouterLink></section>
    <section v-else-if="baselineIncomplete" class="crop-unavailable"><span class="crop-symbol"><AppIcon name="file" :size="25" /></span><h1 tabindex="-1">Datos basales pendientes</h1><p>Completa los datos ficticios antes de revisar la imagen.</p><RouterLink :to="editRoute" class="button button-primary">Completar datos basales</RouterLink></section>
    <section v-else-if="imageIncomplete" class="crop-unavailable"><span class="crop-symbol"><AppIcon name="image" :size="25" /></span><h1 tabindex="-1">Imagen y lado pendientes</h1><p>Selecciona una imagen de demostración y confirma el lado antes de revisar el recorte.</p><RouterLink :to="imageRoute" class="button button-primary">Volver a Radiografía y lado</RouterLink></section>
    <template v-else>
      <header class="crop-heading"><div><span class="section-eyebrow">REVISAR EL EJEMPLO</span><h1 tabindex="-1">Revisión del recorte</h1><p>Compara ambas vistas y registra una decisión de demostración.</p></div><span class="crop-code"><AppIcon name="file" :size="15" />{{ existing?.code }}</span></header>
      <ol class="crop-stepper" aria-label="Recorrido de evaluación"><li v-for="(step,index) in steps" :key="step" :class="{ 'crop-step-current': index === 2, 'crop-step-done': index < 2 }" :aria-current="index === 2 ? 'step' : undefined"><span class="crop-step-number"><AppIcon v-if="index < 2" name="check" :size="14" /><template v-else>{{ index + 1 }}</template></span><span>{{ step }}</span></li></ol>

      <div class="crop-scope-note"><AppIcon name="shield" :size="19" /><p><strong>Revisión de demostración.</strong> Las imágenes y los recortes son sintéticos. Esta decisión prueba el recorrido de la interfaz.</p></div>
      <div class="crop-layout">
        <section class="crop-comparison" aria-labelledby="crop-comparison-title">
          <div class="crop-panel-heading"><div><h2 id="crop-comparison-title">Imagen y región seleccionada</h2><p>{{ sideLabel }} · {{ crop?.label }}</p></div><span class="crop-fixture-version">{{ crop?.version }}</span></div>
          <div v-if="crop" class="crop-views">
            <figure><figcaption><span>Imagen original</span><button type="button" class="text-button" :disabled="!originalLoaded || previewFailed" @click="enlarge('original')"><AppIcon name="scan" :size="15" />Ampliar original</button></figcaption><div class="crop-preview-frame"><DemoCropPreview :fixture="crop" view="original" @load="originalLoaded = true" @error="sourceFailed" /></div><p>El marco señala la región fija de este ejemplo.</p></figure>
            <figure><figcaption><span>Recorte del ejemplo</span><button type="button" class="text-button" :disabled="!cropLoaded || previewFailed" @click="enlarge('crop')"><AppIcon name="scan" :size="15" />Ampliar recorte</button></figcaption><div class="crop-preview-frame"><DemoCropPreview :fixture="crop" view="crop" @load="cropLoaded = true" @error="sourceFailed" /></div><p>Vista del mismo lado y de la misma imagen.</p></figure>
          </div>
          <p v-if="previewFailed" class="crop-load-error" role="alert">No pudimos mostrar correctamente las dos vistas. La aceptación queda bloqueada; vuelve a la imagen para seleccionar otro ejemplo.</p>
          <p v-else-if="!imagesVisible" class="crop-loading" role="status">Preparando las vistas del ejemplo…</p>
          <div class="crop-identity"><AppIcon name="lock" :size="15" /><p>La decisión corresponde a este ejemplo, este lado y esta versión del recorte. Cambiar la imagen o el lado exige una nueva revisión.</p></div>
        </section>

        <aside class="crop-decision-panel" aria-labelledby="crop-decision-title">
          <div class="crop-decision-heading"><span class="crop-symbol"><AppIcon name="check" /></span><h2 id="crop-decision-title">Decisión del ejemplo</h2></div>
          <div :class="['crop-status', `crop-status-${form.decision}`]" role="status" aria-live="polite"><AppIcon :name="form.decision === 'accepted' ? 'check' : form.decision === 'rejected' ? 'x' : 'clock'" :size="17" /><span>{{ decisionLabel }}</span></div>
          <template v-if="decisionLocked">
            <p class="crop-record-note">{{ dirty ? 'Decisión seleccionada, pendiente de guardar.' : 'Esta decisión está conservada en el borrador de demostración.' }}</p>
            <div v-if="form.decision === 'rejected'" class="crop-rejected-reason"><span>Motivo del ejemplo</span><strong>{{ reasonLabel }}</strong></div>
            <button id="crop-reopen" type="button" class="button button-secondary crop-reopen" @click="reopenReview">Reabrir revisión</button>
          </template>
          <template v-else>
            <label class="crop-reviewed-label"><input id="crop-reviewed" v-model="form.reviewed" type="checkbox" :disabled="!imagesVisible" :aria-invalid="Boolean(errors.reviewed)" :aria-describedby="errors.reviewed ? 'crop-reviewed-error' : 'crop-reviewed-hint'" @change="reviewedChanged" /><span>He revisado la imagen y el recorte de este ejemplo</span></label>
            <p id="crop-reviewed-hint" class="crop-hint">Revisa ambas vistas y el lado indicado antes de seleccionar la aceptación.</p>
            <p v-if="errors.reviewed" id="crop-reviewed-error" class="field-error" role="alert">{{ errors.reviewed }}</p>
            <button type="button" class="button button-primary crop-accept" :disabled="!imagesVisible || saving" @click="acceptExample"><AppIcon name="check" :size="16" />Aceptar recorte de ejemplo</button>
            <div class="crop-reject-section"><label for="crop-reason">Si el recorte no es utilizable</label><select id="crop-reason" v-model="form.reason" :aria-invalid="Boolean(errors.reason)" :aria-describedby="errors.reason ? 'crop-reason-error' : 'crop-reason-hint'" @change="errors.reason = ''; message = ''"><option value="">Seleccionar motivo</option><option v-for="reason in demoRejectReasons" :key="reason.id" :value="reason.id">{{ reason.label }}</option></select><p id="crop-reason-hint" class="crop-hint">Opciones de ejemplo para diseñar el registro del rechazo.</p><p v-if="errors.reason" id="crop-reason-error" class="field-error" role="alert">{{ errors.reason }}</p><button type="button" class="button button-secondary crop-reject" :disabled="saving" @click="rejectExample"><AppIcon name="x" :size="16" />Rechazar recorte de ejemplo</button></div>
          </template>
          <div class="crop-session-note"><AppIcon name="clock" :size="17" /><p>Lo que guardes se conserva en este navegador. Los cambios sin guardar pueden perderse.</p></div>
        </aside>
      </div>
      <p v-if="errors.decision" class="crop-action-error" role="alert">{{ errors.decision }}</p>
      <p v-if="saveError" class="crop-action-error" role="alert">{{ saveError }}</p>
      <div class="crop-actions"><RouterLink :to="imageRoute" class="button button-secondary"><AppIcon name="chevron-left" :size="16" />Volver a Radiografía y lado</RouterLink><div class="crop-primary-actions"><button id="crop-save-exit" type="button" class="button button-secondary" :disabled="saving" @click="saveReview">Guardar y salir</button><button id="crop-save-summary" type="button" class="button button-primary" :disabled="saving || form.decision === 'rejected' || (form.decision === 'accepted' && !imagesVisible)" @click="continueToSummary">Guardar y ver resumen<AppIcon name="arrow-right" :size="17" /></button></div></div>
      <p class="crop-feedback" role="status" aria-live="polite">{{ message }}</p><p class="crop-dirty-note">{{ dirty ? 'Tienes cambios sin guardar.' : 'La revisión guardada permanece asociada al ejemplo en este navegador.' }}</p>
    </template>

    <dialog ref="enlarged" class="app-dialog crop-enlarged" aria-labelledby="crop-enlarged-title" @click="closeOnBackdrop($event,enlarged)"><div class="dialog-heading"><h2 id="crop-enlarged-title">{{ enlargedView === 'original' ? 'Imagen original ampliada' : 'Recorte ampliado' }}</h2><button type="button" class="icon-button" aria-label="Cerrar vista ampliada" @click="enlarged?.close()"><AppIcon name="x" /></button></div><DemoCropPreview v-if="crop" :fixture="crop" :view="enlargedView" /><p>{{ sideLabel }} · Ilustración sintética</p><button type="button" class="button button-primary" @click="enlarged?.close()">Volver a la revisión</button></dialog>
    <dialog ref="confirmation" class="app-dialog" aria-labelledby="crop-confirmation-title" @cancel.prevent="answer(false)" @close="confirmationClosed"><div class="dialog-heading"><span class="dialog-symbol"><AppIcon name="file" :size="24" /></span><button type="button" class="icon-button" aria-label="Cancelar y seguir editando" autofocus @click="answer(false)"><AppIcon name="x" /></button></div><h2 id="crop-confirmation-title">{{ confirmationMode === 'leave' ? 'Tienes cambios sin guardar' : '¿Reabrir la revisión?' }}</h2><p>{{ confirmationMode === 'leave' ? 'Puedes seguir editando o salir descartando los cambios de esta pantalla.' : 'La revisión de esta pantalla volverá a pendiente. La decisión conservada en el borrador solo cambiará cuando vuelvas a guardarla.' }}</p><div class="crop-dialog-actions"><button type="button" class="button button-secondary" @click="answer(false)">{{ confirmationMode === 'leave' ? 'Seguir editando' : 'Conservar decisión' }}</button><button type="button" class="button button-primary" @click="answer(true)">{{ confirmationMode === 'leave' ? 'Salir sin guardar' : 'Reabrir revisión' }}</button></div></dialog>
  </div>
</template>

<style scoped>
.crop-page { width: 100%; max-width: 1240px; margin: 0 auto; }
.crop-back { display: inline-flex; align-items: center; gap: 6px; min-height: 32px; margin-bottom: 25px; color: var(--muted); text-decoration: none; font-size: 12px; }
.crop-back:hover { color: var(--brand); text-decoration: underline; text-underline-offset: 3px; }
.crop-heading { display: flex; justify-content: space-between; align-items: center; gap: 20px; margin-bottom: 26px; }
.crop-heading h1 { margin-top: 8px; font-size: clamp(26px,3vw,32px); font-weight: 650; line-height: 1.2; letter-spacing: -.035em; }
.crop-heading p { font-size: 14px; color: var(--muted); margin-top: 10px; }
.crop-code { display: inline-flex; align-items: center; gap: 7px; flex: none; padding: 7px 10px; border: 1px solid var(--border); border-radius: 7px; background: white; color: #4d6878; font-size: 12px; font-weight: 600; }
.crop-stepper { list-style: none; padding: 0; margin: 0 0 25px; display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 10px; }
.crop-stepper li { display: flex; align-items: center; gap: 8px; padding-bottom: 14px; border-bottom: 2px solid #dce6eb; color: #657c8b; font-size: 12px; }
.crop-step-number { display: grid; place-items: center; flex: none; width: 26px; height: 26px; border-radius: 50%; background: #e7edf1; color: #607684; font-size: 11px; }
.crop-stepper .crop-step-current { border-bottom-color: var(--brand); color: var(--brand-dark); font-weight: 600; }
.crop-step-current .crop-step-number { background: var(--brand); color: white; }
.crop-step-done .crop-step-number { background: #e0efec; color: var(--brand); }
.crop-scope-note { display: flex; align-items: flex-start; gap: 10px; padding: 13px 16px; background: #edf6f5; border: 1px solid #d9e9e6; border-radius: 9px; margin-bottom: 23px; color: #486c6c; }
.crop-scope-note > svg { flex: none; margin-top: 1px; color: var(--brand); }
.crop-scope-note p { font-size: 12px; line-height: 1.7; }
.crop-scope-note strong { color: #31595b; font-weight: 600; }
.crop-layout { display: grid; grid-template-columns: minmax(0,1fr) 310px; gap: 24px; align-items: start; }
.crop-comparison,.crop-decision-panel { min-width: 0; background: white; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.crop-panel-heading { padding: 20px 22px; border-bottom: 1px solid #e8eef1; display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap; }
.crop-panel-heading h2,.crop-decision-heading h2 { font-size: 16px; font-weight: 650; }
.crop-panel-heading p { font-size: 12px; color: var(--muted); margin-top: 4px; }
.crop-fixture-version { background: #f1f5f7; color: #567181; padding: 4px 7px; border-radius: 5px; font-size: 10px; white-space: nowrap; }
.crop-views { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 18px; padding: 21px 22px 12px; }
.crop-views figure { min-width: 0; margin: 0; }
.crop-views figcaption { display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; font-size: 12px; font-weight: 600; }
.crop-views .text-button { min-height: 32px; font-size: 12px; font-weight: 500; }
.crop-preview-frame { display: grid; align-items: center; background: #13222e; border-radius: 8px; overflow: hidden; min-height: 230px; padding: 8px; }
.crop-preview-frame :deep(svg) { display: block; width: 100%; height: auto; max-height: 330px; }
.crop-views figure > p { font-size: 11px; line-height: 1.65; color: var(--muted); margin-top: 10px; }
.crop-identity { display: flex; align-items: flex-start; gap: 8px; margin: 6px 22px 21px; padding: 13px; background: #f5f8fa; border-radius: 8px; color: #536c7b; }
.crop-identity > svg { margin-top: 2px; flex: none; }
.crop-identity p { font-size: 11px; line-height: 1.7; }
.crop-load-error { margin: 8px 22px 18px; font-size: 12px; line-height: 1.7; padding: 12px; border-radius: 8px; background: #fbefed; color: var(--danger); }
.crop-loading { margin: 8px 22px 16px; color: var(--muted); font-size: 12px; }
.crop-decision-panel { padding: 21px; }
.crop-decision-heading { display: flex; gap: 11px; align-items: center; margin-bottom: 20px; }
.crop-symbol { display: grid; place-items: center; flex: none; width: 40px; height: 43px; border: 1px solid #dfebed; border-radius: 9px; background: #edf5f6; color: var(--brand); }
.crop-status { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 600; line-height: 1.6; border-radius: 7px; padding: 11px 12px; background: #f1f5f7; color: #516a79; }
.crop-status-accepted { background: #eaf5ef; color: #286846; }
.crop-status-rejected { background: #fff0ee; color: #963f37; }
.crop-reviewed-label { display: flex; align-items: flex-start; gap: 9px; font-size: 12px; font-weight: 600; margin-top: 23px; cursor: pointer; }
.crop-reviewed-label input { width: 16px; height: 16px; margin: 2px 0 0; flex: none; accent-color: var(--brand); }
.crop-hint { font-size: 11px; color: #5b7381; line-height: 1.7; margin-top: 8px; }
.crop-accept { margin-top: 17px; width: 100%; padding: 10px 11px; font-size: 12px; }
.crop-reject-section { border-top: 1px solid #e6edf0; padding-top: 20px; margin-top: 23px; }
.crop-reject-section > label { display: block; font-size: 12px; font-weight: 600; margin-bottom: 9px; }
.crop-reject-section select { width: 100%; min-width: 0; min-height: 44px; padding: 10px 11px; color: var(--ink); background: white; border: 1px solid #ccd9e1; border-radius: 8px; font-size: 12px; }
.crop-reject { margin-top: 13px; width: 100%; color: #963f37; padding: 10px; font-size: 12px; }
.crop-decision-panel .field-error { margin-top: 8px; line-height: 1.7; }
.crop-record-note { font-size: 12px; line-height: 1.7; color: var(--muted); margin-top: 17px; }
.crop-rejected-reason { display: grid; gap: 5px; margin-top: 18px; padding: 12px; background: #f8f5f4; border-radius: 7px; }
.crop-rejected-reason > span { font-size: 11px; color: var(--muted); }
.crop-rejected-reason > strong { font-size: 12px; font-weight: 600; }
.crop-reopen { margin-top: 18px; width: 100%; }
.crop-session-note { display: flex; align-items: flex-start; gap: 8px; border-top: 1px solid #e1e9ed; padding-top: 18px; margin-top: 22px; color: #5c7482; }
.crop-session-note > svg { flex: none; margin-top: 2px; }
.crop-session-note p { font-size: 11px; line-height: 1.7; }
.crop-actions { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 24px; }
.crop-primary-actions { display: flex; gap: 10px; flex-wrap: wrap; }
.crop-feedback { color: var(--brand-dark); font-size: 12px; margin-top: 13px; }
.crop-feedback:empty { display: none; }
.crop-dirty-note { color: var(--muted); font-size: 11px; margin-top: 12px; }
.crop-action-error { font-size: 12px; line-height: 1.7; color: var(--danger); background: #fbefed; padding: 12px 14px; border-radius: 8px; margin-top: 18px; }
.crop-dialog-actions { display: flex; justify-content: flex-end; gap: 10px; flex-wrap: wrap; margin-top: 24px; }
.crop-enlarged { width: min(840px,calc(100% - 32px)); }
.crop-enlarged .dialog-heading h2 { margin: 0; }
.crop-enlarged > :deep(svg) { display: block; max-height: 62dvh; width: 100%; height: auto; background: #13222e; border-radius: 8px; margin-bottom: 15px; }
.crop-enlarged > .button { margin-top: 8px; }
.crop-unavailable { display: grid; justify-items: start; gap: 18px; padding: 34px; background: white; border: 1px solid var(--border); border-radius: 12px; }
.crop-unavailable h1 { font-size: 27px; letter-spacing: -.03em; }
.crop-unavailable p { color: var(--muted); font-size: 14px; }
@media(max-width:1120px) { .crop-layout { grid-template-columns: 1fr; } .crop-decision-panel { max-width: none; } .crop-views { gap: 22px; } }
@media(max-width:720px) {
  .crop-heading { align-items: flex-start; flex-direction: column; gap: 17px; margin-bottom: 24px; }
  .crop-heading p { font-size: 13px; line-height: 1.7; }
  .crop-stepper { grid-template-columns: repeat(2,minmax(0,1fr)); gap: 14px 12px; }
  .crop-stepper li { font-size: 11px; }
  .crop-scope-note { padding: 12px; }
  .crop-scope-note p { font-size: 11px; }
  .crop-panel-heading { padding: 18px; }
  .crop-panel-heading h2 { font-size: 15px; }
  .crop-views { grid-template-columns: 1fr; padding: 19px 18px 12px; gap: 23px; }
  .crop-views figcaption { font-size: 13px; }
  .crop-preview-frame { min-height: 210px; }
  .crop-preview-frame :deep(svg) { max-height: 300px; }
  .crop-identity { margin: 7px 18px 18px; }
  .crop-decision-panel { padding: 20px 18px; }
  .crop-actions { align-items: stretch; }
  .crop-actions > .button,.crop-primary-actions { width: 100%; }
  .crop-primary-actions > .button { flex: 1 1 130px; }
  .crop-unavailable { padding: 24px; }
  .crop-dialog-actions > .button { flex: 1 1 180px; }
}
@media(pointer:coarse) { .crop-reject-section select { font-size: 16px; } .crop-views .text-button { min-height: 40px; } }
</style>
