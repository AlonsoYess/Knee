<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { onBeforeRouteLeave, useRoute, useRouter } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import { closeOnBackdrop } from '../dialog'
import {
  evaluations, createDraftInput, validateDemoDraft,
  createDemoImageInput, validateDemoImage, saveDemoImage,
  type DemoImageInput,
  isEditableDemoEvaluation,
} from '../state/evaluations'
import { demoImages } from '../demo/demoImages'

const route = useRoute()
const router = useRouter()
const requestedId = typeof route.params.id === 'string' ? route.params.id : ''
const existing = computed(() => evaluations.value.find(item => item.id === requestedId))
const unavailable = computed(() => !isEditableDemoEvaluation(existing.value))
const baselineIncomplete = computed(() => Boolean(existing.value && Object.keys(validateDemoDraft(createDraftInput(existing.value), true)).length))
const form = reactive<DemoImageInput>(existing.value ? createDemoImageInput(existing.value) : { sampleId: '', knee: '', sideConfirmed: false })
const baseline = ref(JSON.stringify(form))
const errors = ref<Partial<Record<keyof DemoImageInput, string>>>({})
const saveError = ref('')
const saving = ref(false)
const previewFailed = ref(false)
const loadedPreviewId = ref<DemoImageInput['sampleId']>('')
const sourceChangedSinceSave = ref(false)
const message = ref('')
const enlarged = ref<HTMLDialogElement | null>(null)
const confirmation = ref<HTMLDialogElement | null>(null)
const confirmationMode = ref<'leave' | 'remove'>('leave')
let resolveConfirmation: ((value: boolean) => void) | null = null

const steps = ['Datos basales', 'Radiografía y lado', 'Revisión del recorte', 'Resumen']
const selectedImage = computed(() => demoImages.find(image => image.id === form.sampleId))
const baselineSide = computed(() => existing.value?.knee === 'right' ? 'Derecha del paciente' : existing.value?.knee === 'left' ? 'Izquierda del paciente' : 'Pendiente')
const sideMismatch = computed(() => Boolean(form.knee && existing.value?.knee && form.knee !== existing.value.knee))
const previewLoaded = computed(() => Boolean(selectedImage.value && loadedPreviewId.value === selectedImage.value.id && !previewFailed.value))
const canConfirm = computed(() => Boolean(previewLoaded.value && form.knee && !sideMismatch.value))
const dirty = computed(() => !unavailable.value && !baselineIncomplete.value && (sourceChangedSinceSave.value || JSON.stringify(form) !== baseline.value))
const editRoute = computed(() => ({ name: 'edit-evaluation', params: { id: requestedId } }))

function clearMessages() { errors.value = {}; saveError.value = ''; message.value = '' }
function selectImage(id: DemoImageInput['sampleId']) {
  if (form.sampleId === id) return
  sourceChangedSinceSave.value = true
  form.sampleId = id
  form.sideConfirmed = false
  previewFailed.value = false
  loadedPreviewId.value = ''
  clearMessages()
  message.value = 'Imagen de ejemplo seleccionada. Confirma el lado antes de continuar.'
}
function sideChanged() {
  sourceChangedSinceSave.value = true
  form.sideConfirmed = false
  clearMessages()
}
function previewReady(event: Event) {
  if ((event.currentTarget as HTMLImageElement).getAttribute('src') !== selectedImage.value?.src) return
  loadedPreviewId.value = form.sampleId
  previewFailed.value = false
  delete errors.value.sampleId
}
function previewError(event: Event) {
  if ((event.currentTarget as HTMLImageElement).getAttribute('src') !== selectedImage.value?.src) return
  previewFailed.value = true
  loadedPreviewId.value = ''
  form.sideConfirmed = false
  sourceChangedSinceSave.value = true
}
function ask(mode: 'leave' | 'remove'): Promise<boolean> {
  if (resolveConfirmation) return Promise.resolve(false)
  confirmationMode.value = mode
  return new Promise(resolve => {
    resolveConfirmation = resolve
    confirmation.value?.showModal()
  })
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
async function removeImage() {
  if (!form.sampleId || !await ask('remove')) return
  sourceChangedSinceSave.value = true
  form.sampleId = ''
  form.sideConfirmed = false
  previewFailed.value = false
  loadedPreviewId.value = ''
  clearMessages()
  message.value = 'Imagen retirada del borrador. Puedes seleccionar otro ejemplo.'
  await nextTick()
  document.getElementById('radiograph-sample-0')?.focus()
}
async function validate(requireComplete: boolean): Promise<boolean> {
  if (!existing.value || unavailable.value || baselineIncomplete.value) return false
  errors.value = validateDemoImage(form, existing.value, requireComplete)
  if (requireComplete && previewFailed.value) errors.value.sampleId = 'No pudimos mostrar esta imagen. Selecciona otro ejemplo.'
  else if (requireComplete && selectedImage.value && !previewLoaded.value) errors.value.sampleId = 'Espera a que cargue la imagen antes de continuar.'
  saveError.value = ''
  const first = (['sampleId', 'knee', 'sideConfirmed'] as const).find(key => errors.value[key])
  if (!first) return true
  await nextTick()
  document.getElementById(first === 'sampleId' ? 'radiograph-sample-0' : first === 'knee' ? 'radiograph-side-right' : 'radiograph-confirmation')?.focus()
  return false
}
function saveCurrent(): boolean {
  try {
    saveDemoImage(form, requestedId, { invalidateReview: sourceChangedSinceSave.value })
    sourceChangedSinceSave.value = false
    baseline.value = JSON.stringify(form)
    return true
  } catch {
    saveError.value = 'No pudimos conservar la selección del ejemplo. Revisa la imagen y el lado o vuelve al listado.'
    return false
  }
}
async function saveDraft() {
  if (saving.value || !await validate(false)) return
  saving.value = true
  try { if (saveCurrent()) await router.push('/evaluaciones') } finally { saving.value = false }
}
async function continueToReview() {
  if (saving.value || !await validate(true)) return
  saving.value = true
  try {
    if (!saveCurrent()) return
    await router.push({ name: 'crop-review', params: { id: requestedId } })
  } finally { saving.value = false }
}
function beforeUnload(event: BeforeUnloadEvent) {
  if (!dirty.value) return
  event.preventDefault()
  event.returnValue = ''
}
onBeforeRouteLeave(async () => !dirty.value || await ask('leave'))
onMounted(() => window.addEventListener('beforeunload', beforeUnload))
onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', beforeUnload)
  resolveConfirmation?.(false)
  resolveConfirmation = null
})
</script>

<template>
  <div class="rad-page">
    <p class="simulation-notice" role="status">Las imágenes y la revisión siguen siendo una simulación; sus cambios no se guardan en el servidor. Se reinician al recargar, cerrar sesión o cambiar la revisión basal.</p>
    <RouterLink to="/evaluaciones" class="rad-back"><AppIcon name="chevron-left" :size="16" />Volver a Mis evaluaciones</RouterLink>

    <section v-if="unavailable" class="rad-unavailable">
      <span class="rad-section-icon"><AppIcon name="image" :size="25" /></span><h1 tabindex="-1">Borrador no disponible</h1>
      <p>No pudimos recuperar este caso del servidor o no tienes acceso.</p><RouterLink to="/evaluaciones" class="button button-primary">Volver al listado</RouterLink>
    </section>
    <section v-else-if="baselineIncomplete" class="rad-unavailable">
      <span class="rad-section-icon"><AppIcon name="file" :size="25" /></span><h1 tabindex="-1">Datos basales pendientes</h1>
      <p>Guarda la rodilla a explorar en los datos basales. Los otros datos ausentes pueden permanecer vacíos.</p><RouterLink :to="editRoute" class="button button-primary">Completar datos basales</RouterLink>
    </section>
    <template v-else>
      <header class="rad-heading"><div><span class="section-eyebrow">PREPARAR LA IMAGEN</span><h1 tabindex="-1">Radiografía y lado</h1><p>Selecciona un ejemplo y confirma la rodilla que se evaluará.</p></div><span class="rad-code"><AppIcon name="file" :size="15" />{{ existing?.code }}</span></header>
      <ol class="rad-stepper" aria-label="Recorrido de evaluación"><li v-for="(step, index) in steps" :key="step" :class="{ 'rad-step-current': index === 1, 'rad-step-done': index === 0 }" :aria-current="index === 1 ? 'step' : undefined"><span class="rad-step-number"><AppIcon v-if="index === 0" name="check" :size="14" /><template v-else>{{ index + 1 }}</template></span><span>{{ step }}</span></li></ol>

      <form class="rad-layout" novalidate @submit.prevent="continueToReview">
        <section class="rad-image-panel" aria-labelledby="rad-image-title">
          <div class="rad-panel-heading"><div class="rad-panel-title"><span class="rad-section-icon"><AppIcon name="image" /></span><div><h2 id="rad-image-title">Imagen de demostración</h2><p>Ejemplos sintéticos para probar las pantallas.</p></div></div><span class="rad-fiction-label">SIN DATOS CLÍNICOS</span></div>
          <div class="rad-gallery" aria-label="Catálogo de imágenes ficticias">
            <button v-for="(image, index) in demoImages" :id="`radiograph-sample-${index}`" :key="image.id" type="button" :class="['rad-sample', { 'rad-sample-selected': form.sampleId === image.id }]" :aria-label="index === 0 ? 'Seleccionar imagen A' : 'Seleccionar imagen B'" :aria-pressed="form.sampleId === image.id" @click="selectImage(image.id)"><img :src="image.src" alt="" :width="image.width" :height="image.height" /><span class="rad-sample-description"><strong>{{ image.label }}</strong><span>Imagen ficticia</span></span><span v-if="form.sampleId === image.id" class="rad-sample-check"><AppIcon name="check" :size="15" /></span></button>
          </div>
          <p v-if="errors.sampleId" id="radiograph-sample-error" class="field-error rad-image-error" role="alert">{{ errors.sampleId }}</p>
          <div v-if="selectedImage" class="rad-preview-section">
            <div class="rad-preview-heading"><div><h3>Vista de la imagen seleccionada</h3><p>{{ selectedImage.fileName }}</p></div><button type="button" class="text-button" :disabled="!previewLoaded" @click="enlarged?.showModal()"><AppIcon name="scan" :size="16" />Ampliar imagen</button></div>
            <div class="rad-preview" :aria-busy="!previewLoaded && !previewFailed"><img :key="selectedImage.id" :src="selectedImage.src" :alt="`${selectedImage.label}: imagen sintética sin uso clínico`" :width="selectedImage.width" :height="selectedImage.height" @error="previewError" @load="previewReady" /></div>
            <p v-if="!previewLoaded && !previewFailed" class="rad-hint" role="status" aria-live="polite">Cargando la imagen. Espera a verla para confirmar el lado y continuar.</p>
            <p v-if="previewFailed" class="field-error" role="alert">No pudimos mostrar el ejemplo. Selecciona otra imagen.</p>
            <div class="rad-preview-footer"><span>Imagen ficticia · {{ selectedImage.width }} × {{ selectedImage.height }} px</span><button type="button" class="text-button rad-remove" @click="removeImage"><AppIcon name="x" :size="15" />Quitar imagen</button></div>
          </div>
          <div v-else class="rad-empty-preview"><span><AppIcon name="scan" :size="32" /></span><h3>Aquí verás la imagen</h3><p>Selecciona uno de los dos ejemplos para continuar.</p></div>
          <div class="rad-image-note"><AppIcon name="help" :size="17" /><p>Estos ejemplos son ilustraciones sintéticas. Esta pantalla todavía no recibe archivos de radiografías reales.</p></div>
        </section>

        <aside class="rad-side-panel" aria-labelledby="rad-side-title">
          <div class="rad-side-heading"><span class="rad-section-icon"><AppIcon name="user" /></span><h2 id="rad-side-title">Confirma la rodilla</h2></div>
          <div class="rad-baseline"><span>Rodilla indicada en datos basales</span><strong>{{ baselineSide }}</strong><RouterLink :to="editRoute" class="text-button">Editar datos<AppIcon name="arrow-up-right" :size="14" /></RouterLink></div>
          <fieldset class="rad-side-options" :aria-describedby="errors.knee || sideMismatch ? 'radiograph-side-error' : 'radiograph-side-hint'"><legend>Lado del ejemplo a evaluar</legend><label :class="['rad-side-choice', { 'rad-side-selected': form.knee === 'right' }]"><input id="radiograph-side-right" v-model="form.knee" type="radio" name="radiograph-side" value="right" :aria-invalid="Boolean(errors.knee || sideMismatch)" @change="sideChanged" /><span>Derecha del paciente</span></label><label :class="['rad-side-choice', { 'rad-side-selected': form.knee === 'left' }]"><input id="radiograph-side-left" v-model="form.knee" type="radio" name="radiograph-side" value="left" :aria-invalid="Boolean(errors.knee || sideMismatch)" @change="sideChanged" /><span>Izquierda del paciente</span></label><p id="radiograph-side-hint" class="rad-hint">El lado debe coincidir con los datos del caso.</p><p v-if="errors.knee || sideMismatch" id="radiograph-side-error" class="field-error" role="alert">{{ errors.knee || 'El lado no coincide con los datos basales. Revisa la selección o vuelve a editar los datos.' }}</p></fieldset>
          <label :class="['rad-confirmation', { 'rad-confirmation-disabled': !canConfirm }]"><input id="radiograph-confirmation" v-model="form.sideConfirmed" type="checkbox" :disabled="!canConfirm" :aria-invalid="Boolean(errors.sideConfirmed)" :aria-describedby="errors.sideConfirmed ? 'radiograph-confirmation-error' : 'radiograph-confirmation-hint'" @change="errors.sideConfirmed = undefined; message = ''" /><span>Confirmo el lado de este ejemplo</span></label>
          <p v-if="errors.sideConfirmed" id="radiograph-confirmation-error" class="field-error" role="alert">{{ errors.sideConfirmed }}</p>
          <p id="radiograph-confirmation-hint" class="rad-hint">Esta confirmación identifica el lado. La revisión de calidad del recorte corresponde al siguiente paso.</p>
          <div class="rad-session-note"><AppIcon name="clock" :size="17" /><p>Datos basales: guardados en el servidor. Imagen y revisión: simulación temporal que se reinicia al recargar o cerrar sesión.</p></div>
        </aside>

        <div class="rad-actions-area"><p v-if="saveError" class="rad-save-error" role="alert">{{ saveError }}</p><div class="rad-actions"><RouterLink :to="editRoute" class="button button-secondary"><AppIcon name="chevron-left" :size="16" />Volver a datos basales</RouterLink><div class="rad-main-actions"><button type="button" class="button button-secondary" :disabled="saving" @click="saveDraft">Volver a evaluaciones</button><button type="submit" class="button button-primary" :disabled="saving || Boolean(selectedImage && !previewLoaded)">Continuar con el ejemplo<AppIcon name="arrow-right" :size="17" /></button></div></div><p class="rad-feedback" role="status" aria-live="polite">{{ message }}</p><p class="rad-draft-state">{{ dirty ? 'Tienes cambios sin guardar.' : 'Selecciones aplicadas solo a este recorrido temporal.' }}</p></div>
      </form>
    </template>

    <dialog ref="enlarged" class="app-dialog rad-enlarged" aria-labelledby="rad-enlarged-title" @click="closeOnBackdrop($event, enlarged)"><div class="dialog-heading"><h2 id="rad-enlarged-title">Vista ampliada de la imagen</h2><button type="button" class="icon-button" aria-label="Cerrar vista ampliada" @click="enlarged?.close()"><AppIcon name="x" /></button></div><img v-if="selectedImage" :src="selectedImage.src" :alt="`${selectedImage.label}: ilustración sintética`" :width="selectedImage.width" :height="selectedImage.height" /><p>Imagen ficticia para explorar el recorrido de la aplicación.</p><button type="button" class="button button-primary" @click="enlarged?.close()">Volver a la selección</button></dialog>
    <dialog ref="confirmation" class="app-dialog" aria-labelledby="rad-confirmation-title" @cancel.prevent="answer(false)" @close="confirmationClosed"><div class="dialog-heading"><span class="dialog-symbol"><AppIcon name="image" :size="24" /></span><button type="button" class="icon-button" aria-label="Cancelar y seguir editando" autofocus @click="answer(false)"><AppIcon name="x" /></button></div><h2 id="rad-confirmation-title">{{ confirmationMode === 'leave' ? 'Tienes cambios sin guardar' : '¿Quitar la imagen seleccionada?' }}</h2><p>{{ confirmationMode === 'leave' ? 'Puedes seguir editando o salir descartando los cambios de esta pantalla.' : 'Se retirará el ejemplo de esta pantalla y tendrás que confirmar el lado de nuevo al elegir otra imagen. Esto solo cambia el recorrido temporal; no modifica el servidor.' }}</p><div class="rad-summary-actions"><button type="button" class="button button-secondary" @click="answer(false)">Seguir editando</button><button type="button" class="button button-primary" @click="answer(true)">{{ confirmationMode === 'leave' ? 'Salir sin guardar' : 'Quitar imagen' }}</button></div></dialog>
  </div>
</template>

<style scoped>
.rad-page { width: 100%; max-width: 1240px; margin: 0 auto; }
.rad-back { display: inline-flex; align-items: center; gap: 6px; text-decoration: none; font-size: 12px; color: var(--muted); margin-bottom: 25px; }
.rad-back:hover { color: var(--brand); text-decoration: underline; text-underline-offset: 3px; }
.rad-heading { display: flex; justify-content: space-between; align-items: center; gap: 20px; margin-bottom: 26px; }
.rad-heading h1 { margin-top: 8px; font-size: clamp(26px,3vw,32px); line-height: 1.2; font-weight: 650; letter-spacing: -.035em; }
.rad-heading p { color: var(--muted); font-size: 14px; margin-top: 10px; }
.rad-code { display: inline-flex; align-items: center; gap: 7px; flex: none; padding: 7px 10px; border: 1px solid var(--border); border-radius: 7px; background: white; color: #4d6878; font-size: 12px; font-weight: 600; }
.rad-stepper { list-style: none; display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); padding: 0; margin: 0 0 27px; gap: 10px; }
.rad-stepper li { display: flex; align-items: center; gap: 8px; padding-bottom: 14px; border-bottom: 2px solid #dce6eb; color: #657c8b; font-size: 12px; }
.rad-step-number { width: 26px; height: 26px; flex: none; display: grid; place-items: center; border-radius: 50%; background: #e7edf1; color: #607684; font-size: 11px; }
.rad-stepper .rad-step-current { color: var(--brand-dark); border-bottom-color: var(--brand); font-weight: 600; }
.rad-step-current .rad-step-number { background: var(--brand); color: white; }
.rad-step-done .rad-step-number { background: #e0efec; color: var(--brand); }
.rad-layout { display: grid; grid-template-columns: minmax(0,1fr) 295px; gap: 24px; align-items: start; }
.rad-image-panel,.rad-side-panel { min-width: 0; background: white; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.rad-panel-heading { padding: 22px 24px; border-bottom: 1px solid #e8eef1; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.rad-panel-title,.rad-side-heading { display: flex; align-items: center; gap: 11px; }
.rad-panel-title h2,.rad-side-heading h2 { font-size: 16px; font-weight: 650; }
.rad-panel-title p { margin-top: 3px; font-size: 12px; color: var(--muted); }
.rad-section-icon { display: grid; place-items: center; flex: none; width: 40px; height: 43px; background: #edf5f6; color: var(--brand); border: 1px solid #dfebed; border-radius: 9px; }
.rad-fiction-label { font-size: 9px; letter-spacing: 1px; font-weight: 650; color: #55737b; }
.rad-gallery { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; padding: 22px 24px; }
.rad-sample { display: flex; gap: 11px; align-items: center; min-width: 0; border: 1px solid #d5e0e6; border-radius: 9px; background: white; padding: 10px; color: var(--ink); text-align: left; position: relative; }
.rad-sample:hover { border-color: #83aab2; background: #f7fafb; }
.rad-sample-selected { border-color: var(--brand); background: #f0f8f7; box-shadow: 0 0 0 1px var(--brand); }
.rad-sample img { display: block; width: 63px; height: 62px; flex: none; object-fit: contain; background: #172532; border-radius: 5px; }
.rad-sample-description { display: grid; gap: 4px; padding-right: 15px; min-width: 0; }
.rad-sample-description strong { font-size: 12px; font-weight: 600; }
.rad-sample-description > span { font-size: 11px; color: var(--muted); }
.rad-sample-check { position: absolute; right: 7px; top: 8px; color: var(--brand); }
.rad-image-error { margin: -6px 24px 18px; }
.rad-preview-section { padding: 0 24px; }
.rad-preview-heading { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }
.rad-preview-heading h3 { font-size: 12px; font-weight: 600; }
.rad-preview-heading p { font-size: 11px; color: var(--muted); margin-top: 4px; overflow-wrap: anywhere; }
.rad-preview-heading .text-button { font-size: 12px; }
.rad-preview { border-radius: 9px; overflow: hidden; background: #12212e; display: grid; place-items: center; padding: 12px; }
.rad-preview > img { display: block; width: 100%; max-height: 340px; height: auto; object-fit: contain; }
.rad-preview-footer { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 12px 0 5px; }
.rad-preview-footer > span { font-size: 11px; color: var(--muted); }
.rad-remove { color: #8f4b43; font-size: 12px; }
.rad-empty-preview { margin: 0 24px; min-height: 220px; border: 1px dashed #cbdbe3; border-radius: 9px; background: #f7fafb; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; gap: 9px; padding: 24px; }
.rad-empty-preview > span { color: #6e8c9c; margin-bottom: 4px; }
.rad-empty-preview h3 { font-size: 15px; font-weight: 600; }
.rad-empty-preview p { font-size: 12px; color: var(--muted); }
.rad-image-note { display: flex; gap: 8px; align-items: flex-start; padding: 18px 24px; color: #526e7a; }
.rad-image-note svg { margin-top: 2px; }
.rad-image-note p { font-size: 11px; line-height: 1.65; }
.rad-side-panel { padding: 22px; }
.rad-side-heading { margin-bottom: 22px; }
.rad-baseline { background: #f4f8f9; border: 1px solid #e2eaee; border-radius: 8px; padding: 13px; margin-bottom: 23px; display: grid; gap: 5px; }
.rad-baseline > span { color: #56707f; font-size: 11px; }
.rad-baseline strong { font-size: 17px; font-weight: 600; }
.rad-baseline .text-button { margin-top: 3px; font-size: 11px; }
.rad-side-options { margin: 0; padding: 0; border: none; min-width: 0; display: grid; gap: 10px; }
.rad-side-options legend { padding: 0; margin-bottom: 10px; font-size: 12px; font-weight: 600; }
.rad-side-choice { display: flex; align-items: center; gap: 10px; border: 1px solid #cddae2; border-radius: 8px; padding: 12px 13px; min-height: 45px; font-size: 13px; cursor: pointer; }
.rad-side-selected { border-color: var(--brand); background: #f1f8f7; color: var(--brand-dark); }
.rad-side-choice input { accent-color: var(--brand); width: 16px; height: 16px; margin: 0; }
.rad-hint { color: #5d7482; font-size: 11px; line-height: 1.65; }
.rad-confirmation { display: flex; align-items: flex-start; gap: 9px; font-size: 12px; font-weight: 600; margin-top: 24px; margin-bottom: 9px; cursor: pointer; }
.rad-confirmation input { accent-color: var(--brand); width: 16px; height: 16px; margin: 2px 0 0; flex: none; }
.rad-confirmation-disabled { color: #667b88; cursor: default; }
.rad-session-note { display: flex; align-items: flex-start; gap: 8px; border-top: 1px solid #e1e9ed; padding-top: 20px; margin-top: 22px; color: #5c7482; }
.rad-session-note p { font-size: 11px; line-height: 1.65; }
.rad-actions-area { grid-column: 1 / -1; }
.rad-actions { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.rad-main-actions { display: flex; gap: 10px; flex-wrap: wrap; }
.rad-feedback { color: var(--brand-dark); font-size: 12px; margin-top: 13px; }
.rad-feedback:empty { display: none; }
.rad-draft-state { color: var(--muted); font-size: 11px; margin-top: 13px; }
.rad-save-error { color: var(--danger); font-size: 12px; margin-bottom: 16px; background: #fbefed; padding: 12px 14px; border-radius: 8px; }
.rad-summary-data { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 18px; margin: 22px 0; padding: 18px 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
.rad-summary-data dt { color: var(--muted); font-size: 11px; }
.rad-summary-data dd { margin: 5px 0 0; font-size: 13px; font-weight: 600; }
.rad-summary-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 24px; flex-wrap: wrap; }
.rad-enlarged { width: min(840px,calc(100% - 32px)); }
.rad-enlarged .dialog-heading { gap: 16px; }
.rad-enlarged h2 { margin: 0; font-size: 21px; }
.rad-enlarged > img { display: block; width: 100%; height: auto; max-height: 62dvh; object-fit: contain; background: #12212e; border-radius: 9px; }
.rad-enlarged > p { margin-top: 13px; font-size: 12px; }
.rad-unavailable { display: grid; justify-items: start; gap: 18px; padding: 32px; border: 1px solid var(--border); border-radius: 12px; background: white; }
.rad-unavailable h1 { font-size: 27px; letter-spacing: -.5px; }
.rad-unavailable p { color: var(--muted); }
@media(max-width:1150px) { .rad-layout { grid-template-columns: minmax(0,1fr) 260px; gap: 18px; } .rad-side-panel { padding: 18px; } .rad-gallery { gap: 9px; padding: 18px; } .rad-sample { padding: 8px; gap: 8px; } .rad-sample img { width: 43px; height: 50px; } .rad-panel-heading { padding: 20px 18px; } .rad-preview-section { padding: 0 18px; } .rad-empty-preview { margin: 0 18px; } }
@media(max-width:950px) { .rad-layout { grid-template-columns: 1fr; } .rad-side-panel { display: grid; grid-template-columns: 1fr 1fr; gap: 0 22px; } .rad-side-heading,.rad-session-note { grid-column: 1 / -1; } .rad-baseline { grid-column: 1; grid-row: 2 / 5; align-self: start; } .rad-side-options,.rad-confirmation,.rad-side-panel > .rad-hint,.rad-side-panel > .field-error { grid-column: 2; } .rad-confirmation { margin-top: 18px; } }
@media(max-width:600px) {
  .rad-heading { align-items: flex-start; flex-wrap: wrap; gap: 16px; }
  .rad-heading p { font-size: 13px; line-height: 1.7; }
  .rad-stepper { grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px 15px; }
  .rad-stepper li { font-size: 11px; }
  .rad-gallery { gap: 10px; padding: 17px; }
  .rad-sample { flex-direction: column; align-items: stretch; padding: 9px; gap: 8px; }
  .rad-sample img { width: 100%; height: 78px; }
  .rad-sample-description { padding-right: 0; }
  .rad-sample-description strong { font-size: 11px; }
  .rad-sample-description > span { font-size: 10px; }
  .rad-panel-heading { padding: 18px; }
  .rad-panel-title h2 { font-size: 15px; }
  .rad-panel-title p { font-size: 11px; }
  .rad-preview-heading { gap: 10px; }
  .rad-preview { padding: 5px; }
  .rad-preview-footer > span { font-size: 10px; }
  .rad-image-note { padding: 17px; }
  .rad-side-panel { display: block; }
  .rad-side-heading { margin-bottom: 18px; }
  .rad-actions,.rad-main-actions { width: 100%; }
  .rad-actions > .button,.rad-main-actions > .button { width: 100%; }
  .rad-actions > .button { justify-content: flex-start; }
  .rad-summary-actions { flex-direction: column; }
  .rad-summary-actions .button { width: 100%; }
  .rad-unavailable { padding: 24px; }
}
@media(pointer:coarse) { .rad-confirmation { min-height: 44px; align-items: center; } .rad-side-choice,.rad-sample { min-height: 44px; } }
</style>
