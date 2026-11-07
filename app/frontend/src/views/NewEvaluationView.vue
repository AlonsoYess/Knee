<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { onBeforeRouteLeave, useRoute, useRouter } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import {
  evaluations, createDraftInput, validateDemoDraft, saveDemoDraft,
  isEditableDemoEvaluation,
  type DemoDraftInput,
} from '../state/evaluations'
import { ApiError } from '../api/http'
import { apiMessage } from '../state/session'

const route = useRoute()
const router = useRouter()
const requestedId = typeof route.params.id === 'string' ? route.params.id : undefined
const original = requestedId ? evaluations.value.find(item => item.id === requestedId) : undefined
const unavailable = Boolean(requestedId && !isEditableDemoEvaluation(original))
const form = reactive<DemoDraftInput>(createDraftInput(!unavailable ? original : undefined))
const savedId = ref<string | undefined>(!unavailable ? original?.id : undefined)
// Retained only while this form is open; retrying an uncertain creation keeps the same intent.
const creationIntent = crypto.randomUUID()
const baseline = ref(JSON.stringify(form))
const errors = ref<Partial<Record<keyof DemoDraftInput, string>>>({})
const message = ref('')
const saveError = ref('')
const saving = ref(false)
let navigatingAfterSave = false
const sideChangedSinceSave = ref(false)
const confirmationDialog = ref<HTMLDialogElement | null>(null)
const confirmationMode = ref<'leave' | 'clear'>('leave')
let resolveConfirmation: ((value: boolean) => void) | null = null

const steps = ['Datos basales', 'Radiografía y lado', 'Revisión del recorte', 'Resumen']
const dirty = computed(() => !unavailable && (sideChangedSinceSave.value || JSON.stringify(form) !== baseline.value))
const hasFields = computed(() => Boolean(form.age || form.sex || form.bmi || form.knee))
const title = original ? 'Editar borrador' : 'Nueva evaluación'

function fieldChanged(field: keyof DemoDraftInput) {
  if (field === 'knee') sideChangedSinceSave.value = true
  delete errors.value[field]
  message.value = ''
  saveError.value = ''
}

function fillExample() {
  const missing: Partial<DemoDraftInput> = {}
  if (!form.age.trim()) missing.age = '62'
  if (!form.sex) missing.sex = 'F'
  if (!form.bmi.trim()) missing.bmi = '28.4'
  if (!form.knee) missing.knee = 'right'
  const filledFields = Object.keys(missing) as (keyof DemoDraftInput)[]
  if (!filledFields.length) {
    message.value = 'Todos los campos ya tienen un valor. No se modificaron tus datos.'
    return
  }
  if (missing.knee) sideChangedSinceSave.value = true
  Object.assign(form, missing)
  for (const field of filledFields) delete errors.value[field]
  saveError.value = ''
  message.value = 'Campos pendientes completados con datos ficticios. Se conservaron los valores que ya escribiste.'
}

function askConfirmation(mode: 'leave' | 'clear'): Promise<boolean> {
  if (resolveConfirmation) return Promise.resolve(false)
  confirmationMode.value = mode
  return new Promise(resolve => {
    resolveConfirmation = resolve
    confirmationDialog.value?.showModal()
  })
}

function answerConfirmation(confirmed: boolean) {
  const resolve = resolveConfirmation
  resolveConfirmation = null
  confirmationDialog.value?.close()
  resolve?.(confirmed)
}

function confirmationClosed() {
  const resolve = resolveConfirmation
  resolveConfirmation = null
  resolve?.(false)
}

async function clearForm() {
  if (hasFields.value && !await askConfirmation('clear')) return
  if (form.knee) sideChangedSinceSave.value = true
  Object.assign(form, { age: '', sex: '', bmi: '', knee: '' })
  errors.value = {}
  saveError.value = ''
  message.value = 'Campos vaciados. El código de la evaluación se conserva.'
  await nextTick()
  document.getElementById('draft-age')?.focus()
}

async function validate(requireComplete: boolean): Promise<boolean> {
  errors.value = validateDemoDraft(form, requireComplete)
  saveError.value = ''
  const firstField = (['code', 'age', 'sex', 'bmi', 'knee'] as const).find(field => errors.value[field])
  if (!firstField) return true
  message.value = ''
  await nextTick()
  document.getElementById(firstField === 'knee' ? 'draft-knee-right' : `draft-${firstField}`)?.focus()
  return false
}

async function saveCurrent(): Promise<boolean> {
  try {
    const result = await saveDemoDraft(form, savedId.value, { invalidateImageSide: sideChangedSinceSave.value, idempotencyKey: creationIntent })
    savedId.value = result.id
    Object.assign(form, createDraftInput(result))
    sideChangedSinceSave.value = false
    baseline.value = JSON.stringify(form)
    return true
  } catch (error) {
    saveError.value = apiMessage(error)
    if (error instanceof ApiError) {
      for (const field of error.fields) {
        const name = field.path.split(/[./]/).filter(Boolean).pop()
        const key = name === 'side' ? 'knee' : name
        if (key === 'age' || key === 'sex' || key === 'bmi' || key === 'knee') errors.value[key] = field.message
      }
    }
    return false
  }
}

async function saveDraft() {
  if (saving.value || !await validate(false)) return
  saving.value = true
  try {
    if (!await saveCurrent()) return
    navigatingAfterSave = true
    await router.push('/evaluaciones')
  } finally { navigatingAfterSave = false; saving.value = false }
}

async function continueToImage() {
  if (saving.value || !await validate(true)) return
  saving.value = true
  try {
    if (!await saveCurrent()) return
    navigatingAfterSave = true
    await router.push({ name: 'radiograph', params: { id: savedId.value } })
  } finally { navigatingAfterSave = false; saving.value = false }
}

function beforeUnload(event: BeforeUnloadEvent) {
  if (!dirty.value) return
  event.preventDefault()
  event.returnValue = ''
}

onBeforeRouteLeave(async () => {
  if (navigatingAfterSave && !dirty.value) return true
  return !saving.value && (!dirty.value || await askConfirmation('leave'))
})
onMounted(() => window.addEventListener('beforeunload', beforeUnload))
onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', beforeUnload)
  resolveConfirmation?.(false)
  resolveConfirmation = null
})
</script>

<template>
  <div class="new-page">
    <RouterLink to="/evaluaciones" class="new-back"><AppIcon name="chevron-left" :size="16" />Volver a Mis evaluaciones</RouterLink>

    <template v-if="unavailable">
      <section class="new-unavailable">
        <span class="new-section-icon"><AppIcon name="file" :size="25" /></span>
        <h1 tabindex="-1">Borrador no disponible</h1>
        <p>El caso solicitado no existe en esta demostración o ya no es un borrador editable.</p>
        <RouterLink to="/evaluaciones" class="button button-primary">Volver al listado</RouterLink>
      </section>
    </template>

    <template v-else>
      <header class="new-heading">
        <div><span class="section-eyebrow">PREPARAR UN CASO</span><h1 tabindex="-1">{{ title }}</h1><p>Comienza con los datos de la radiografía inicial.</p></div>
        <span class="new-code-badge"><AppIcon name="file" :size="15" />{{ form.code }}</span>
      </header>

      <ol class="new-stepper" aria-label="Recorrido de evaluación">
        <li v-for="(step, index) in steps" :key="step" :class="{ 'new-current-step': index === 0 }" :aria-current="index === 0 ? 'step' : undefined"><span class="new-step-number">{{ index + 1 }}</span><span>{{ step }}</span></li>
      </ol>

      <div class="new-layout">
        <form class="new-form-panel" :aria-busy="saving" novalidate @submit.prevent="continueToImage">
          <div class="new-panel-heading"><div class="new-panel-title"><span class="new-section-icon"><AppIcon name="file" /></span><div><h2>Datos basales</h2><p>Una evaluación corresponde a una rodilla.</p></div></div><button type="button" class="text-button"  :disabled="saving" @click="fillExample">Completar ejemplo<AppIcon name="arrow-up-right" :size="15" /></button></div>

          <div class="new-fields">
            <div class="field new-code-field"><label for="draft-code">Código de evaluación</label><div class="new-code-input"><input id="draft-code" :value="form.code" readonly :aria-invalid="Boolean(errors.code)" :aria-describedby="errors.code ? 'draft-code-error' : 'draft-code-hint'" /><AppIcon name="lock" :size="16" /></div><p id="draft-code-hint" class="new-field-hint">Lo asigna el servidor cuando confirma el primer guardado.</p><p v-if="errors.code" id="draft-code-error" class="field-error" role="alert">{{ errors.code }}</p></div>

            <div class="new-field-grid">
              <div class="field"><label for="draft-age">Edad en la radiografía <span>(años)</span></label><input id="draft-age" :disabled="saving" v-model="form.age" type="text" inputmode="numeric" autocomplete="off" placeholder="Ejemplo: 62" :aria-invalid="Boolean(errors.age)" :aria-describedby="errors.age ? 'draft-age-error' : undefined" @input="fieldChanged('age')" /><p v-if="errors.age" id="draft-age-error" class="field-error" role="alert">{{ errors.age }}</p></div>
              <div class="field"><label for="draft-sex">Sexo registrado</label><select id="draft-sex" :disabled="saving" v-model="form.sex" :aria-invalid="Boolean(errors.sex)" :aria-describedby="errors.sex ? 'draft-sex-error' : undefined" @change="fieldChanged('sex')"><option value="">Seleccionar</option><option value="F">Femenino</option><option value="M">Masculino</option></select><p v-if="errors.sex" id="draft-sex-error" class="field-error" role="alert">{{ errors.sex }}</p></div>
            </div>

            <div class="field new-bmi-field"><label for="draft-bmi">IMC en la radiografía <span>(kg/m²)</span></label><input id="draft-bmi" :disabled="saving" v-model="form.bmi" type="text" inputmode="decimal" autocomplete="off" placeholder="Ejemplo: 28,4" :aria-invalid="Boolean(errors.bmi)" :aria-describedby="errors.bmi ? 'draft-bmi-error' : 'draft-bmi-hint'" @input="fieldChanged('bmi')" /><p id="draft-bmi-hint" class="new-field-hint">Usa el valor basal. Puedes escribir coma o punto decimal.</p><p v-if="errors.bmi" id="draft-bmi-error" class="field-error" role="alert">{{ errors.bmi }}</p></div>

            <fieldset class="new-knee-field" :aria-describedby="errors.knee ? 'draft-knee-error' : 'draft-knee-hint'"><legend>Rodilla a evaluar</legend><div class="new-knee-choices"><label :class="['new-knee-option', { 'new-knee-selected': form.knee === 'right' }]"><input id="draft-knee-right" :disabled="saving" v-model="form.knee" type="radio" name="evaluation-knee" value="right" :aria-invalid="Boolean(errors.knee)" @change="fieldChanged('knee')" /><span>Derecha</span></label><label :class="['new-knee-option', { 'new-knee-selected': form.knee === 'left' }]"><input id="draft-knee-left" :disabled="saving" v-model="form.knee" type="radio" name="evaluation-knee" value="left" :aria-invalid="Boolean(errors.knee)" @change="fieldChanged('knee')" /><span>Izquierda</span></label></div><p id="draft-knee-hint" class="new-field-hint">El lado se confirmará nuevamente al revisar la imagen.</p><p v-if="errors.knee" id="draft-knee-error" class="field-error" role="alert">{{ errors.knee }}</p></fieldset>
          </div>

          <div class="new-form-note"><AppIcon name="clock" :size="17" /><p>Usa solo valores ficticios. Deja vacíos los datos ausentes; no se inventan valores ni se excluye el caso por IMC ausente.</p></div>
          <div v-if="saveError" class="new-save-error" role="alert">{{ saveError }}</div>
          <div class="new-form-actions"><button type="button" class="text-button new-clear" :disabled="!hasFields || saving" @click="clearForm">Limpiar campos</button><div class="new-main-actions"><button type="button" class="button button-secondary" :disabled="saving" @click="saveDraft">Guardar y salir</button><button type="submit" class="button button-primary" :disabled="saving">Guardar y continuar<AppIcon name="arrow-right" :size="17" /></button></div></div>
          <p class="new-feedback" role="status" aria-live="polite">{{ message }}</p>
        </form>

        <aside class="new-context" aria-label="Guía para preparar el caso">
          <div class="new-context-intro"><span class="new-context-icon"><AppIcon name="evaluations" :size="25" /></span><h2>Primero, prepara los datos</h2><p>Revisa la información de una rodilla antes de continuar con su imagen.</p></div>
          <div class="new-context-section"><h3>Puedes guardar y retomar</h3><p>Los datos basales se guardan en el servidor. Para cargar el DICOM, solo necesitas indicar la rodilla.</p></div>
          <div class="new-context-section"><h3>Solo datos de ejemplo</h3><p>Utiliza valores ficticios. El siguiente paso permite cargar un DICOM y registrar la preparación en el servidor.</p></div>
          <div class="new-session-note"><AppIcon name="help" :size="17" /><p>El guardado se confirma solo cuando responde el servidor. Si hay un error, tus campos permanecen aquí sin confirmar. Los cambios sin guardar pueden perderse al recargar.</p></div>
        </aside>
      </div>
      <p class="new-bottom-note"><span :class="['new-draft-dot', { 'new-draft-dirty': dirty }]" aria-hidden="true"></span>{{ dirty ? 'Tienes cambios sin guardar.' : savedId ? 'Datos basales confirmados en el servidor.' : 'Los campos se guardan al pulsar Guardar y salir o Guardar y continuar.' }}</p>
    </template>

    <dialog ref="confirmationDialog" class="app-dialog" aria-labelledby="new-confirmation-title" @cancel.prevent="answerConfirmation(false)" @close="confirmationClosed"><div class="dialog-heading"><span class="dialog-symbol"><AppIcon name="file" :size="24" /></span><button type="button" class="icon-button" aria-label="Cancelar y seguir editando" autofocus @click="answerConfirmation(false)"><AppIcon name="x" /></button></div><h2 id="new-confirmation-title">{{ confirmationMode === 'leave' ? 'Tienes cambios sin guardar' : '¿Limpiar los campos?' }}</h2><p>{{ confirmationMode === 'leave' ? 'Puedes seguir editando para conservarlos o salir descartando los cambios de esta pantalla.' : 'Se vaciarán los datos del formulario. El código del caso se conservará y el borrador guardado solo cambiará cuando vuelvas a guardarlo.' }}</p><div class="new-summary-actions"><button type="button" class="button button-secondary" @click="answerConfirmation(false)">Seguir editando</button><button type="button" class="button button-primary" @click="answerConfirmation(true)">{{ confirmationMode === 'leave' ? 'Salir sin guardar' : 'Limpiar campos' }}</button></div></dialog>
  </div>
</template>

<style scoped>
.new-page { width: 100%; max-width: 1240px; margin: 0 auto; }
.new-back { display: inline-flex; align-items: center; gap: 6px; text-decoration: none; font-size: 12px; color: var(--muted); margin-bottom: 25px; }
.new-back:hover { color: var(--brand); text-decoration: underline; text-underline-offset: 3px; }
.new-heading { display: flex; justify-content: space-between; align-items: center; gap: 20px; margin-bottom: 26px; }
.new-heading h1 { margin-top: 8px; font-size: clamp(26px, 3vw, 32px); line-height: 1.2; font-weight: 650; letter-spacing: -.035em; }
.new-heading p { color: var(--muted); font-size: 14px; margin-top: 10px; }
.new-code-badge { display: inline-flex; gap: 7px; align-items: center; flex: none; padding: 7px 10px; border: 1px solid var(--border); border-radius: 7px; background: #fff; color: #4d6878; font-size: 12px; font-weight: 600; }
.new-stepper { list-style: none; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); padding: 0; margin: 0 0 27px; gap: 10px; }
.new-stepper li { display: flex; align-items: center; gap: 8px; padding-bottom: 14px; border-bottom: 2px solid #dce6eb; color: #657c8b; font-size: 12px; }
.new-stepper .new-current-step { color: var(--brand-dark); border-bottom-color: var(--brand); font-weight: 600; }
.new-step-number { display: grid; place-items: center; flex: none; width: 26px; height: 26px; border-radius: 50%; background: #e7edf1; color: #607684; font-size: 11px; }
.new-current-step .new-step-number { background: var(--brand); color: white; }
.new-layout { display: grid; grid-template-columns: minmax(0, 1fr) 265px; gap: 24px; align-items: start; }
.new-form-panel { background: #fff; border: 1px solid var(--border); border-radius: 12px; min-width: 0; overflow: hidden; }
.new-panel-heading { padding: 22px 25px; border-bottom: 1px solid #e8eef1; display: flex; justify-content: space-between; align-items: center; gap: 18px; flex-wrap: wrap; }
.new-panel-title { display: flex; align-items: center; gap: 11px; }
.new-section-icon { display: grid; place-items: center; width: 40px; height: 43px; background: #edf5f6; color: var(--brand); border: 1px solid #dfebed; border-radius: 9px; flex: none; }
.new-panel-title h2 { font-size: 16px; font-weight: 650; }
.new-panel-title p { font-size: 12px; color: var(--muted); margin-top: 3px; }
.new-panel-heading > .text-button { font-size: 12px; }
.new-fields { padding: 24px 25px; display: grid; gap: 24px; }
.new-field-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.new-page .field input, .new-page .field select { min-width: 0; font-size: 14px; }
.new-page .field label > span { color: var(--muted); font-weight: 400; }
.new-field-hint { font-size: 11px; color: #617785; line-height: 1.6; }
.new-code-field { max-width: 360px; }
.new-code-input { position: relative; }
.new-code-input input { background: #f7fafb; color: #526e7d; padding-right: 40px; }
.new-code-input > svg { position: absolute; right: 14px; top: 15px; color: #6d8591; }
.new-bmi-field { max-width: 360px; }
.new-knee-field { margin: 0; padding: 0; border: 0; min-width: 0; }
.new-knee-field legend { font-size: 13px; font-weight: 600; padding: 0; margin-bottom: 10px; }
.new-knee-choices { display: flex; gap: 12px; margin-bottom: 8px; }
.new-knee-option { border: 1px solid #cddae2; background: white; border-radius: 8px; padding: 12px 15px; min-height: 46px; min-width: 130px; display: flex; align-items: center; gap: 10px; font-size: 13px; cursor: pointer; }
.new-knee-selected { border-color: var(--brand); background: #f1f8f7; color: var(--brand-dark); }
.new-knee-option input { margin: 0; width: 16px; height: 16px; accent-color: var(--brand); }
.new-knee-field .field-error { margin-top: 7px; }
.new-form-note { display: flex; gap: 8px; align-items: flex-start; margin: 0 25px 22px; padding: 12px 14px; background: #f4f8fa; border-radius: 8px; color: #4f6b7c; }
.new-form-note svg { margin-top: 2px; }
.new-form-note p { font-size: 12px; line-height: 1.7; }
.new-form-actions { padding: 20px 25px; border-top: 1px solid #e8eef1; display: flex; align-items: center; justify-content: space-between; gap: 15px; }
.new-main-actions { display: flex; gap: 10px; flex-wrap: wrap; }
.new-clear { font-size: 12px; color: #597382; font-weight: 500; }
.new-feedback { color: var(--brand-dark); font-size: 12px; padding: 0 25px; }
.new-feedback:not(:empty) { padding-bottom: 18px; }
.new-save-error { margin: 0 25px 20px; padding: 12px 14px; background: #fbefed; border-radius: 7px; color: var(--danger); font-size: 12px; }
.new-context { padding: 21px 3px 0; }
.new-context-icon { display: inline-flex; color: var(--brand); margin-bottom: 13px; }
.new-context h2 { font-size: 17px; line-height: 1.45; font-weight: 600; margin-bottom: 10px; }
.new-context-intro > p, .new-context-section p { color: #5c7483; font-size: 12px; line-height: 1.8; }
.new-context-section { padding-top: 20px; margin-top: 20px; border-top: 1px solid #dce6eb; }
.new-context-section h3 { font-size: 12px; font-weight: 650; color: #3d5b6c; margin-bottom: 7px; }
.new-session-note { display: flex; gap: 8px; align-items: flex-start; margin-top: 25px; padding: 14px; border-radius: 8px; border: 1px solid #d9e6e9; color: #567381; }
.new-session-note > svg { margin-top: 1px; }
.new-session-note p { font-size: 11px; line-height: 1.7; }
.new-bottom-note { display: flex; gap: 7px; align-items: center; color: #5e7684; font-size: 11px; margin: 17px 0 0; }
.new-draft-dot { width: 6px; height: 6px; border-radius: 50%; background: #6a8f8a; flex: none; }
.new-draft-dirty { background: #b48332; }
.new-summary-data { margin: 20px 0; padding: 18px 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.new-summary-data dt { color: var(--muted); font-size: 12px; }
.new-summary-data dd { color: var(--ink); font-size: 14px; font-weight: 600; margin: 4px 0 0; }
.new-summary-actions { display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 10px; margin-top: 24px; }
.new-unavailable { display: grid; justify-items: start; gap: 17px; border: 1px solid var(--border); border-radius: 12px; background: white; padding: 30px; }
.new-unavailable h1 { font-size: 26px; letter-spacing: -.4px; line-height: 1.3; }
.new-unavailable p { max-width: 520px; color: var(--muted); }
@media(max-width:1150px) { .new-layout { grid-template-columns: minmax(0, 1fr) 220px; gap: 20px; } .new-form-actions { align-items: flex-start; flex-direction: column; } .new-main-actions { width: 100%; justify-content: flex-end; } }
@media(max-width:980px) { .new-layout { grid-template-columns: 1fr; } .new-context { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 18px; padding: 0; } .new-context-intro { grid-column: 1 / -1; } .new-context-icon { display: none; } .new-context-section { margin: 0; padding-top: 16px; } .new-session-note { grid-column: 1 / -1; margin-top: 0; } .new-main-actions { width: auto; } .new-form-actions { flex-direction: row; align-items: center; } }
@media(max-width:600px) { .new-heading { align-items: flex-start; flex-direction: column; gap: 14px; } .new-heading p { font-size: 13px; } .new-back { margin-bottom: 20px; } .new-stepper { grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px 15px; margin-bottom: 22px; } .new-stepper li { font-size: 11px; padding-bottom: 10px; } .new-step-number { width: 23px; height: 23px; } .new-panel-heading { padding: 18px; gap: 15px; } .new-fields { padding: 21px 18px; gap: 22px; } .new-field-grid { grid-template-columns: 1fr; gap: 22px; } .new-bmi-field,.new-code-field { max-width: none; } .new-form-note { margin: 0 18px 18px; } .new-form-actions { padding: 18px; flex-direction: column; align-items: stretch; } .new-clear { align-self: flex-start; } .new-main-actions { display: grid; grid-template-columns: 1fr; width: 100%; } .new-main-actions > .button { width: 100%; } .new-context { grid-template-columns: 1fr; } .new-context-intro,.new-session-note { grid-column: auto; } .new-knee-option { flex: 1; min-width: 0; padding: 12px; } .new-summary-actions { justify-content: stretch; } .new-summary-actions > .button { flex: 1 1 100%; } .new-feedback { padding-left: 18px; padding-right: 18px; } }
@media(pointer:coarse) { .new-page .field input, .new-page .field select { font-size: 16px; } .new-knee-option { min-height: 48px; } }
</style>
