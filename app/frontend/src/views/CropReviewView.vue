<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { onBeforeRouteLeave, useRoute, useRouter } from 'vue-router'
import PreparationImage from '../components/PreparationImage.vue'
import UnsavedChangesDialog from '../components/UnsavedChangesDialog.vue'
import { ApiError } from '../api/http'
import type { Mark, Point, PreparationView, QcObservation, QualityReviewDto, ReviewSubmission } from '../api/preparationContracts'
import { evaluations, loadEvaluation } from '../state/evaluations'
import { getActivePreparation, loadActivePreparation, recordQualityReview } from '../state/preparations'
import { apiMessage, sessionGeneration } from '../state/session'

type Choice = '' | 'yes' | 'no'
const route = useRoute(), router = useRouter(), evaluationId = typeof route.params.id === 'string' ? route.params.id : ''
const existing = computed(() => evaluations.value.find(item => item.id === evaluationId))
const preparation = computed(() => getActivePreparation(evaluationId))
const busy = ref(false), loading = ref(true), error = ref(''), details = ref<string[]>([]), message = ref(''), editing = ref(true), contextChanged = ref(false)
const confirmedReview = ref<QualityReviewDto | null>(null)
const persistedReview = computed(() => confirmedReview.value ?? preparation.value?.activeReview ?? null)
const discardDialog = ref<InstanceType<typeof UnsavedChangesDialog> | null>(null)
const readiness = reactive<Record<PreparationView, boolean>>({ full_a: false, full_b: false, half_a: false, half_b: false, crop_a: false, crop_b: false })
const views: { view: PreparationView; label: string }[] = [
  { view: 'full_a', label: 'Original bilateral — ventana 1–99' }, { view: 'half_a', label: 'Mitad seleccionada — ventana 1–99' }, { view: 'crop_a', label: 'Recorte automático — ventana 1–99' },
  { view: 'full_b', label: 'Original bilateral — ventana 5–95' }, { view: 'half_b', label: 'Mitad seleccionada — ventana 5–95' }, { view: 'crop_b', label: 'Recorte automático — ventana 5–95' },
]
function emptyForm() { return {
  confirmedSide: false, confirmedCheckIds: [] as string[], side_supported: '' as Choice, coverage_complete: '' as Choice,
  visibility: { medial_joint_borders: '', lateral_joint_borders: '', subchondral_bone: '' }, hardware: '', text_or_mask: '', marks: [] as Mark[], mark_effect: '', physical_origin_known: '' as Choice,
  note: '', original_half_inspected: false, both_windows_inspected: false, outcome_blinded: false, lateral_margins_visible: '' as Choice, unclassified_mark_present: '' as Choice, ruler_outside_bone: '' as Choice, ruler_locations_xy: [] as Point[],
  reason: '', includeObservationOnReject: false,
} }
const form = reactive(emptyForm()), baseline = ref(JSON.stringify(form))
const reviewIntent = ref(crypto.randomUUID())
const nextPoint = reactive({ x: '', y: '' })
const dirty = computed(() => editing.value && JSON.stringify(form) !== baseline.value)
const allViewsReady = computed(() => Object.values(readiness).every(Boolean))
const ready = computed(() => preparation.value?.status === 'ready' && preparation.value.isCurrent && Boolean(preparation.value.sourceReference))
const acceptanceAvailable = computed(() => ready.value && preparation.value?.sourceReference?.centerResolved && preparation.value.reviewContext?.canReview && preparation.value.reviewContext.blockerCodes.length === 0 && allViewsReady.value && !contextChanged.value)
const hasRuler = computed(() => form.marks.includes('peripheral_ruler'))
const booleanFields = [
  { key: 'side_supported' as const, label: '¿La asignación de lado está sustentada por la imagen?' }, { key: 'coverage_complete' as const, label: '¿La cobertura de la región está completa?' },
  { key: 'physical_origin_known' as const, label: '¿Se conoce el origen físico de las marcas?' }, { key: 'lateral_margins_visible' as const, label: '¿Son visibles ambos márgenes laterales?' }, { key: 'unclassified_mark_present' as const, label: '¿Hay alguna marca sin clasificar?' },
]
const visibilityFields = [ { key: 'medial_joint_borders' as const, label: 'Bordes articulares mediales' }, { key: 'lateral_joint_borders' as const, label: 'Bordes articulares laterales' }, { key: 'subchondral_bone' as const, label: 'Hueso subcondral' } ]
const markOptions: { value: Mark; label: string }[] = [ { value: 'peripheral_ruler', label: 'Regla periférica' }, { value: 'fine_line', label: 'Línea fina' }, { value: 'isolated_spot', label: 'Punto o mancha aislada' }, { value: 'hardware', label: 'Material metálico' }, { value: 'text_or_mask', label: 'Texto o máscara' } ]
let active = true, internalNavigation = false, allowContextReset = true
let loadController: AbortController | null = null
function initialize() {
  const saved = preparation.value?.activeReview, observation = saved?.observation
  const next = emptyForm()
  if (observation) {
    for (const field of booleanFields) next[field.key] = observation[field.key] ? 'yes' : 'no'
    Object.assign(next, { visibility: { ...observation.visibility }, hardware: observation.hardware, text_or_mask: observation.text_or_mask, marks: [...observation.marks], mark_effect: observation.mark_effect,
      note: observation.note, original_half_inspected: observation.original_half_inspected, both_windows_inspected: observation.both_windows_inspected, outcome_blinded: observation.outcome_blinded,
      ruler_outside_bone: observation.ruler_outside_bone === null ? '' : observation.ruler_outside_bone ? 'yes' : 'no', ruler_locations_xy: observation.ruler_locations_xy.map(point => [...point] as Point), includeObservationOnReject: true,
    })
  }
  if (saved) { next.confirmedSide = true; next.confirmedCheckIds = [...saved.confirmedCheckIds]; next.reason = saved.reason ?? '' }
  reviewIntent.value = crypto.randomUUID(); Object.assign(form, next); baseline.value = JSON.stringify(form); editing.value = !saved; contextChanged.value = false; confirmedReview.value = null
}
watch(() => `${preparation.value?.id}:${preparation.value?.preparationRevisionId}:${preparation.value?.reviewContext?.qcPolicyVersion}:${preparation.value?.activeReview?.id}`, () => {
  if (dirty.value && !allowContextReset) contextChanged.value = true
  else initialize()
  allowContextReset = false
})
async function discardChanges() { return !dirty.value || Boolean(await discardDialog.value?.ask()) }
async function reload() {
  if (busy.value || !await discardChanges()) return
  loading.value = true; error.value = ''; details.value = []; loadController?.abort(); loadController = new AbortController(); allowContextReset = true
  const generation = sessionGeneration.value
  try { await loadEvaluation(evaluationId); await loadActivePreparation(evaluationId, loadController.signal); if (active && generation === sessionGeneration.value) initialize() }
  catch (failure) { if (active && generation === sessionGeneration.value && !loadController.signal.aborted) error.value = apiMessage(failure) }
  finally { if (active && generation === sessionGeneration.value) { loading.value = false; allowContextReset = false } }
}
function startNewReview() { reviewIntent.value = crypto.randomUUID(); editing.value = true; confirmedReview.value = null; message.value = 'Vas a registrar una nueva decisión. El registro anterior se conservará en el historial.'; baseline.value = JSON.stringify(form) }
function rulerChanged() { if (!hasRuler.value) { form.ruler_locations_xy = []; form.ruler_outside_bone = '' } }
function addPoint(point?: Point) {
  if (busy.value || !editing.value || !hasRuler.value || !preparation.value?.cropGeometry || !preparation.value.nativeHalfShape) return
  const candidate: Point = point ?? [Number(nextPoint.x), Number(nextPoint.y)]
  const [height, width] = preparation.value.nativeHalfShape
  if ((!point && (!nextPoint.x.trim() || !nextPoint.y.trim())) || !candidate.every(Number.isFinite) || candidate[0] < 0 || candidate[1] < 0 || candidate[0] >= width || candidate[1] >= height) { error.value = 'El punto debe estar dentro de las dimensiones nativas de la mitad.'; return }
  form.ruler_locations_xy.push(candidate); nextPoint.x = ''; nextPoint.y = ''; error.value = ''
}
function observation(): QcObservation | null {
  if (booleanFields.some(field => !form[field.key]) || visibilityFields.some(field => !['visible', 'obscured', 'uncertain'].includes(form.visibility[field.key])) || !['absent', 'present', 'uncertain'].includes(form.hardware) || !['absent', 'present', 'uncertain'].includes(form.text_or_mask) || !['none', 'non_obscuring', 'obscuring', 'uncertain'].includes(form.mark_effect) || form.note.trim().length < 35 || !form.original_half_inspected || !form.both_windows_inspected || !form.outcome_blinded) {
    error.value = 'Completa todas las observaciones, confirma la inspección y el cegamiento, y escribe una nota de al menos 35 caracteres.'; return null
  }
  if (hasRuler.value && (!preparation.value?.cropGeometry || !form.ruler_outside_bone || form.ruler_locations_xy.length === 0)) { error.value = 'Para una regla periférica, indica su relación con el hueso y registra sus puntos en la mitad nativa. La captura requiere la geometría exacta del recorte.'; return null }
  const dimensions = preparation.value?.nativeHalfShape
  if (!dimensions || form.ruler_locations_xy.some(point => point.some(value => !Number.isFinite(value)) || point[0] < 0 || point[1] < 0 || point[0] >= dimensions[1] || point[1] >= dimensions[0])) { error.value = 'Revisa los puntos de la regla: deben estar dentro de la mitad nativa.'; return null }
  return {
    side_supported: form.side_supported === 'yes', coverage_complete: form.coverage_complete === 'yes', visibility: { ...form.visibility } as QcObservation['visibility'], hardware: form.hardware as QcObservation['hardware'], text_or_mask: form.text_or_mask as QcObservation['text_or_mask'], marks: [...form.marks], mark_effect: form.mark_effect as QcObservation['mark_effect'], physical_origin_known: form.physical_origin_known === 'yes', note: form.note.trim(), original_half_inspected: form.original_half_inspected, both_windows_inspected: form.both_windows_inspected, outcome_blinded: form.outcome_blinded, lateral_margins_visible: form.lateral_margins_visible === 'yes', unclassified_mark_present: form.unclassified_mark_present === 'yes', ruler_outside_bone: hasRuler.value ? form.ruler_outside_bone === 'yes' : null, ruler_locations_xy: hasRuler.value ? form.ruler_locations_xy.map(point => [point[0], point[1]]) : [],
  }
}
async function submit(decision: 'accepted' | 'rejected') {
  const current = preparation.value
  if (busy.value || !current || !ready.value || contextChanged.value) return
  error.value = ''; details.value = []; message.value = ''
  if (!form.confirmedSide) { error.value = 'Confirma la rodilla que estás revisando.'; return }
  if (decision === 'accepted' && !acceptanceAvailable.value) { error.value = 'La referencia, las vistas o el procedimiento QC todavía no permiten registrar una aceptación.'; return }
  if (decision === 'rejected' && (!form.reason.trim() || form.reason.trim().length > 2000)) { error.value = 'Escribe un motivo de rechazo entre 1 y 2000 caracteres.'; return }
  if (decision === 'accepted' && current.reviewContext?.checks.some(check => check.required && !form.confirmedCheckIds.includes(check.id))) { error.value = 'Confirma los controles obligatorios indicados por el procedimiento vigente.'; return }
  const raw = decision === 'accepted' || form.includeObservationOnReject ? observation() : null
  if ((decision === 'accepted' || form.includeObservationOnReject) && !raw) return
  const payload: ReviewSubmission = { preparationRevisionId: current.preparationRevisionId, confirmedSide: current.side, decision, confirmedCheckIds: [...form.confirmedCheckIds], reason: decision === 'rejected' ? form.reason.trim() : null, observation: raw }
  busy.value = true
  const generation = sessionGeneration.value
  try {
    const saved = await recordQualityReview(evaluationId, payload, reviewIntent.value)
    if (!active || generation !== sessionGeneration.value) return
    confirmedReview.value = saved; editing.value = false; baseline.value = JSON.stringify(form); message.value = 'El servidor confirmó la decisión y sus observaciones.'
    internalNavigation = true
    await router.push({ name: 'case-summary', params: { id: evaluationId } })
  } catch (failure) { if (active && generation === sessionGeneration.value) { error.value = apiMessage(failure); if (failure instanceof ApiError) details.value = failure.fields.map(field => field.message) } }
  finally { if (active && generation === sessionGeneration.value) { busy.value = false; internalNavigation = false } }
}
function beforeUnload(event: BeforeUnloadEvent) { if (dirty.value || busy.value) { event.preventDefault(); event.returnValue = '' } }
onBeforeRouteLeave(async () => internalNavigation || (!busy.value && await discardChanges()))
onMounted(() => { window.addEventListener('beforeunload', beforeUnload); void reload() })
onBeforeUnmount(() => { active = false; loadController?.abort(); window.removeEventListener('beforeunload', beforeUnload) })
</script>

<template>
  <div class="workflow-page">
    <RouterLink to="/evaluaciones" class="workflow-back">Volver a Mis evaluaciones</RouterLink>
    <header class="workflow-heading"><div><span class="section-eyebrow">REVISIÓN DE CALIDAD</span><h1 tabindex="-1">Revisión del recorte</h1><p>Registra lo que observas en la fuente y en el recorte automático. El procedimiento QC congelado decide si las observaciones permiten aceptar.</p></div><span v-if="existing" class="workflow-code">{{ existing.code }}</span></header>
    <p class="workflow-notice">Se muestran dos ventanas y vistas a tamaño nativo. No se permite desplazar, redibujar ni corregir manualmente el recorte. La predicción permanece bloqueada.</p>
    <p v-if="loading" role="status">Recuperando la preparación y su revisión persistida…</p>
    <div v-if="error" id="quality-review-error" class="workflow-error" role="alert">{{ error }}<ul v-if="details.length"><li v-for="(detail, index) in details" :key="index">{{ detail }}</li></ul></div>
    <p v-if="message" class="workflow-notice" role="status">{{ message }}</p>
    <div v-if="contextChanged" class="workflow-error" role="alert">La revisión de preparación o la política cambió mientras editabas. Tus observaciones siguen aquí, pero no se pueden aplicar a otra revisión. Recupera el estado actual para comenzar de nuevo.</div>
    <section v-if="!ready" class="workflow-card"><h2>La preparación todavía no permite revisar el recorte</h2><p>Debe existir una referencia sellada y una preparación vigente disponible.</p><RouterLink :to="{ name: 'radiograph', params: { id: evaluationId } }" class="button button-primary">Ir a la preparación</RouterLink></section>
    <template v-else-if="preparation">
      <section v-if="persistedReview && !editing" class="workflow-card"><h2>Decisión guardada en el servidor</h2><span id="quality-review-saved-status" class="workflow-state">{{ persistedReview.decision === 'accepted' ? 'Aceptado por el procedimiento QC' : 'Rechazado' }}</span><dl class="workflow-meta"><div><dt>Fecha</dt><dd>{{ new Date(persistedReview.createdAt).toLocaleString('es-PE') }}</dd></div><div><dt>Política QC</dt><dd>{{ persistedReview.qcPolicyVersion }}</dd></div><div v-if="persistedReview.reason"><dt>Motivo del rechazo</dt><dd>{{ persistedReview.reason }}</dd></div><div v-if="persistedReview.observation"><dt>Nota de inspección</dt><dd>{{ persistedReview.observation.note }}</dd></div></dl><div class="workflow-actions"><button type="button" class="button button-secondary" :disabled="busy || loading" @click="startNewReview">Registrar nueva decisión</button><RouterLink :to="{ name: 'case-summary', params: { id: evaluationId } }" class="button button-primary">Ver resumen</RouterLink></div></section>
      <section class="workflow-card"><h2>Fuente y recorte en las dos ventanas</h2><p>La referencia visual ya fue sellada antes de obtener estas vistas. Las marcas de regla son anotaciones de observación, no correcciones geométricas.</p><div v-if="!preparation.sourceReference?.centerResolved" class="workflow-notice workflow-warning">El centro de referencia no pudo resolverse. Esta preparación no puede aceptarse como adecuada.</div><div v-if="!preparation.reviewContext?.canReview || preparation.reviewContext.blockerCodes.length" class="workflow-notice workflow-warning">El procedimiento del servidor mantiene un bloqueo para aceptar esta preparación.<span v-for="blocker in preparation.reviewContext?.blockerCodes ?? []" :key="blocker"> {{ blocker }}</span></div><div class="workflow-grid" style="margin-top: 22px"><PreparationImage v-for="view in views" :key="`${preparation.preparationRevisionId}:${view.view}`" :data-view="view.view" :preparation="preparation" :view="view.view" :label="view.label" :native-shape="view.view.startsWith('crop_') ? preparation.cropGeometry?.previewShapes[view.view as 'crop_a' | 'crop_b'] : view.view.startsWith('full_') ? preparation.sourceContext?.fullNativeShape : preparation.nativeHalfShape" :coordinate-box="view.view.startsWith('crop_') ? preparation.cropGeometry?.nativeHalfBox : null" :points="view.view.startsWith('crop_') ? form.ruler_locations_xy : []" :interactive="view.view.startsWith('crop_') && hasRuler && editing && !busy && !loading && Boolean(preparation.cropGeometry)" @ready="readiness[view.view] = $event" @point="addPoint" /></div></section>
      <fieldset v-if="editing" class="workflow-card" :aria-busy="busy || loading" :disabled="busy || loading"><h2>Observaciones de calidad</h2><label class="workflow-check"><input id="qc-confirm-side" v-model="form.confirmedSide" type="checkbox" :disabled="busy" />Confirmo que reviso la rodilla {{ preparation.side === 'right' ? 'derecha' : 'izquierda' }} del paciente.</label><div class="workflow-fields"><div v-for="field in booleanFields" :key="field.key" class="workflow-field"><label :for="`qc-${field.key}`">{{ field.label }}</label><select :id="`qc-${field.key}`" v-model="form[field.key]" :disabled="busy"><option value="">Seleccionar</option><option value="yes">Sí</option><option value="no">No</option></select></div><div v-for="field in visibilityFields" :key="field.key" class="workflow-field"><label :for="`qc-${field.key}`">{{ field.label }}</label><select :id="`qc-${field.key}`" v-model="form.visibility[field.key]" :disabled="busy"><option value="">Seleccionar</option><option value="visible">Visible</option><option value="obscured">Oculto</option><option value="uncertain">Incierto</option></select></div><div class="workflow-field"><label for="qc-hardware">Material metálico</label><select id="qc-hardware" v-model="form.hardware" :disabled="busy"><option value="">Seleccionar</option><option value="absent">Ausente</option><option value="present">Presente</option><option value="uncertain">Incierto</option></select></div><div class="workflow-field"><label for="qc-text_or_mask">Texto o máscara</label><select id="qc-text_or_mask" v-model="form.text_or_mask" :disabled="busy"><option value="">Seleccionar</option><option value="absent">Ausente</option><option value="present">Presente</option><option value="uncertain">Incierto</option></select></div><div class="workflow-field"><label for="qc-mark_effect">Efecto de las marcas</label><select id="qc-mark_effect" v-model="form.mark_effect" :disabled="busy"><option value="">Seleccionar</option><option value="none">Sin marcas</option><option value="non_obscuring">No ocultan la región</option><option value="obscuring">Ocultan la región</option><option value="uncertain">Incierto</option></select></div></div>
      <fieldset :disabled="busy" style="border: 1px solid #dae4ea; border-radius: 9px; padding: 15px"><legend>Marcas observadas</legend><label v-for="mark in markOptions" :key="mark.value" class="workflow-check"><input v-model="form.marks" type="checkbox" :value="mark.value" :id="`qc-mark-${mark.value}`" @change="rulerChanged" />{{ mark.label }}</label></fieldset>
      <section v-if="hasRuler" id="ruler-point-capture"><h3>Puntos de la regla en la mitad nativa</h3><p>Haz clic sobre el recorte para registrar puntos o utiliza las coordenadas. La conversión usa exclusivamente la caja XYXY comunicada por el servidor.</p><p v-if="!preparation.cropGeometry" class="workflow-error" role="alert">El servidor no proporcionó la transformación exacta. No se permite inferir coordenadas ni aceptar con puntos aproximados.</p><div class="workflow-fields"><div class="workflow-field"><label for="qc-ruler_outside_bone">¿La regla está fuera del hueso?</label><select id="qc-ruler_outside_bone" v-model="form.ruler_outside_bone" :disabled="busy"><option value="">Seleccionar</option><option value="yes">Sí</option><option value="no">No</option></select></div><div class="workflow-field"><label for="ruler-point-x">Nuevo punto X (mitad nativa)</label><input id="ruler-point-x" v-model="nextPoint.x" type="number" step="any" :disabled="busy || !preparation.cropGeometry" /></div><div class="workflow-field"><label for="ruler-point-y">Nuevo punto Y (mitad nativa)</label><input id="ruler-point-y" v-model="nextPoint.y" type="number" step="any" :disabled="busy || !preparation.cropGeometry" /></div></div><button id="add-ruler-point" type="button" class="button button-secondary" :disabled="busy || !preparation.cropGeometry" @click="addPoint()">Añadir punto</button><ol class="workflow-points" style="margin-top: 18px"><li v-for="(point, index) in form.ruler_locations_xy" :key="index"><div class="workflow-field"><label :for="`ruler-x-${index}`">Punto {{ index + 1 }} · X</label><input :id="`ruler-x-${index}`" v-model.number="point[0]" type="number" step="any" :disabled="busy" /></div><div class="workflow-field"><label :for="`ruler-y-${index}`">Punto {{ index + 1 }} · Y</label><input :id="`ruler-y-${index}`" v-model.number="point[1]" type="number" step="any" :disabled="busy" /></div><button type="button" class="button button-secondary" :disabled="busy" :aria-label="`Quitar punto ${index + 1}`" @click="form.ruler_locations_xy.splice(index, 1)">Quitar</button></li></ol></section>
      <div class="workflow-field" style="margin-top: 22px"><label for="qc-note">Nota de inspección QC</label><textarea id="qc-note" v-model="form.note" maxlength="2000" :disabled="busy" aria-describedby="qc-note-hint" /><small id="qc-note-hint">Describe los hallazgos de las vistas y las marcas. Mínimo 35 caracteres; no consultes el resultado futuro.</small></div><label class="workflow-check"><input id="qc-original_half_inspected" v-model="form.original_half_inspected" type="checkbox" :disabled="busy" />He inspeccionado el original bilateral y la mitad.</label><label class="workflow-check"><input id="qc-both_windows_inspected" v-model="form.both_windows_inspected" type="checkbox" :disabled="busy" />He inspeccionado ambas ventanas, incluido el recorte.</label><label class="workflow-check"><input id="qc-outcome_blinded" v-model="form.outcome_blinded" type="checkbox" :disabled="busy" />La inspección se realizó sin consultar el resultado futuro.</label><label v-for="check in preparation.reviewContext?.checks ?? []" :key="check.id" class="workflow-check"><input v-model="form.confirmedCheckIds" type="checkbox" :value="check.id" :disabled="busy" />{{ check.label }}{{ check.required ? ' (obligatorio)' : '' }}</label>
      <div class="workflow-notice">Registrar una aceptación envía estas observaciones completas al procedimiento QC congelado. La solicitud puede ser rechazada aunque hayas seleccionado todos los controles.</div><div class="workflow-actions"><button id="submit-qc-accepted" type="button" class="button button-primary" :disabled="busy || !acceptanceAvailable" @click="submit('accepted')">Registrar aceptación</button></div>
      <h3>Registrar rechazo</h3><div class="workflow-field"><label for="qc-rejection-reason">Motivo del rechazo</label><textarea id="qc-rejection-reason" v-model="form.reason" maxlength="2000" :disabled="busy" /></div><label class="workflow-check"><input id="qc-include-observation" v-model="form.includeObservationOnReject" type="checkbox" :disabled="busy" />Incluir también las observaciones QC completas.</label><p class="workflow-notice">Puedes registrar solo el motivo del rechazo o adjuntar las observaciones de calidad completas.</p><div class="workflow-actions"><button id="submit-qc-rejected" type="button" class="button button-secondary" :disabled="busy || contextChanged" @click="submit('rejected')">Registrar rechazo</button></div>
      </fieldset>
    </template>
    <div class="workflow-actions"><button type="button" class="button button-secondary" :disabled="busy" @click="reload">Recuperar estado actual</button><RouterLink :to="{ name: 'radiograph', params: { id: evaluationId } }" class="button button-secondary">Volver a la imagen</RouterLink></div>
    <p v-if="dirty" class="workflow-notice" role="status">Tienes observaciones sin guardar. Se conservan en esta pantalla mientras resuelves cualquier error.</p>
    <UnsavedChangesDialog ref="discardDialog" />
  </div>
</template>
