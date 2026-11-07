<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { onBeforeRouteLeave, useRoute, useRouter } from 'vue-router'
import PreparationImage from '../components/PreparationImage.vue'
import { ApiError } from '../api/http'
import UnsavedChangesDialog from '../components/UnsavedChangesDialog.vue'
import { evaluations, loadEvaluation } from '../state/evaluations'
import { getActivePreparation, loadActivePreparation, retryPreparation, sealSourceReference, uploadPreparation } from '../state/preparations'
import { apiMessage, sessionGeneration } from '../state/session'
import type { Point, ReferenceDto, SourceView } from '../api/preparationContracts'

const route = useRoute(), router = useRouter()
const evaluationId = typeof route.params.id === 'string' ? route.params.id : ''
const existing = computed(() => evaluations.value.find(item => item.id === evaluationId))
const preparation = computed(() => getActivePreparation(evaluationId))
const selectedFile = ref<File | null>(null), busy = ref(false), loading = ref(true), error = ref(''), message = ref('')
const discardDialog = ref<InstanceType<typeof UnsavedChangesDialog> | null>(null)
const confirmedReference = ref<ReferenceDto | null>(null)
const sealedReference = computed(() => preparation.value?.sourceReference ?? confirmedReference.value)
const sourceReady = reactive<Record<SourceView, boolean>>({ full_a: false, full_b: false, half_a: false, half_b: false })
const reference = reactive({ confirmedSide: false, centerResolved: '' as '' | 'resolved' | 'unresolved', x: '', y: '', inspected: false, blinded: false, note: '' })
const referenceBaseline = ref(JSON.stringify(reference))
const uploadIntent = ref(crypto.randomUUID())
const referenceIntent = ref(crypto.randomUUID()), retryIntent = ref<string | null>(null)
const dirty = computed(() => Boolean(selectedFile.value || (!sealedReference.value && JSON.stringify(reference) !== referenceBaseline.value)))
const sideLabel = computed(() => existing.value?.knee === 'right' ? 'Derecha del paciente' : existing.value?.knee === 'left' ? 'Izquierda del paciente' : 'Sin indicar')
const sourceViews: { view: SourceView; label: string }[] = [
  { view: 'full_a', label: 'Original bilateral — ventana 1–99' }, { view: 'half_a', label: 'Mitad seleccionada — ventana 1–99' },
  { view: 'full_b', label: 'Original bilateral — ventana 5–95' }, { view: 'half_b', label: 'Mitad seleccionada — ventana 5–95' },
]
const allSourceReady = computed(() => Object.values(sourceReady).every(Boolean))
const center = computed<Point[]>(() => reference.centerResolved === 'resolved' && reference.x !== '' && reference.y !== '' && Number.isInteger(Number(reference.x)) && Number.isInteger(Number(reference.y)) ? [[Number(reference.x), Number(reference.y)]] : [])
const statusLabel = computed(() => preparation.value ? ({ queued: 'Preparación en cola', processing: 'Procesando la imagen', awaiting_reference: sealedReference.value ? sealedReference.value.centerResolved ? 'Referencia sellada; recuperando estado' : 'Referencia sellada sin centro resuelto' : 'Referencia visual pendiente', ready: 'Preparación disponible', failed: 'Preparación interrumpida o fallida' }[preparation.value.status]) : 'Sin imagen cargada')
let pollTimer: ReturnType<typeof setTimeout> | null = null
let pollController: AbortController | null = null
let active = true, internalNavigation = false
function stopPolling() { if (pollTimer) clearTimeout(pollTimer); pollTimer = null; pollController?.abort(); pollController = null }
function schedulePoll() {
  if (!active || busy.value || !preparation.value || preparation.value.status === 'ready' || preparation.value.status === 'failed' || (preparation.value.status === 'awaiting_reference' && !sealedReference.value?.centerResolved)) return
  pollTimer = setTimeout(() => { void poll() }, 1200)
}
async function poll() {
  if (!active || busy.value) return
  const generation = sessionGeneration.value
  pollController = new AbortController()
  try { await loadActivePreparation(evaluationId, pollController.signal); if (generation === sessionGeneration.value) schedulePoll() }
  catch (failure) { if (active && generation === sessionGeneration.value && !pollController?.signal.aborted) error.value = apiMessage(failure) }
}
async function discardChanges() { return !dirty.value || Boolean(await discardDialog.value?.ask()) }
async function reload() {
  if (busy.value || !await discardChanges()) return
  stopPolling(); loading.value = true; error.value = ''
  const generation = sessionGeneration.value
  try { await loadEvaluation(evaluationId); await loadActivePreparation(evaluationId); if (active && generation === sessionGeneration.value) { selectedFile.value = null; schedulePoll() } }
  catch (failure) { if (active && generation === sessionGeneration.value) error.value = apiMessage(failure) }
  finally { if (active && generation === sessionGeneration.value) loading.value = false }
}
function chooseFile(event: Event) { selectedFile.value = (event.target as HTMLInputElement).files?.[0] ?? null; uploadIntent.value = crypto.randomUUID(); error.value = ''; message.value = '' }
async function upload() {
  if (busy.value || !selectedFile.value || !existing.value?.knee) return
  stopPolling(); busy.value = true; error.value = ''; message.value = ''
  const generation = sessionGeneration.value
  try {
    await uploadPreparation(evaluationId, selectedFile.value, uploadIntent.value)
    if (!active || generation !== sessionGeneration.value) return
    selectedFile.value = null; confirmedReference.value = null
    message.value = 'El servidor confirmó la carga. La preparación todavía no acredita calidad ni permite una predicción.'
  } catch (failure) { if (active && generation === sessionGeneration.value) error.value = apiMessage(failure) }
  finally { if (active && generation === sessionGeneration.value) { busy.value = false; schedulePoll() } }
}
function markCenter(point: Point) {
  if (busy.value || sealedReference.value || !preparation.value?.nativeHalfShape) return
  const [height, width] = preparation.value.nativeHalfShape
  if (point[0] < 0 || point[0] > width || point[1] < 0 || point[1] > height) return
  reference.centerResolved = 'resolved'; reference.x = String(Math.min(width - 1, Math.floor(point[0]))); reference.y = String(Math.min(height - 1, Math.floor(point[1])))
}
async function seal() {
  const current = preparation.value
  if (busy.value || !current || current.status !== 'awaiting_reference' || sealedReference.value) return
  error.value = ''; message.value = ''
  if (!allSourceReady.value || !reference.confirmedSide || !reference.inspected || !reference.blinded || !reference.centerResolved || reference.note.trim().length < 35) {
    error.value = 'Revisa las cuatro vistas, confirma el lado y las declaraciones, indica si el centro está resuelto y escribe una nota de al menos 35 caracteres.'; return
  }
  const point = reference.centerResolved === 'resolved' ? center.value[0] : null
  if (reference.centerResolved === 'resolved' && (!point || !current.nativeHalfShape || point[0] < 0 || point[1] < 0 || point[0] >= current.nativeHalfShape[1] || point[1] >= current.nativeHalfShape[0])) { error.value = 'Indica un centro dentro de las dimensiones nativas de la mitad.'; return }
  stopPolling(); busy.value = true
  const generation = sessionGeneration.value
  try {
    confirmedReference.value = await sealSourceReference(evaluationId, { preparationRevisionId: current.preparationRevisionId, confirmedSide: current.side, centerResolved: reference.centerResolved === 'resolved', centerXY: point ?? null, originalAndHalfInspected: reference.inspected, outcomeBlinded: reference.blinded, note: reference.note.trim() }, referenceIntent.value)
    if (!active || generation !== sessionGeneration.value) return
    referenceBaseline.value = JSON.stringify(reference)
    message.value = 'Referencia visual sellada en el servidor. No podrá editarse después de mostrar el recorte.'
    await loadActivePreparation(evaluationId)
  } catch (failure) { if (active && generation === sessionGeneration.value) error.value = apiMessage(failure) }
  finally { if (active && generation === sessionGeneration.value) { busy.value = false; schedulePoll() } }
}
async function retry() {
  if (busy.value) return
  busy.value = true; error.value = ''
  const generation = sessionGeneration.value
  try {
    retryIntent.value ??= crypto.randomUUID()
    const result = await retryPreparation(evaluationId, retryIntent.value)
    if (active && generation === sessionGeneration.value) {
      retryIntent.value = null
      message.value = result.status === 'failed' ? 'Se recuperó el intento anterior, que terminó fallido. Puedes iniciar un nuevo reintento explícito.' : 'El servidor confirmó el intento de preparación.'
    }
  } catch (failure) {
    if (active && generation === sessionGeneration.value) {
      error.value = apiMessage(failure)
      if (failure instanceof ApiError && failure.status === 503) {
        try { if ((await loadActivePreparation(evaluationId))?.status === 'failed') retryIntent.value = null } catch { /* Keep the same key until failure is confirmed. */ }
      }
    }
  }
  finally { if (active && generation === sessionGeneration.value) { busy.value = false; schedulePoll() } }
}
async function continueToReview() { if (preparation.value?.status === 'ready' && !dirty.value && !busy.value) { internalNavigation = true; try { await router.push({ name: 'crop-review', params: { id: evaluationId } }) } finally { internalNavigation = false } } }
watch(() => JSON.stringify([preparation.value?.id, preparation.value?.preparationRevisionId, preparation.value?.sourceReference?.id]), () => {
  for (const view of Object.keys(sourceReady) as SourceView[]) sourceReady[view] = false
  if (!preparation.value?.sourceReference && !confirmedReference.value) {
    Object.assign(reference, { confirmedSide: false, centerResolved: '', x: '', y: '', inspected: false, blinded: false, note: '' }); referenceBaseline.value = JSON.stringify(reference); referenceIntent.value = crypto.randomUUID()
  }
})
function beforeUnload(event: BeforeUnloadEvent) { if (dirty.value || busy.value) { event.preventDefault(); event.returnValue = '' } }
onBeforeRouteLeave(async () => internalNavigation || (!busy.value && await discardChanges()))
onMounted(() => { window.addEventListener('beforeunload', beforeUnload); void reload() })
onBeforeUnmount(() => { active = false; stopPolling(); window.removeEventListener('beforeunload', beforeUnload) })
</script>

<template>
  <div class="workflow-page">
    <RouterLink to="/evaluaciones" class="workflow-back">Volver a Mis evaluaciones</RouterLink>
    <header class="workflow-heading"><div><span class="section-eyebrow">PREPARAR LA IMAGEN</span><h1 tabindex="-1">Radiografía y lado</h1><p>Carga un DICOM, revisa la fuente y sella su referencia visual antes de que se muestre el recorte.</p></div><span v-if="existing" class="workflow-code">{{ existing.code }}</span></header>
    <p class="workflow-notice">La imagen, la referencia y la preparación se conservan en el servidor de esta aplicación. La predicción sigue bloqueada. No se permite corregir manualmente la geometría del recorte.</p>
    <p v-if="loading" role="status">Recuperando la preparación guardada…</p>
    <div v-if="error" id="preparation-error" class="workflow-error" role="alert">{{ error }}</div>
    <p v-if="message" class="workflow-notice" role="status">{{ message }}</p>
    <section v-if="!existing" class="workflow-card"><h2>Caso no disponible</h2><p>Recupera el listado para elegir un caso de tu cuenta.</p></section>
    <template v-else>
      <section class="workflow-card" :aria-busy="busy"><h2>Archivo y lado basal</h2><p><strong>Rodilla guardada: {{ sideLabel }}</strong></p><RouterLink :to="{ name: 'edit-evaluation', params: { id: evaluationId } }" class="text-button">Corregir los datos basales</RouterLink><div v-if="!existing.knee" class="workflow-notice workflow-warning">Guarda el lado antes de cargar el DICOM. Los datos clínicos ausentes pueden permanecer vacíos.</div><div class="workflow-field" style="margin-top: 18px"><label for="dicom-file">DICOM basal</label><input id="dicom-file" type="file" accept=".dcm,application/dicom,application/octet-stream" :disabled="busy || loading || !existing.knee" @change="chooseFile" /><small>El servidor verificará el contenido. El nombre o la extensión no acreditan que sea un DICOM válido.</small></div><p v-if="selectedFile">Archivo pendiente: {{ selectedFile.name }} · {{ (selectedFile.size / 1024 / 1024).toFixed(2) }} MB</p><p v-if="selectedFile && preparation" class="workflow-notice workflow-warning">Sustituir la imagen dejará sin vigencia la revisión activa anterior. El historial se conserva.</p><div class="workflow-actions"><button v-if="selectedFile" type="button" class="button button-secondary" :disabled="busy" @click="selectedFile = null">Descartar archivo pendiente</button><button type="button" class="button button-primary" :disabled="busy || loading || !selectedFile || !existing.knee" @click="upload">{{ busy ? 'Procesando solicitud…' : preparation ? 'Sustituir DICOM' : 'Cargar DICOM' }}</button></div></section>
      <section v-if="preparation && !selectedFile" class="workflow-card"><h2>Estado de la preparación</h2><span class="workflow-state" role="status" aria-live="polite">{{ statusLabel }}</span><p v-if="preparation.error" class="workflow-error" role="alert">{{ preparation.error.message }}</p><p v-if="!preparation.isCurrent" class="workflow-error">Esta preparación es histórica y no permite registrar una decisión vigente.</p><div class="workflow-actions"><button type="button" class="button button-secondary" :disabled="busy" @click="reload">Actualizar estado</button><button v-if="preparation.status === 'failed' && preparation.isCurrent" type="button" class="button button-secondary" :disabled="busy" @click="retry">Reintentar preparación</button></div></section>
      <template v-if="preparation?.sourceContext && !selectedFile">
        <section class="workflow-card"><h2>{{ sealedReference ? 'Fuente de la referencia sellada' : 'Inspección de la fuente antes del recorte' }}</h2><p>Revisa el original bilateral y la mitad seleccionada en las ventanas 1–99 y 5–95. El centro se registra sobre la mitad nativa; esta referencia no modifica ni guía el recorte automático.</p><div class="workflow-grid" style="margin-top: 20px"><PreparationImage v-for="view in sourceViews" :key="`${preparation.preparationRevisionId}:${view.view}`" :preparation="preparation" :view="view.view" :data-view="view.view" :label="view.label" :native-shape="view.view.startsWith('half_') ? preparation.nativeHalfShape : preparation.sourceContext.fullNativeShape" :points="view.view.startsWith('half_') ? sealedReference?.centerXY ? [sealedReference.centerXY] : center : []" :interactive="view.view.startsWith('half_') && preparation.status === 'awaiting_reference' && !sealedReference && !busy && !loading" @ready="sourceReady[view.view] = $event" @point="markCenter" /></div></section>
        <fieldset v-if="preparation.status === 'awaiting_reference' && !sealedReference" class="workflow-card" :aria-busy="busy || loading" :disabled="busy || loading"><h2>Referencia visual definitiva</h2><p>Debes sellarla antes de acceder al recorte. No habrá edición posterior del centro.</p><label class="workflow-check"><input id="reference-confirm-side" v-model="reference.confirmedSide" type="checkbox" :disabled="busy" />Confirmo que la mitad corresponde a {{ sideLabel.toLocaleLowerCase('es') }}.</label><div class="workflow-fields"><div class="workflow-field"><label for="reference-resolved">¿Pudiste determinar el centro?</label><select id="reference-resolved" v-model="reference.centerResolved" :disabled="busy"><option value="">Seleccionar</option><option value="resolved">Sí, centro resuelto</option><option value="unresolved">No, centro no resuelto</option></select></div><div class="workflow-field"><span>Dimensiones nativas de la mitad</span><p>{{ preparation.nativeHalfShape?.[1] }} × {{ preparation.nativeHalfShape?.[0] }} px</p></div><div class="workflow-field"><label for="reference-center-x">Centro X en la mitad nativa (px)</label><input id="reference-center-x" v-model="reference.x" type="number" step="1" min="0" :max="(preparation.nativeHalfShape?.[1] ?? 1) - 1" :disabled="busy || reference.centerResolved !== 'resolved'" /></div><div class="workflow-field"><label for="reference-center-y">Centro Y en la mitad nativa (px)</label><input id="reference-center-y" v-model="reference.y" type="number" step="1" min="0" :max="(preparation.nativeHalfShape?.[0] ?? 1) - 1" :disabled="busy || reference.centerResolved !== 'resolved'" /></div></div><label class="workflow-check"><input id="reference-inspected" v-model="reference.inspected" type="checkbox" :disabled="busy" />He inspeccionado el original y la mitad en las dos ventanas.</label><label class="workflow-check"><input id="reference-blinded" v-model="reference.blinded" type="checkbox" :disabled="busy" />Realicé esta inspección sin consultar el resultado futuro.</label><div class="workflow-field"><label for="reference-note">Nota de referencia visual</label><textarea id="reference-note" v-model="reference.note" :disabled="busy" maxlength="2000" aria-describedby="reference-note-hint" /><small id="reference-note-hint">Describe la inspección y cómo determinaste el centro, o por qué no pudo resolverse. Mínimo 35 caracteres.</small></div><p v-if="reference.centerResolved === 'unresolved'" class="workflow-notice workflow-warning">Una referencia no resuelta no permitirá aceptar la preparación como adecuada.</p><div class="workflow-actions"><button id="seal-reference" type="button" class="button button-primary" :disabled="busy || !allSourceReady" @click="seal">Sellar referencia visual</button></div></fieldset>
        <section v-else-if="sealedReference" class="workflow-card"><h2>Referencia sellada</h2><dl class="workflow-meta"><div><dt>Centro</dt><dd>{{ sealedReference.centerResolved ? `${sealedReference.centerXY?.[0]}, ${sealedReference.centerXY?.[1]} px de la mitad nativa` : 'No resuelto' }}</dd></div><div><dt>Nota persistida</dt><dd>{{ sealedReference.note }}</dd></div><div><dt>Fecha</dt><dd>{{ new Date(sealedReference.createdAt).toLocaleString('es-PE') }}</dd></div><div><dt>Referencia inmutable</dt><dd>{{ sealedReference.id }}</dd></div></dl><p>El centro ya no puede editarse. Las coordenadas no se usan como entrada del localizador.</p></section>
      </template>
      <div class="workflow-actions"><RouterLink to="/evaluaciones" class="button button-secondary">Volver a evaluaciones</RouterLink><button id="continue-quality-review" type="button" class="button button-primary" :disabled="busy || loading || dirty || preparation?.status !== 'ready' || !preparation?.isCurrent" @click="continueToReview">Revisar el recorte</button></div>
    </template>
    <UnsavedChangesDialog ref="discardDialog" />
  </div>
</template>
