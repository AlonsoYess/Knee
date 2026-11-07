<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import PreparationImage from '../components/PreparationImage.vue'
import { evaluations, loadEvaluation } from '../state/evaluations'
import { getActivePreparation, loadActivePreparation } from '../state/preparations'
import { apiMessage, sessionGeneration } from '../state/session'
import type { PreparationView } from '../api/preparationContracts'

const route = useRoute(), evaluationId = typeof route.params.id === 'string' ? route.params.id : ''
const existing = computed(() => evaluations.value.find(item => item.id === evaluationId))
const preparation = computed(() => getActivePreparation(evaluationId))
const review = computed(() => preparation.value?.activeReview ?? null)
const loading = ref(true), error = ref(''), verified = ref(false)
let active = true, controller: AbortController | null = null
const views: { view: PreparationView; label: string }[] = [
  { view: 'full_a', label: 'Original bilateral — ventana 1–99' }, { view: 'half_a', label: 'Mitad seleccionada — ventana 1–99' }, { view: 'crop_a', label: 'Recorte automático — ventana 1–99' },
  { view: 'full_b', label: 'Original bilateral — ventana 5–95' }, { view: 'half_b', label: 'Mitad seleccionada — ventana 5–95' }, { view: 'crop_b', label: 'Recorte automático — ventana 5–95' },
]
const visibleViews = computed(() => views.filter(view => preparation.value?.sourceContext && (!view.view.startsWith('crop_') || (preparation.value?.status === 'ready' && preparation.value.sourceReference))))
const statusLabel = computed(() => !preparation.value ? 'Sin imagen preparada' : ({ queued: 'Preparación en cola', processing: 'Procesamiento en curso', awaiting_reference: 'Referencia visual pendiente', ready: 'Preparación disponible', failed: 'Preparación fallida' }[preparation.value.status]))
const reviewLabel = computed(() => !review.value ? 'Sin decisión vigente' : review.value.decision === 'accepted' ? 'Aceptado por el procedimiento QC' : review.value.assessment ? 'Rechazado con observaciones QC' : 'Rechazo administrativo')
const date = new Intl.DateTimeFormat('es-PE', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'America/Lima' })
function formatDate(value: string) { return date.format(new Date(value)) }
async function reload() {
  loading.value = true; error.value = ''; verified.value = false; controller?.abort(); controller = new AbortController()
  const generation = sessionGeneration.value
  try { await loadEvaluation(evaluationId); await loadActivePreparation(evaluationId, controller.signal); if (active && generation === sessionGeneration.value) verified.value = true }
  catch (failure) { if (active && generation === sessionGeneration.value && !controller.signal.aborted) error.value = apiMessage(failure) }
  finally { if (active && generation === sessionGeneration.value) loading.value = false }
}
onMounted(() => { void reload() })
onBeforeUnmount(() => { active = false; controller?.abort() })
</script>

<template>
  <div class="workflow-page">
    <RouterLink to="/evaluaciones" class="workflow-back">Volver a Mis evaluaciones</RouterLink>
    <header class="workflow-heading"><div><span class="section-eyebrow">CONSULTAR EL CASO</span><h1 tabindex="-1">Resumen del caso</h1><p>Datos basales, preparación y decisión recuperados del servidor. La calidad registrada no sustituye la validación clínica del prototipo.</p></div><span v-if="existing" class="workflow-code">{{ existing.code }}</span></header>
    <section v-if="existing && verified" id="summary-prediction-block" class="workflow-card"><h2>Predicción bloqueada</h2><p>No se calcula progresión, probabilidad ni clasificación. Los modelos predictivos siguen sin conectar.</p><div class="workflow-actions"><button type="button" class="button button-primary" disabled>Calcular predicción</button></div></section>
    <p v-if="loading" role="status">Recuperando los registros persistidos…</p>
    <div v-if="error" id="summary-error" class="workflow-error" role="alert">{{ error }}<p>No se sustituye el registro del servidor por una simulación local.</p></div>
    <section v-if="!existing && !loading" class="workflow-card"><h2>Resumen no disponible</h2><p>No pudimos recuperar este caso de tu cuenta. Vuelve al listado para elegir un caso disponible.</p></section>
    <template v-if="existing && verified">
      <section class="workflow-card"><h2>Datos basales guardados</h2><dl class="workflow-meta summary-data-grid"><div><dt>Evaluación</dt><dd>{{ existing.code }}</dd></div><div><dt>Rodilla</dt><dd>{{ existing.knee === 'right' ? 'Derecha del paciente' : existing.knee === 'left' ? 'Izquierda del paciente' : 'Sin indicar' }}</dd></div><div><dt>Edad basal</dt><dd>{{ existing.age == null ? 'Ausente' : `${existing.age} años` }}</dd></div><div><dt>Sexo registrado</dt><dd>{{ existing.sex === 'F' ? 'Femenino' : existing.sex === 'M' ? 'Masculino' : 'Ausente' }}</dd></div><div><dt>IMC basal</dt><dd>{{ existing.bmi == null ? 'Ausente' : `${existing.bmi} kg/m²` }}</dd></div><div><dt>Último guardado basal</dt><dd>{{ formatDate(existing.updatedAt) }} · Lima</dd></div></dl><RouterLink :to="{ name: 'edit-evaluation', params: { id: evaluationId } }" class="text-button">Revisar datos basales</RouterLink></section>
      <section class="workflow-card"><h2>Preparación de la imagen</h2><span id="summary-preparation-status" class="workflow-state">{{ statusLabel }}</span><div v-if="preparation"><p style="margin-top: 16px">{{ preparation.sourceReference ? 'Referencia visual sellada antes de obtener el recorte.' : 'Referencia visual pendiente de sellar.' }}</p><p v-if="preparation.error" class="workflow-error" role="alert">{{ preparation.error.message }}</p><div v-if="preparation.sourceReference" class="workflow-notice"><strong>Referencia inmutable:</strong> {{ preparation.sourceReference.centerResolved ? `centro ${preparation.sourceReference.centerXY?.[0]}, ${preparation.sourceReference.centerXY?.[1]} px de la mitad nativa` : 'centro no resuelto' }}. {{ preparation.sourceReference.note }}</div><details class="workflow-trace"><summary>Ver trazabilidad</summary><dl class="workflow-meta"><div><dt>Identidad de imagen</dt><dd>{{ preparation.image.imageId }}</dd></div><div><dt>Versión de la preparación</dt><dd>{{ preparation.preparationRevisionId }}</dd></div><div><dt>Versión del flujo compartido</dt><dd>{{ preparation.pipelineVersion ?? 'Pendiente' }}</dd></div></dl></details></div><RouterLink :to="{ name: 'radiograph', params: { id: evaluationId } }" class="text-button">{{ preparation ? 'Consultar preparación' : 'Cargar DICOM y preparar' }}</RouterLink></section>
      <section class="workflow-card"><h2>Decisión persistida</h2><span id="summary-review-status" class="workflow-state">{{ reviewLabel }}</span><div v-if="review" class="summary-review-details"><dl class="workflow-meta"><div><dt>Fecha y hora</dt><dd>{{ formatDate(review.createdAt) }} · Lima</dd></div><div v-if="review.reason"><dt>Motivo del rechazo</dt><dd>{{ review.reason }}</dd></div><div v-if="review.observation"><dt>Nota de inspección</dt><dd>{{ review.observation.note }}</dd></div></dl><details class="workflow-trace"><summary>Ver trazabilidad</summary><dl class="workflow-meta"><div><dt>Política QC</dt><dd>{{ review.qcPolicyVersion }}</dd></div><div><dt>Identificador de decisión</dt><dd>{{ review.id }}</dd></div><div><dt>Revisor</dt><dd>{{ review.reviewerId }}</dd></div></dl></details></div><p v-if="review && !review.observation" class="workflow-notice">Decisión de rechazo registrada con el motivo del revisor.</p><p v-if="review?.assessment?.reasons.length">Motivos del procedimiento: {{ review.assessment.reasons.join(' · ') }}</p><div class="workflow-actions"><RouterLink v-if="preparation?.status === 'ready'" :to="{ name: 'crop-review', params: { id: evaluationId } }" class="button button-secondary">{{ review ? 'Consultar revisión' : 'Registrar revisión' }}</RouterLink></div></section>
      <section v-if="preparation && visibleViews.length" class="workflow-card"><h2>Vistas autenticadas de esta preparación</h2><p>Las vistas corresponden a la revisión indicada. No se ofrecen herramientas de corrección manual.</p><div class="workflow-grid" style="margin-top: 20px"><PreparationImage v-for="view in visibleViews" :key="`${preparation.preparationRevisionId}:${view.view}`" :data-view="view.view" :preparation="preparation" :view="view.view" :label="view.label" :native-shape="view.view.startsWith('crop_') ? preparation.cropGeometry?.previewShapes[view.view as 'crop_a' | 'crop_b'] : view.view.startsWith('full_') ? preparation.sourceContext?.fullNativeShape : preparation.nativeHalfShape" :coordinate-box="view.view.startsWith('crop_') ? preparation.cropGeometry?.nativeHalfBox : null" :points="view.view.startsWith('crop_') ? review?.observation?.ruler_locations_xy ?? [] : []" /></div></section>
    </template>
    <div class="workflow-actions"><button type="button" class="button button-secondary" :disabled="loading" @click="reload">Actualizar resumen</button><RouterLink to="/evaluaciones" class="button button-secondary">Ir a Mis evaluaciones</RouterLink></div>
  </div>
</template>
