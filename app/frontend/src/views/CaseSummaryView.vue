<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import DemoCropPreview from '../components/DemoCropPreview.vue'
import { closeOnBackdrop } from '../dialog'
import { evaluations, isEditableDemoEvaluation, getDemoSummaryState } from '../state/evaluations'
import { demoImages } from '../demo/demoImages'
import { getDemoCrop, demoRejectReasons } from '../demo/demoCrops'

const route = useRoute()
const requestedId = typeof route.params.id === 'string' ? route.params.id : ''
const existing = computed(() => evaluations.value.find(item => item.id === requestedId))
const state = computed(() => existing.value ? getDemoSummaryState(existing.value) : null)
const editable = computed(() => isEditableDemoEvaluation(existing.value))
const image = computed(() => demoImages.find(item => item.id === existing.value?.image?.sampleId))
const crop = computed(() => state.value?.imageComplete ? getDemoCrop(existing.value?.image?.sampleId ?? '', existing.value?.knee ?? null) : undefined)
const enlarged = ref<HTMLDialogElement | null>(null)
const enlargedView = ref<'original' | 'crop'>('original')
const imageLoaded = ref(false)
const cropLoaded = ref(false)
const imageFailed = ref(false)
const cropFailed = ref(false)
const previewFailed = computed(() => imageFailed.value || cropFailed.value)
const sideLabel = computed(() => existing.value?.knee === 'right' ? 'Derecha del paciente' : existing.value?.knee === 'left' ? 'Izquierda del paciente' : 'Pendiente')
const sexLabel = computed(() => existing.value?.sex === 'F' ? 'Femenino' : existing.value?.sex === 'M' ? 'Masculino' : 'Pendiente')
const reviewLabel = computed(() => ({
  missing: 'Sin revisión asociada', pending: 'Revisión pendiente', accepted: 'Aceptado en demostración',
  rejected: 'Rechazado en demostración', outdated: 'Revisión no vigente',
}[state.value?.reviewState ?? 'missing']))
const reviewDescription = computed(() => ({
  missing: 'Todavía no se ha conservado una decisión para la imagen y el lado actuales.',
  pending: 'El ejemplo permanece pendiente de una decisión explícita.',
  accepted: 'La decisión corresponde a esta imagen, este lado y esta versión del recorte de ejemplo.',
  rejected: 'El recorte de ejemplo fue rechazado. Revisa el motivo antes de continuar con la preparación.',
  outdated: 'La revisión conservada no corresponde a la preparación actual. Se requiere una nueva revisión del ejemplo.',
}[state.value?.reviewState ?? 'missing']))
const reasonLabel = computed(() => demoRejectReasons.find(reason => reason.id === existing.value?.review?.reason)?.label ?? 'Motivo no disponible')
const steps = computed(() => [
  { label: 'Datos basales', done: Boolean(state.value?.baselinesComplete) },
  { label: 'Radiografía y lado', done: Boolean(state.value?.imageComplete) },
  { label: 'Revisión del recorte', done: state.value?.reviewState === 'accepted' },
  { label: 'Resumen', done: false },
])
const decimal = new Intl.NumberFormat('es-PE', { minimumFractionDigits: 1, maximumFractionDigits: 20 })
const date = new Intl.DateTimeFormat('es-PE', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'America/Lima' })
function formatDate(value?: string | null): string {
  if (!value || !Number.isFinite(new Date(value).getTime())) return 'Pendiente'
  return date.format(new Date(value))
}
function enlarge(view: 'original' | 'crop') { enlargedView.value = view; enlarged.value?.showModal() }
const editRoute = computed(() => ({ name: 'edit-evaluation', params: { id: requestedId } }))
const imageRoute = computed(() => ({ name: 'radiograph', params: { id: requestedId } }))
const reviewRoute = computed(() => ({ name: 'crop-review', params: { id: requestedId } }))
const nextStep = computed(() => {
  if (!editable.value || !state.value || state.value.preparationComplete) return null
  if (!state.value.baselinesComplete) return { label: 'Completar datos basales', route: editRoute.value }
  if (!state.value.imageComplete) return { label: 'Preparar imagen y lado', route: imageRoute.value }
  return { label: state.value.reviewState === 'rejected' ? 'Revisar recorte rechazado' : 'Completar revisión del recorte', route: reviewRoute.value }
})
</script>

<template>
  <div class="summary-page">
    <p class="simulation-notice" role="status">Las imágenes y la revisión siguen siendo una simulación; sus cambios no se guardan en el servidor. Se reinician al recargar, cerrar sesión o cambiar la revisión basal.</p>
    <RouterLink to="/evaluaciones" class="summary-back"><AppIcon name="chevron-left" :size="16" />Volver a Mis evaluaciones</RouterLink>
    <section v-if="!existing" class="summary-unavailable"><span class="summary-symbol"><AppIcon name="file" :size="25" /></span><h1 tabindex="-1">Resumen no disponible</h1><p>No pudimos recuperar este caso del servidor o no tienes acceso. Vuelve al listado para elegir un caso disponible.</p><RouterLink to="/evaluaciones" class="button button-primary">Volver al listado</RouterLink></section>
    <template v-else>
      <header class="summary-heading"><div><span class="section-eyebrow">CONSULTAR EL CASO</span><h1 tabindex="-1">Resumen del caso</h1><p>Los datos basales están guardados en el servidor. Las imágenes y la revisión de ejemplo solo pertenecen a este recorrido temporal.</p></div><div class="summary-header-meta"><span class="summary-code"><AppIcon name="file" :size="15" />{{ existing.code }}</span><span>Actualización: {{ formatDate(existing.updatedAt) }} · Lima</span></div></header>
      <ol class="summary-stepper" aria-label="Recorrido de evaluación"><li v-for="(step,index) in steps" :key="step.label" :class="{ 'summary-step-current': index === 3, 'summary-step-done': step.done }" :aria-current="index === 3 ? 'step' : undefined"><span class="summary-step-number"><AppIcon v-if="step.done" name="check" :size="14" /><template v-else>{{ index + 1 }}</template></span><span>{{ step.label }}</span></li></ol>

      <div class="summary-scope"><AppIcon name="shield" :size="19" /><p><strong>Resumen de demostración.</strong> Los datos, las imágenes y las decisiones son ficticios.</p></div>

      <section id="summary-prediction-block" class="summary-prediction" aria-labelledby="summary-prediction-title">
        <div class="summary-prediction-main"><span class="summary-prediction-icon"><AppIcon name="lock" :size="25" /></span><div><span class="summary-prediction-eyebrow">PROGRESIÓN ESTRUCTURAL A 48 MESES</span><h2 id="summary-prediction-title">Predicción bloqueada</h2><p class="summary-model-state">Modelos sin conectar</p></div></div>
        <div class="summary-prediction-explanation"><p id="summary-prediction-reason">Esta demostración permite revisar el caso. No calcula una probabilidad ni una clasificación predictiva.</p><button type="button" class="button button-primary" disabled aria-describedby="summary-prediction-reason">Calcular predicción<AppIcon name="lock" :size="16" /></button></div>
      </section>

      <div class="summary-layout">
        <section class="summary-panel" aria-labelledby="summary-baseline-title">
          <div class="summary-panel-heading"><div><h2 id="summary-baseline-title">Datos basales</h2><p>Información de la radiografía inicial.</p></div><RouterLink v-if="editable" :to="editRoute" class="text-button">Revisar datos basales<AppIcon name="arrow-up-right" :size="14" /></RouterLink></div>
          <dl class="summary-data-grid"><div><dt>Evaluación</dt><dd>{{ existing.code }}</dd></div><div><dt>Rodilla</dt><dd>{{ sideLabel }}</dd></div><div><dt>Edad basal</dt><dd>{{ existing.age == null ? 'Pendiente' : `${existing.age} años` }}</dd></div><div><dt>Sexo registrado</dt><dd>{{ sexLabel }}</dd></div><div><dt>IMC basal</dt><dd>{{ existing.bmi == null ? 'Pendiente' : `${decimal.format(existing.bmi)} kg/m²` }}</dd></div><div><dt>Preparación de los datos</dt><dd>{{ state?.baselinesComplete ? 'Lado basal disponible' : 'Lado pendiente' }}</dd></div></dl>
        </section>

        <section class="summary-panel summary-review-panel" aria-labelledby="summary-review-title">
          <div class="summary-panel-heading"><div><h2 id="summary-review-title">Revisión del recorte</h2><p>Decisión simulada temporal; no es QC del servidor.</p></div><RouterLink v-if="editable && state?.baselinesComplete && state?.imageComplete" :to="reviewRoute" class="text-button">Revisar recorte<AppIcon name="arrow-up-right" :size="14" /></RouterLink></div>
          <div class="summary-review-content"><span id="summary-review-status" :class="['summary-review-status', `summary-review-${state?.reviewState}`]"><AppIcon :name="state?.reviewState === 'rejected' ? 'x' : state?.reviewState === 'accepted' ? 'check' : 'clock'" :size="16" />{{ reviewLabel }}</span><p>{{ reviewDescription }}</p><div v-if="state?.reviewState === 'rejected'" class="summary-rejection"><span>Motivo del rechazo de ejemplo</span><strong>{{ reasonLabel }}</strong></div><dl v-if="existing.review && (state?.reviewState === 'accepted' || state?.reviewState === 'rejected')" class="summary-review-details"><div><dt>Decisión simulada en esta sesión</dt><dd>{{ formatDate(existing.review.decidedAt) }} · Lima</dd></div><div><dt>Versión del recorte de ejemplo</dt><dd>{{ crop?.version ?? 'No disponible' }}</dd></div></dl></div>
        </section>

        <section class="summary-panel summary-image-panel" aria-labelledby="summary-image-title">
          <div class="summary-panel-heading"><div><h2 id="summary-image-title">Imagen y lado</h2><p>{{ image ? `${image.label} · ilustración sintética` : 'No se ha seleccionado una imagen de ejemplo.' }}</p></div><RouterLink v-if="editable && state?.baselinesComplete" :to="imageRoute" class="text-button">Revisar imagen y lado<AppIcon name="arrow-up-right" :size="14" /></RouterLink></div>
          <div v-if="image" class="summary-image-content">
            <div class="summary-image-identity"><span>{{ image.fileName }}</span><span>{{ state?.imageComplete ? 'Lado confirmado en el ejemplo' : 'Preparación de imagen pendiente' }}</span></div>
            <div class="summary-previews">
              <figure><figcaption><span>Imagen original</span><button type="button" class="text-button" :disabled="!imageLoaded || imageFailed" @click="enlarge('original')">Ampliar original<AppIcon name="scan" :size="14" /></button></figcaption><div class="summary-preview"><img :src="image.src" :alt="`${image.label}, ilustración sintética`" :width="image.width" :height="image.height" @load="imageLoaded = true" @error="imageFailed = true; imageLoaded = false" /></div><p v-if="imageFailed" class="field-error" role="alert">No pudimos mostrar la imagen de ejemplo. Los datos basales del servidor no cambian; la decisión de ejemplo sigue en memoria.</p><p v-else-if="!imageLoaded" class="summary-preview-loading" role="status">Preparando la imagen original…</p></figure>
              <figure v-if="crop"><figcaption><span>Recorte del mismo lado</span><button type="button" class="text-button" :disabled="!cropLoaded || cropFailed" @click="enlarge('crop')">Ampliar recorte<AppIcon name="scan" :size="14" /></button></figcaption><div class="summary-preview"><DemoCropPreview :fixture="crop" view="crop" @load="cropLoaded = true" @error="cropFailed = true; cropLoaded = false" /></div><p v-if="cropFailed" class="field-error" role="alert">No pudimos mostrar el recorte de ejemplo. Los datos basales del servidor no cambian; la decisión de ejemplo sigue en memoria.</p><p v-else-if="!cropLoaded" class="summary-preview-loading" role="status">Preparando el recorte del ejemplo…</p></figure>
              <div v-else class="summary-no-crop"><AppIcon name="scan" :size="24" /><strong>Recorte pendiente</strong><p>Confirma la imagen y el lado antes de asociar un recorte al caso.</p></div>
            </div>
            <p class="summary-image-note">Las vistas pertenecen a ejemplos de interfaz. No representan un estudio clínico ni una lectura DICOM.</p>
          </div>
          <div v-else class="summary-empty-image"><AppIcon name="image" :size="29" /><h3>Sin imagen asociada</h3><p>Este caso todavía no tiene una imagen sintética seleccionada.</p></div>
        </section>
      </div>

      <div class="summary-preparation"><AppIcon :name="state?.preparationComplete ? 'file' : 'clock'" :size="19" /><div><h2>{{ state?.preparationComplete ? (previewFailed ? 'Simulación temporal; hay una vista no disponible' : 'Datos e imagen preparados en demostración') : 'La preparación del caso tiene pasos pendientes' }}</h2><p v-if="state?.preparationComplete && previewFailed">La preparación y la decisión del ejemplo siguen en memoria, sin guardarse en el servidor. No pudimos mostrar todas las vistas en esta pantalla; puedes volver al listado y abrir el resumen para intentarlo de nuevo. La predicción permanece bloqueada.</p><p v-else-if="state?.preparationComplete">La preparación del ejemplo solo permanece en memoria. La predicción permanece bloqueada y la revisión simulada no acredita calidad clínica.</p><p v-else-if="!state?.baselinesComplete">Completa los datos basales y revisa los pasos de imagen y recorte.</p><p v-else-if="!state?.imageComplete">Selecciona una imagen y confirma el lado para preparar la revisión del recorte.</p><p v-else-if="state?.reviewState === 'rejected'">El recorte de ejemplo está rechazado. Revisa la decisión o sustituye la imagen antes de continuar la preparación.</p><p v-else>Se necesita una aceptación explícita y vigente del recorte de ejemplo.</p><RouterLink v-if="nextStep" id="summary-next-step" :to="nextStep.route" class="button button-primary summary-next-step">{{ nextStep.label }}<AppIcon name="arrow-right" :size="16" /></RouterLink></div></div>
      <p v-if="!editable" class="summary-legacy-note">Este caso precargado permite consultar sus datos; no tiene un recorrido editable asociado. Puedes iniciar una nueva evaluación para explorar el proceso completo.</p>
      <div class="summary-actions"><RouterLink to="/evaluaciones" class="button button-secondary"><AppIcon name="chevron-left" :size="16" />Ir a Mis evaluaciones</RouterLink><p>Datos basales: guardados en el servidor. Imagen y revisión: simulación temporal que se reinicia al recargar o cerrar sesión.</p></div>
    </template>

    <dialog ref="enlarged" class="app-dialog summary-enlarged" aria-labelledby="summary-enlarged-title" @click="closeOnBackdrop($event,enlarged)"><div class="dialog-heading"><h2 id="summary-enlarged-title">{{ enlargedView === 'original' ? 'Imagen original del caso' : 'Recorte del caso' }}</h2><button type="button" class="icon-button" aria-label="Cerrar vista ampliada" @click="enlarged?.close()"><AppIcon name="x" /></button></div><img v-if="enlargedView === 'original' && image" :src="image.src" :alt="`${image.label}, ilustración sintética`" :width="image.width" :height="image.height" /><DemoCropPreview v-else-if="enlargedView === 'crop' && crop" :fixture="crop" view="crop" /><p>{{ sideLabel }} · Imagen ficticia</p><button type="button" class="button button-primary" @click="enlarged?.close()">Volver al resumen</button></dialog>
  </div>
</template>

<style scoped>
.summary-page { width: 100%; max-width: 1240px; margin: 0 auto; }
.summary-back { display: inline-flex; align-items: center; gap: 6px; min-height: 32px; margin-bottom: 25px; color: var(--muted); text-decoration: none; font-size: 12px; }
.summary-back:hover { color: var(--brand); text-decoration: underline; text-underline-offset: 3px; }
.summary-heading { display: flex; justify-content: space-between; align-items: center; gap: 22px; margin-bottom: 26px; }
.summary-heading h1 { margin-top: 8px; font-size: clamp(26px,3vw,32px); line-height: 1.2; font-weight: 650; letter-spacing: -.035em; }
.summary-heading p { color: var(--muted); font-size: 14px; margin-top: 10px; }
.summary-header-meta { display: grid; gap: 9px; justify-items: end; flex: none; }
.summary-header-meta > span:last-child { color: var(--muted); font-size: 11px; }
.summary-code { display: inline-flex; gap: 7px; align-items: center; border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px; font-size: 12px; font-weight: 600; background: white; color: #4d6878; }
.summary-stepper { list-style: none; margin: 0 0 25px; padding: 0; display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 10px; }
.summary-stepper li { display: flex; align-items: center; gap: 8px; padding-bottom: 14px; border-bottom: 2px solid #dce6eb; color: #607684; font-size: 12px; }
.summary-step-number { display: grid; place-items: center; width: 26px; height: 26px; flex: none; border-radius: 50%; background: #e7edf1; color: #607684; font-size: 11px; }
.summary-step-done .summary-step-number { background: #e6f0f1; color: var(--brand); }
.summary-stepper .summary-step-current { border-bottom-color: var(--brand); color: var(--brand-dark); font-weight: 600; }
.summary-step-current .summary-step-number { background: var(--brand); color: white; }
.summary-scope { display: flex; align-items: center; gap: 10px; padding: 13px 16px; border: 1px solid #d9e7e8; border-radius: 9px; background: #edf6f5; color: #416970; margin-bottom: 22px; }
.summary-scope p { font-size: 12px; }
.summary-scope strong { font-weight: 600; }
.summary-prediction { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; padding: 25px 27px; background: white; border: 1px solid #cddde4; border-left: 4px solid #7596a3; border-radius: 11px; margin-bottom: 24px; }
.summary-prediction-main { display: flex; align-items: center; gap: 17px; }
.summary-prediction-icon { width: 51px; height: 54px; display: grid; place-items: center; border-radius: 11px; background: #edf3f6; color: #557585; flex: none; }
.summary-prediction-eyebrow { font-size: 9px; letter-spacing: 1.2px; font-weight: 650; color: #607987; }
.summary-prediction h2 { font-size: 25px; font-weight: 650; letter-spacing: -.025em; margin: 5px 0 4px; }
.summary-model-state { color: #526e7c; font-size: 13px; }
.summary-prediction-explanation { display: grid; justify-items: start; gap: 14px; align-content: center; }
.summary-prediction-explanation p { color: var(--muted); font-size: 12px; line-height: 1.75; max-width: 390px; }
.summary-prediction-explanation button:disabled { background: #e3ebef; color: #516c7b; border-color: #d2dee5; opacity: 1; }
.summary-layout { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 24px; align-items: start; }
.summary-panel { background: white; border: 1px solid var(--border); border-radius: 11px; overflow: hidden; min-width: 0; }
.summary-panel-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 15px; padding: 21px 24px; border-bottom: 1px solid #e7eef1; flex-wrap: wrap; }
.summary-panel-heading h2 { font-size: 16px; font-weight: 650; }
.summary-panel-heading p { font-size: 12px; color: var(--muted); margin-top: 4px; }
.summary-panel-heading .text-button { min-height: 32px; font-size: 12px; }
.summary-data-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 23px 20px; margin: 0; padding: 24px; }
.summary-data-grid dt,.summary-review-details dt { color: #5e7684; font-size: 11px; }
.summary-data-grid dd { margin: 5px 0 0; color: var(--ink); font-size: 14px; font-weight: 600; overflow-wrap: anywhere; }
.summary-review-content { padding: 24px; }
.summary-review-status { display: inline-flex; align-items: center; gap: 7px; padding: 7px 9px; border-radius: 7px; background: #eef3f6; color: #526c7a; font-size: 12px; font-weight: 600; }
.summary-review-accepted { background: #eaf3f4; color: #2c6570; }
.summary-review-rejected { background: #fcf0ed; color: #954d40; }
.summary-review-outdated { background: #fff5e7; color: #866127; }
.summary-review-content > p { font-size: 12px; color: var(--muted); line-height: 1.75; margin-top: 15px; }
.summary-rejection { display: grid; gap: 5px; margin-top: 18px; padding: 13px; border-radius: 7px; background: #fff7f4; }
.summary-rejection span { font-size: 11px; color: #83594e; }
.summary-rejection strong { font-size: 12px; font-weight: 600; color: #7d4035; }
.summary-review-details { display: grid; gap: 15px; margin: 20px 0 0; padding-top: 17px; border-top: 1px solid #e5edf0; }
.summary-review-details dd { font-size: 12px; color: var(--ink); margin: 4px 0 0; }
.summary-image-panel { grid-column: 1/-1; }
.summary-image-content { padding: 22px 24px; }
.summary-image-identity { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; color: #587381; font-size: 11px; margin-bottom: 17px; }
.summary-previews { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 22px; }
.summary-previews figure { margin: 0; min-width: 0; }
.summary-previews figcaption { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; font-size: 12px; font-weight: 600; }
.summary-previews figcaption .text-button { min-height: 32px; font-size: 12px; }
.summary-preview-loading { margin-top: 10px; color: var(--muted); font-size: 12px; line-height: 1.7; }
.summary-preview { height: 240px; background: #12212e; border-radius: 8px; padding: 8px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
.summary-preview > img { width: 100%; height: 100%; object-fit: contain; }
.summary-no-crop { min-height: 240px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 25px; gap: 11px; color: #5e7987; border: 1px dashed #cbdbe3; border-radius: 8px; text-align: center; background: #f8fbfc; }
.summary-no-crop strong { font-size: 13px; font-weight: 600; }
.summary-no-crop p { max-width: 230px; font-size: 12px; line-height: 1.75; }
.summary-image-note { color: #5b7483; font-size: 11px; margin-top: 17px; }
.summary-empty-image { min-height: 200px; padding: 30px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 10px; color: #607a88; }
.summary-empty-image h3 { font-size: 15px; font-weight: 600; color: var(--ink); }
.summary-empty-image p { font-size: 12px; }
.summary-preparation { display: flex; gap: 12px; align-items: flex-start; margin-top: 24px; padding: 19px 21px; background: #edf3f6; border: 1px solid #dbe6ec; border-radius: 9px; color: #4e6b7d; }
.summary-preparation > svg { margin-top: 2px; }
.summary-preparation h2 { font-size: 13px; color: #355566; font-weight: 600; }
.summary-preparation p { margin-top: 6px; font-size: 12px; line-height: 1.7; }
.summary-next-step { margin-top: 15px; }
.summary-legacy-note { color: var(--muted); font-size: 12px; margin-top: 18px; line-height: 1.7; }
.summary-actions { display: flex; gap: 20px; align-items: center; justify-content: space-between; margin-top: 25px; flex-wrap: wrap; }
.summary-actions p { color: var(--muted); font-size: 11px; max-width: 440px; }
.summary-symbol { width: 50px; height: 52px; border-radius: 10px; display: grid; place-items: center; background: #e9f2f3; color: var(--brand); }
.summary-unavailable { padding: 48px 28px; border: 1px solid var(--border); background: white; border-radius: 12px; display: grid; justify-items: start; gap: 20px; }
.summary-unavailable h1 { font-size: 27px; }
.summary-unavailable p { color: var(--muted); max-width: 500px; }
.summary-enlarged { width: min(820px,calc(100% - 32px)); }
.summary-enlarged > img { display: block; width: 100%; height: auto; max-height: 60dvh; object-fit: contain; border-radius: 8px; background: #12212e; }
.summary-enlarged > :deep(svg.demo-crop-viewport) { height: min(520px,60dvh); }
.summary-enlarged > p { margin-top: 15px; font-size: 12px; }
.summary-enlarged > .button { margin-top: 5px; }
@media(max-width:1050px) { .summary-heading { align-items: flex-start; flex-wrap: wrap; } .summary-header-meta { justify-items: start; } .summary-prediction { grid-template-columns: 1fr; gap: 19px; } }
@media(max-width:760px) {
  .summary-heading { gap: 16px; margin-bottom: 24px; }
  .summary-heading p { font-size: 13px; }
  .summary-stepper { grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; }
  .summary-stepper li { font-size: 11px; }
  .summary-scope { align-items: flex-start; padding: 13px; }
  .summary-scope p { line-height: 1.7; }
  .summary-prediction { padding: 20px 17px; }
  .summary-prediction-main { gap: 13px; }
  .summary-prediction h2 { font-size: 23px; }
  .summary-prediction-eyebrow { font-size: 8px; letter-spacing: .9px; }
  .summary-layout { grid-template-columns: 1fr; gap: 20px; }
  .summary-panel-heading { padding: 19px; gap: 13px; }
  .summary-data-grid { padding: 20px; gap: 22px 15px; }
  .summary-data-grid dd { font-size: 13px; }
  .summary-review-content { padding: 20px; }
  .summary-image-content { padding: 20px; }
  .summary-previews { grid-template-columns: 1fr; gap: 21px; }
  .summary-preview { height: 220px; }
  .summary-image-note { line-height: 1.7; }
  .summary-preparation { padding: 16px; }
  .summary-actions { align-items: stretch; gap: 16px; }
  .summary-actions .button { width: 100%; }
  .summary-actions p { font-size: 11px; }
}
@media(max-width:360px) { .summary-data-grid { grid-template-columns: 1fr; } .summary-previews figcaption { flex-wrap: wrap; } .summary-prediction-main { align-items: flex-start; } .summary-prediction-icon { width: 40px; height: 43px; } .summary-prediction h2 { font-size: 21px; } }
</style>
