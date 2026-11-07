<script setup lang="ts">
import { computed, nextTick, ref, toRefs } from 'vue'
import { useRouter } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import type { Evaluation, EvaluationStatus } from '../demo/demoEvaluations'
import { evaluations, evaluationIssue, listFailed, listLoading, loadEvaluations } from '../state/evaluations'
import { demoListPreferences } from '../demoListPreferences'

const router = useRouter()
const { search, statusFilter } = toRefs(demoListPreferences)
const searchInput = ref<HTMLInputElement | null>(null)
const statuses: Partial<Record<EvaluationStatus, { label: string; icon: string }>> = {
  draft: { label: 'Datos basales guardados', icon: 'file' },
  crop_accepted: { label: 'Ejemplo aceptado solo en esta sesión', icon: 'check' },
  crop_rejected: { label: 'Ejemplo rechazado solo en esta sesión', icon: 'x' },
}
const steps = ['Datos basales en servidor', 'Imagen simulada', 'Revisión simulada', 'Resumen del ejemplo']
const filtered = computed(() => evaluations.value.filter(item => item.code.toLocaleLowerCase('es').includes(search.value.trim().toLocaleLowerCase('es')) && (statusFilter.value === 'all' || item.status === statusFilter.value)))
const hasFilters = computed(() => Boolean(search.value.trim() || statusFilter.value !== 'all'))
const dateFormatter = new Intl.DateTimeFormat('es-PE', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'America/Lima' })
const timeFormatter = new Intl.DateTimeFormat('es-PE', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'America/Lima' })
function actionLabel(item: Evaluation): string { return item.progressStep === 4 ? 'Ver resumen' : item.review ? 'Ver revisión' : 'Retomar' }
function openCase(item: Evaluation) {
  void router.push({ name: item.progressStep === 4 ? 'case-summary' : item.review ? 'crop-review' : item.progressStep === 2 ? 'radiograph' : 'edit-evaluation', params: { id: item.id } })
}
async function clearFilters() { search.value = ''; statusFilter.value = 'all'; await nextTick(); searchInput.value?.focus() }
async function reloadList() { try { await loadEvaluations() } catch { /* Error remains visible; never populate fixtures. */ } }
</script>

<template>
  <div class="eval-page">
    <header class="eval-page-heading"><div><p class="eval-eyebrow">ESPACIO DE TRABAJO</p><h1 tabindex="-1">Mis evaluaciones</h1><p class="eval-intro">Datos basales ficticios guardados en el servidor y disponibles para tu cuenta.</p></div><button type="button" class="eval-button eval-button-primary" :disabled="listLoading" @click="router.push('/evaluaciones/nueva')"><AppIcon name="plus" /><span>Nueva evaluación</span></button></header>
    <div class="eval-demo-notice"><span class="eval-demo-symbol"><AppIcon name="shield" /></span><div><strong>Acceso real · casos ficticios</strong><p>El servidor conserva los datos basales. Las imágenes y la revisión siguen siendo una simulación temporal; la predicción permanece bloqueada.</p></div></div>
    <section class="eval-workspace" aria-label="Listado de evaluaciones" :aria-busy="listLoading">
      <div class="eval-toolbar"><div class="eval-search-group"><label for="evaluation-search">Buscar evaluación</label><div class="eval-search-field"><AppIcon name="search" /><input id="evaluation-search" ref="searchInput" v-model="search" type="search" autocomplete="off" placeholder="Buscar por código" /></div></div><div class="eval-filter-group"><label for="evaluation-status">Estado del recorrido</label><div class="eval-select-field"><select id="evaluation-status" v-model="statusFilter"><option value="all">Todos los estados</option><option v-for="(value, key) in statuses" :key="key" :value="key">{{ value?.label }}</option></select><AppIcon name="chevron-down" /></div></div></div>
      <div class="eval-list-meta"><p role="status" aria-live="polite"><template v-if="listLoading">Consultando el servidor…</template><template v-else-if="!listFailed"><strong>{{ filtered.length }}</strong> {{ filtered.length === 1 ? 'evaluación' : 'evaluaciones' }}{{ hasFilters ? ' encontradas' : ' guardadas en el servidor' }}</template></p><button v-if="hasFilters" class="eval-reset-button" @click="clearFilters"><AppIcon name="x" />Limpiar filtros</button><button v-else type="button" class="eval-reset-button" :disabled="listLoading" @click="reloadList">Actualizar listado</button></div>
      <div v-if="listFailed" id="server-evaluations-error" class="eval-empty-state" role="alert"><h2>No pudimos recuperar tus evaluaciones</h2><p>{{ evaluationIssue }}</p><p>No se muestran ejemplos locales como sustituto del servidor.</p><button type="button" class="eval-button eval-button-secondary" :disabled="listLoading" @click="reloadList">Reintentar</button></div>
      <div v-else-if="filtered.length" class="eval-table-scroll"><table class="eval-table"><caption class="eval-sr-only">Casos ficticios de tu cuenta ordenados por actualización del servidor</caption><thead><tr><th scope="col">Evaluación</th><th scope="col">Último guardado basal</th><th scope="col">Paso del recorrido</th><th scope="col">Estado</th><th scope="col"><span class="eval-sr-only">Acción</span></th></tr></thead><tbody><tr v-for="item in filtered" :key="item.id"><th scope="row" class="eval-case-cell"><span class="eval-file-icon"><AppIcon name="file" /></span><span><RouterLink :to="{ name: 'case-summary', params: { id: item.id } }" class="eval-code eval-code-link" :aria-label="`Ver resumen del caso ${item.code}`">{{ item.code }}</RouterLink><span class="eval-cell-secondary">{{ item.knee === 'right' ? 'Rodilla derecha' : item.knee === 'left' ? 'Rodilla izquierda' : 'Rodilla pendiente' }}</span></span></th><td class="eval-date-cell"><time :datetime="item.updatedAt">{{ dateFormatter.format(new Date(item.updatedAt)) }}</time><span class="eval-cell-secondary">{{ timeFormatter.format(new Date(item.updatedAt)) }} · Lima</span></td><td class="eval-step-cell"><span class="eval-step-label">{{ steps[item.progressStep - 1] }}</span></td><td class="eval-status-cell"><span :class="['eval-status', `eval-status-${item.status}`]"><AppIcon :name="statuses[item.status]?.icon ?? 'file'" />{{ statuses[item.status]?.label ?? 'Datos basales guardados' }}</span></td><td class="eval-action-cell"><button type="button" class="eval-row-button" :aria-label="`${actionLabel(item)} ${item.code}`" @click="openCase(item)">{{ actionLabel(item) }}<AppIcon name="arrow-right" /></button></td></tr></tbody></table></div>
      <div v-else-if="!listLoading" class="eval-empty-state"><span class="eval-empty-icon"><AppIcon name="search" /></span><h2>{{ hasFilters ? 'No encontramos evaluaciones' : 'Todavía no tienes evaluaciones' }}</h2><p>{{ hasFilters ? 'Prueba con otro código o cambia el filtro.' : 'Crea un caso ficticio para guardar sus datos basales en tu cuenta.' }}</p><button v-if="hasFilters" class="eval-button eval-button-secondary" @click="clearFilters">Mostrar todas las evaluaciones</button></div>
      <footer class="eval-table-footer"><AppIcon name="lock" /><p>El servidor verifica tu sesión y el acceso a cada caso. La simulación de imágenes no es una revisión clínica guardada.</p></footer>
    </section>
    <p class="eval-bottom-note">Una evaluación corresponde a una rodilla. Usa únicamente datos ficticios en esta fase.</p>
  </div>
</template>

<style scoped>
.eval-page { --eval-ink: #173042; --eval-muted: #667b87; --eval-teal: #126d78; --eval-line: #e2e9ec; color: var(--eval-ink); width: 100%; max-width: 1240px; margin: 0 auto; }
.eval-page-heading { display: flex; justify-content: space-between; align-items: center; gap: 24px; flex-wrap: wrap; margin-bottom: 28px; }
.eval-eyebrow { color: var(--eval-teal); font-size: 10px; font-weight: 700; letter-spacing: .16em; margin: 0 0 9px; }
.eval-page h1 { margin: 0; font-size: clamp(26px, 3vw, 32px); line-height: 1.2; font-weight: 650; letter-spacing: -.035em; }
.eval-intro { margin: 10px 0 0; color: var(--eval-muted); font-size: 14px; }
.eval-button { display: inline-flex; align-items: center; justify-content: center; gap: 9px; min-height: 44px; border: 1px solid transparent; border-radius: 9px; padding: 11px 17px; font: inherit; font-size: 13px; font-weight: 600; cursor: pointer; }
.eval-button-primary { color: white; background: var(--eval-teal); box-shadow: 0 3px 7px #126d7815; }
.eval-button-primary:hover { background: #0c5b65; }
.eval-button-secondary { color: var(--eval-ink); background: white; border-color: #cfdbdf; }
.eval-button :deep(svg) { width: 17px; height: 17px; }
.eval-demo-notice { display: flex; gap: 12px; align-items: center; border: 1px solid #dbe9e8; border-radius: 10px; background: #eef7f6; padding: 15px 18px; margin-bottom: 24px; }
.eval-demo-symbol { color: var(--eval-teal); flex: none; display: flex; }
.eval-demo-symbol :deep(svg) { width: 20px; height: 20px; }
.eval-demo-notice strong { font-size: 12px; font-weight: 650; }
.eval-demo-notice p { margin: 3px 0 0; font-size: 12px; color: #54716e; }
.eval-workspace { background: white; border: 1px solid var(--eval-line); border-radius: 12px; box-shadow: 0 3px 16px #17304203; overflow: hidden; }
.eval-toolbar { display: flex; gap: 16px; flex-wrap: wrap; padding: 24px 24px 20px; }
.eval-search-group { flex: 1 1 290px; min-width: 0; }
.eval-filter-group { flex: 0 1 230px; min-width: 200px; }
.eval-toolbar label { display: block; font-size: 11px; font-weight: 650; color: #536b78; margin-bottom: 7px; }
.eval-search-field, .eval-select-field { position: relative; }
.eval-search-field > :deep(svg) { position: absolute; width: 17px; height: 17px; color: #81939c; left: 13px; top: 13px; pointer-events: none; }
.eval-search-field input, .eval-select-field select { width: 100%; min-width: 0; height: 44px; background: white; border: 1px solid #d6e0e4; border-radius: 8px; color: var(--eval-ink); font: inherit; font-size: 13px; }
.eval-search-field input { padding: 10px 12px 10px 39px; }
.eval-search-field input::placeholder { color: #8999a2; }
.eval-select-field select { appearance: none; padding: 10px 35px 10px 13px; }
.eval-select-field > :deep(svg) { position: absolute; right: 13px; top: 15px; width: 14px; height: 14px; color: #6f828e; pointer-events: none; }
.eval-list-meta { padding: 0 24px 17px; display: flex; justify-content: space-between; align-items: center; gap: 12px; min-height: 36px; }
.eval-list-meta p { margin: 0; font-size: 12px; color: var(--eval-muted); }
.eval-list-meta strong { font-weight: 650; color: var(--eval-ink); }
.eval-sort-label { color: #80909b; font-size: 11px; }
.eval-reset-button { background: transparent; border: 0; padding: 5px 0; color: var(--eval-teal); font: inherit; font-size: 12px; display: inline-flex; align-items: center; gap: 5px; cursor: pointer; }
.eval-reset-button :deep(svg) { width: 13px; height: 13px; }
.eval-table-scroll { overflow-x: auto; }
.eval-table { width: 100%; border-collapse: collapse; text-align: left; font-size: 12px; }
.eval-table thead { background: #f8fafb; }
.eval-table thead th { font-size: 10px; font-weight: 650; color: #738690; letter-spacing: .025em; padding: 13px 15px; white-space: nowrap; border-top: 1px solid #edf1f3; border-bottom: 1px solid #e8eef0; }
.eval-table thead th:first-child { padding-left: 24px; }
.eval-table tbody th, .eval-table tbody td { padding: 18px 15px; border-bottom: 1px solid #edf1f3; vertical-align: middle; }
.eval-table tbody tr:last-child th, .eval-table tbody tr:last-child td { border-bottom: 0; }
.eval-table tbody tr:hover { background: #fcfdfd; }
.eval-case-cell { display: flex; align-items: center; gap: 11px; padding-left: 24px !important; }
.eval-file-icon { display: flex; align-items: center; justify-content: center; width: 34px; height: 39px; background: #f1f6f8; color: #607d8b; border: 1px solid #e7eef0; border-radius: 7px; flex: none; }
.eval-file-icon :deep(svg) { width: 17px; height: 17px; }
.eval-code { display: block; font-size: 12px; font-weight: 650; letter-spacing: .025em; }
.eval-code-link { color: var(--eval-ink); text-decoration: none; }
.eval-code-link:hover { color: var(--eval-teal); text-decoration: underline; text-underline-offset: 3px; }
.eval-cell-secondary { display: block; font-size: 10px; font-weight: 400; color: #82939d; margin-top: 4px; white-space: nowrap; }
.eval-date-cell { white-space: nowrap; color: #526c7a; font-size: 11px; }
.eval-step-label { display: block; font-size: 11px; color: #607985; white-space: nowrap; }
.eval-step-track { display: flex; gap: 3px; margin-top: 7px; width: 73px; }
.eval-step-track > span { width: 16px; height: 3px; border-radius: 1px; background: #e6edf0; }
.eval-step-track > span.eval-step-done { background: #7eafb4; }
.eval-status { display: inline-flex; align-items: center; gap: 5px; font-size: 10px; font-weight: 550; border-radius: 5px; padding: 5px 7px; white-space: nowrap; }
.eval-status :deep(svg) { width: 12px; height: 12px; flex: none; }
.eval-status-draft { background: #f0f3f5; color: #4f6572; }
.eval-status-review { background: #fff5e6; color: #8c641e; }
.eval-status-rejected { background: #fff0ee; color: #963f37; }
.eval-status-model_pending { background: #edf2f8; color: #435e79; }
.eval-status-crop_accepted { background: #e8f4ed; color: #2f6a47; }
.eval-status-crop_rejected { background: #fff0ee; color: #963f37; }
.eval-action-cell { text-align: right; padding-right: 24px !important; }
.eval-row-button { color: var(--eval-teal); background: transparent; border: 1px solid #dbe8e9; border-radius: 6px; padding: 7px 9px; display: inline-flex; justify-content: center; align-items: center; gap: 6px; font: inherit; font-size: 11px; font-weight: 550; cursor: pointer; white-space: nowrap; min-height: 34px; }
.eval-row-button:hover { background: #eef7f6; border-color: #b4d2d5; }
.eval-row-button :deep(svg) { width: 13px; height: 13px; }
.eval-table-footer { border-top: 1px solid #e8eef0; display: flex; gap: 7px; padding: 14px 24px; align-items: center; color: #7d909b; background: #fdfefe; }
.eval-table-footer :deep(svg) { width: 12px; height: 12px; flex: none; }
.eval-table-footer p { font-size: 10px; margin: 0; }
.eval-bottom-note { margin: 18px 0 0; font-size: 11px; color: #85959e; }
.eval-empty-state { min-height: 300px; padding: 38px 24px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; border-top: 1px solid #edf1f3; }
.eval-empty-icon { display: grid; place-items: center; background: #f1f6f7; border-radius: 50%; width: 50px; height: 50px; color: #78969d; margin-bottom: 16px; }
.eval-empty-icon :deep(svg) { width: 23px; height: 23px; }
.eval-empty-state h2 { margin: 0; font-size: 17px; font-weight: 600; letter-spacing: -.02em; }
.eval-empty-state p { color: var(--eval-muted); font-size: 13px; margin: 8px 0 22px; }
.eval-dialog { color: var(--eval-ink); border: 1px solid #dbe5e8; border-radius: 16px; padding: 26px; width: min(520px, calc(100vw - 32px)); max-height: calc(100dvh - 40px); box-shadow: 0 20px 80px #17304226; overflow-y: auto; }
.eval-dialog::backdrop { background: #17304266; backdrop-filter: blur(3px); }
.eval-dialog-heading { display: flex; justify-content: space-between; align-items: center; margin-bottom: 11px; }
.eval-dialog-eyebrow { color: #82939c; font-size: 10px; font-weight: 650; letter-spacing: .12em; }
.eval-dialog-close { display: grid; place-items: center; width: 32px; height: 32px; border: 1px solid #e3e9ec; border-radius: 7px; background: white; color: #607684; cursor: pointer; }
.eval-dialog-close :deep(svg) { width: 16px; height: 16px; }
.eval-dialog h2 { font-size: 24px; letter-spacing: -.03em; margin: 0; font-weight: 650; }
.eval-dialog-description { margin: 6px 0 18px; font-size: 13px; color: var(--eval-muted); }
.eval-detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px 18px; margin: 24px 0; padding: 22px 0; border-top: 1px solid #e9eef0; border-bottom: 1px solid #e9eef0; }
.eval-detail-grid dt { font-size: 12px; color: #536b79; }
.eval-detail-grid dd { margin: 5px 0 0; font-size: 13px; font-weight: 550; }
.eval-dialog-callout { display: flex; align-items: flex-start; gap: 10px; background: #f1f7f7; border-radius: 8px; padding: 13px; }
.eval-dialog-callout > :deep(svg) { width: 17px; height: 17px; color: var(--eval-teal); flex: none; margin-top: 2px; }
.eval-dialog-callout p { margin: 0; color: #53747a; font-size: 12px; line-height: 1.6; }
.eval-dialog-scope { font-size: 12px; line-height: 1.6; color: #536b79; margin: 17px 0 0; }
.eval-dialog-actions { margin-top: 26px; display: flex; justify-content: flex-end; }
.eval-next-steps { list-style: none; padding: 4px 0 10px; margin: 0; display: grid; gap: 16px; }
.eval-next-steps li { display: flex; align-items: center; gap: 11px; font-size: 13px; }
.eval-next-steps li > span { width: 27px; height: 27px; display: grid; place-items: center; background: #edf5f5; color: var(--eval-teal); font-size: 12px; font-weight: 600; border-radius: 50%; }
.eval-sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
.eval-page button:focus-visible, .eval-page input:focus-visible, .eval-page select:focus-visible { outline: 3px solid #66a9b1; outline-offset: 3px; }
.eval-eyebrow, .eval-dialog-eyebrow, .eval-table thead th { font-size: 11px; }
.eval-cell-secondary, .eval-sort-label, .eval-step-label, .eval-status, .eval-row-button, .eval-date-cell, .eval-table-footer p, .eval-bottom-note { font-size: 12px; }
.eval-cell-secondary, .eval-sort-label, .eval-table-footer, .eval-bottom-note, .eval-dialog-eyebrow { color: #647c88; }
@media (max-width: 1050px) { .eval-step-cell { display: none; } .eval-table thead th:nth-child(3) { display: none; } }
@media (max-width: 720px) {
  .eval-page-heading { gap: 18px; margin-bottom: 22px; align-items: flex-start; }
  .eval-intro { font-size: 13px; line-height: 1.6; }
  .eval-demo-notice { padding: 13px; align-items: flex-start; }
  .eval-demo-notice p { line-height: 1.6; }
  .eval-toolbar { padding: 18px; gap: 14px; }
  .eval-search-group, .eval-filter-group { flex: 1 1 100%; min-width: 0; }
  .eval-search-field input, .eval-select-field select { font-size: 14px; }
  .eval-list-meta { padding: 0 18px 16px; }
  .eval-sort-label { font-size: 11px; }
  .eval-table thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); }
  .eval-table, .eval-table tbody { display: block; }
  .eval-table tbody tr { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px 12px; border-top: 1px solid #e8eef0; padding: 18px; }
  .eval-table tbody th, .eval-table tbody td { padding: 0 !important; border-bottom: 0; }
  .eval-case-cell { grid-column: 1; grid-row: 1; }
  .eval-date-cell { grid-column: 1; grid-row: 3; padding-left: 45px !important; }
  .eval-date-cell .eval-cell-secondary { display: inline; margin-left: 5px; }
  .eval-status-cell { grid-column: 1 / -1; grid-row: 2; padding-left: 45px !important; }
  .eval-action-cell { grid-column: 2; grid-row: 1; align-self: center; }
  .eval-row-button { min-height: 40px; }
  .eval-table-footer { padding: 14px 18px; align-items: flex-start; }
  .eval-table-footer p { line-height: 1.5; }
  .eval-dialog { padding: 21px; }
}
@media (pointer: coarse) { .eval-row-button, .eval-dialog-close { min-height: 44px; } .eval-search-field input, .eval-select-field select { font-size: 16px; } }
</style>
