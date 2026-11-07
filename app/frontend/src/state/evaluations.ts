import { computed, ref, watch } from 'vue'
import { apiRequest, ApiError } from '../api/http'
import { expectedStatus, parseEvaluation, parseEvaluationPage, strongEtag, type ClinicalDto, type EvaluationDto } from '../api/contracts'
import { apiMessage, currentCsrf, sessionGeneration } from './session'
import type { Evaluation } from '../demo/demoEvaluations'
import type { DemoImageState } from '../demo/demoImages'
import { demoImages } from '../demo/demoImages'
import { demoRejectReasons, getDemoCrop, type DemoReviewInput } from '../demo/demoCrops'

export type DemoImageInput = DemoImageState
export type { DemoReviewInput } from '../demo/demoCrops'
export interface DemoDraftInput { code: string; knee: '' | 'right' | 'left'; age: string; sex: '' | 'F' | 'M'; bmi: string }
type Simulation = { revisionId: string; image: Evaluation['image']; review: Evaluation['review']; progressStep: Evaluation['progressStep']; status: Evaluation['status'] }

const records = ref<EvaluationDto[]>([])
const etags = new Map<string, string>()
const simulations = ref<Record<string, Simulation>>({})
export const evaluationIssue = ref('')
export const listLoading = ref(false)
export const listFailed = ref(false)
export const legacyDraftsPresent = ref(false)
try { legacyDraftsPresent.value = localStorage.getItem('kneeoai:demo-evaluations:v1') !== null } catch { /* Never delete or replace a legacy copy. */ }

export const evaluations = computed<Evaluation[]>(() => records.value.map(item => {
  const simulation = simulations.value[item.id]
  const current = simulation?.revisionId === item.currentRevisionId ? simulation : undefined
  return {
    id: item.id, code: item.code, knee: item.clinical.side, age: item.clinical.age, sex: item.clinical.sex, bmi: item.clinical.bmi,
    updatedAt: item.updatedAt, status: current?.status ?? 'draft', progressStep: current?.progressStep ?? 1,
    image: current?.image ?? null, review: current?.review ?? null,
  }
}))

export function resetWorkspace(): void {
  records.value = []; simulations.value = {}; etags.clear(); evaluationIssue.value = ''; listFailed.value = false; listLoading.value = false
}
watch(sessionGeneration, resetWorkspace, { flush: 'sync' })
function assertCurrentSession(generation: number): void {
  if (generation !== sessionGeneration.value) {
    throw new ApiError({ status: 0, code: 'SESSION_CHANGED', message: 'La sesión cambió mientras se procesaba la solicitud. Se descartó la respuesta anterior.' })
  }
}
function accept(item: EvaluationDto, etag?: string): void {
  const previous = records.value.find(record => record.id === item.id)
  if (previous?.currentRevisionId !== item.currentRevisionId) delete simulations.value[item.id]
  records.value = [item, ...records.value.filter(record => record.id !== item.id)].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  if (etag) etags.set(item.id, etag)
  else if (previous?.currentRevisionId !== item.currentRevisionId) etags.delete(item.id)
}
export async function loadEvaluation(id: string): Promise<void> {
  const generation = sessionGeneration.value
  evaluationIssue.value = ''
  try {
    const response = await apiRequest(`/evaluations/${encodeURIComponent(id)}`)
    assertCurrentSession(generation)
    expectedStatus(response.status, 200)
    const item = parseEvaluation(response.data)
    if (item.id !== id) throw new ApiError({ status: 200, code: 'INVALID_RESPONSE', message: 'La respuesta no corresponde al caso solicitado.' })
    accept(item, strongEtag(response.etag))
  } catch (error) {
    if (generation !== sessionGeneration.value) throw error
    records.value = records.value.filter(item => item.id !== id)
    delete simulations.value[id]; etags.delete(id)
    evaluationIssue.value = apiMessage(error)
    throw error
  }
}
export async function loadEvaluations(): Promise<void> {
  const generation = sessionGeneration.value
  listLoading.value = true; listFailed.value = false; evaluationIssue.value = ''
  try {
    const items: EvaluationDto[] = []
    const cursors = new Set<string>()
    let cursor: string | null = null
    do {
      const response = await apiRequest(`/evaluations?limit=100${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''}`)
      assertCurrentSession(generation)
      expectedStatus(response.status, 200)
      const page = parseEvaluationPage(response.data)
      items.push(...page.items)
      cursor = page.nextCursor
      if (cursor && cursors.has(cursor)) throw new ApiError({ status: 200, code: 'INVALID_RESPONSE', message: 'No pudimos verificar la paginación del listado.' })
      if (cursor) cursors.add(cursor)
    } while (cursor)
    if (new Set(items.map(item => item.id)).size !== items.length) throw new ApiError({ status: 200, code: 'INVALID_RESPONSE', message: 'El listado contiene identificadores repetidos.' })
    const ids = new Set(items.map(item => item.id))
    for (const id of Object.keys(simulations.value)) if (!ids.has(id)) delete simulations.value[id]
    const prior = new Map(records.value.map(item => [item.id, item.currentRevisionId]))
    for (const item of items) if (prior.get(item.id) !== item.currentRevisionId) { delete simulations.value[item.id]; etags.delete(item.id) }
    records.value = items.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  } catch (error) {
    if (generation !== sessionGeneration.value) throw error
    records.value = []; simulations.value = {}; etags.clear(); listFailed.value = true
    evaluationIssue.value = apiMessage(error)
    throw error
  } finally { if (generation === sessionGeneration.value) listLoading.value = false }
}

export function isEditableDemoEvaluation(existing: Evaluation | undefined): boolean { return Boolean(existing) }
export function createDraftInput(existing?: Evaluation): DemoDraftInput {
  return { code: existing?.code ?? 'Se asigna al guardar', knee: existing?.knee ?? '', age: existing?.age == null ? '' : String(existing.age), sex: existing?.sex ?? '', bmi: existing?.bmi == null ? '' : String(existing.bmi) }
}
function decimal(value: string): number | null {
  if (!value.trim()) return null
  const normalized = value.trim().replace(',', '.')
  return /^\d+(?:\.\d+)?$/.test(normalized) ? Number(normalized) : NaN
}
export function validateDemoDraft(input: DemoDraftInput, requireSide: boolean): Partial<Record<keyof DemoDraftInput, string>> {
  const errors: Partial<Record<keyof DemoDraftInput, string>> = {}
  const age = decimal(input.age), bmi = decimal(input.bmi)
  if (age !== null && (!Number.isFinite(age) || !Number.isInteger(age) || age <= 0)) errors.age = 'Escribe una edad entera mayor que cero o deja el campo vacío si no está disponible.'
  if (bmi !== null && (!Number.isFinite(bmi) || bmi <= 0)) errors.bmi = 'Escribe un IMC mayor que cero o deja el campo vacío si no está disponible.'
  if (input.sex !== '' && input.sex !== 'F' && input.sex !== 'M') errors.sex = 'Selecciona una opción disponible.'
  if (input.knee !== '' && input.knee !== 'right' && input.knee !== 'left') errors.knee = 'Selecciona una rodilla disponible.'
  if (requireSide && !input.knee) errors.knee = 'Selecciona la rodilla antes de explorar la imagen sintética.'
  return errors
}
export async function saveDemoDraft(input: DemoDraftInput, existingId?: string, options: { invalidateImageSide?: boolean; idempotencyKey?: string } = {}): Promise<Evaluation> {
  const generation = sessionGeneration.value
  if (Object.keys(validateDemoDraft(input, false)).length) throw new Error('draft_validation_required')
  const clinical: ClinicalDto = { age: decimal(input.age), sex: input.sex || null, bmi: decimal(input.bmi), side: input.knee || null }
  const ifMatch = existingId ? etags.get(existingId) : undefined
  if (existingId && !ifMatch) throw new ApiError({ status: 428, code: 'VERSION_REQUIRED', message: 'Vuelve a abrir el caso para recuperar su versión.' })
  if (!existingId && !options.idempotencyKey) throw new ApiError({ status: 0, code: 'INVALID_REQUEST', message: 'No pudimos identificar la intención de crear el caso.' })
  const response = await apiRequest(existingId ? `/evaluations/${encodeURIComponent(existingId)}` : '/evaluations', {
    method: existingId ? 'PATCH' : 'POST', csrfToken: currentCsrf(), ifMatch, idempotencyKey: existingId ? undefined : options.idempotencyKey, body: { clinical },
  })
  assertCurrentSession(generation)
  if (existingId) expectedStatus(response.status, 200)
  else if (response.status !== 200 && response.status !== 201) expectedStatus(response.status, 201)
  const item = parseEvaluation(response.data)
  if (existingId && item.id !== existingId) throw new ApiError({ status: response.status, code: 'INVALID_RESPONSE', message: 'El servidor no confirmó el caso solicitado.' })
  accept(item, strongEtag(response.etag))
  if (options.invalidateImageSide) delete simulations.value[item.id]
  return evaluations.value.find(record => record.id === item.id)!
}

// SVG choices and review decisions are isolated UI simulation, never API QC and never Storage.
function simulationFor(existing: Evaluation): Simulation {
  const canonical = records.value.find(item => item.id === existing.id)
  if (!canonical) throw new Error('server_case_required')
  return { revisionId: canonical.currentRevisionId, image: existing.image, review: existing.review, progressStep: existing.progressStep, status: existing.status }
}
export function createDemoImageInput(existing: Evaluation): DemoImageInput {
  return { sampleId: existing.image?.sampleId ?? '', knee: existing.image?.knee ?? existing.knee ?? '', sideConfirmed: Boolean(existing.image?.sampleId && existing.image.knee === existing.knee && existing.image.sideConfirmed) }
}
export function validateDemoImage(input: DemoImageInput, existing: Evaluation, required: boolean): Partial<Record<keyof DemoImageInput, string>> {
  const errors: Partial<Record<keyof DemoImageInput, string>> = {}
  if (!existing.knee) errors.knee = 'Guarda el lado en los datos basales antes de explorar el ejemplo.'
  if (input.sampleId && !demoImages.some(item => item.id === input.sampleId)) errors.sampleId = 'Selecciona una imagen de ejemplo disponible.'
  if (required && !input.sampleId) errors.sampleId = 'Selecciona una imagen de ejemplo.'
  if (input.knee && input.knee !== existing.knee) errors.knee = 'El lado no coincide con los datos basales guardados.'
  if (required && !input.knee) errors.knee = 'Selecciona el lado del ejemplo.'
  if (required && !input.sideConfirmed) errors.sideConfirmed = 'Confirma el lado del ejemplo.'
  if (input.sideConfirmed && (!input.sampleId || input.knee !== existing.knee)) errors.sideConfirmed = 'Revisa la imagen y el lado antes de confirmar.'
  return errors
}
export function saveDemoImage(input: DemoImageInput, id: string, options: { invalidateReview?: boolean } = {}): Evaluation {
  const existing = evaluations.value.find(item => item.id === id)
  if (!existing || Object.keys(validateDemoImage(input, existing, false)).length) throw new Error('invalid_simulation_image')
  const review = !options.invalidateReview && input.sampleId === existing.image?.sampleId && input.knee === existing.image?.knee && input.sideConfirmed ? existing.review : null
  simulations.value[id] = { ...simulationFor(existing), image: { ...input }, review, status: review ? existing.status : 'draft', progressStep: review ? existing.progressStep : 2 }
  return evaluations.value.find(item => item.id === id)!
}
export function createDemoReviewInput(existing: Evaluation): DemoReviewInput {
  const crop = getDemoCrop(existing.image?.sampleId ?? '', existing.knee)
  return existing.review && existing.review.sourceKey === crop?.id && existing.image?.sideConfirmed
    ? { decision: existing.review.decision, reason: existing.review.reason, reviewed: existing.review.reviewed }
    : { decision: 'pending', reason: '', reviewed: false }
}
export function saveDemoReview(input: DemoReviewInput, id: string, options: { reachSummary?: boolean } = {}): Evaluation {
  const existing = evaluations.value.find(item => item.id === id)
  if (!existing) throw new Error('server_case_required')
  const crop = getDemoCrop(existing.image?.sampleId ?? '', existing.knee)
  if (!crop || Object.keys(validateDemoImage(createDemoImageInput(existing), existing, true)).length) throw new Error('confirmed_simulation_image_required')
  if (!['pending', 'accepted', 'rejected'].includes(input.decision) || (input.decision === 'accepted' && !input.reviewed)) throw new Error('simulation_decision_required')
  if ((input.decision === 'rejected' || input.reason) && !demoRejectReasons.some(reason => reason.id === input.reason)) throw new Error('simulation_reason_required')
  if (options.reachSummary && input.decision !== 'accepted') throw new Error('simulation_acceptance_required')
  simulations.value[id] = {
    ...simulationFor(existing), review: { ...input, reason: input.decision === 'accepted' ? '' : input.reason, sourceKey: crop.id, decidedAt: input.decision === 'pending' ? null : new Date().toISOString() },
    status: input.decision === 'accepted' ? 'crop_accepted' : input.decision === 'rejected' ? 'crop_rejected' : 'draft', progressStep: options.reachSummary ? 4 : 3,
  }
  return evaluations.value.find(item => item.id === id)!
}
export function getDemoSummaryState(existing: Evaluation) {
  const baselinesComplete = Boolean(existing.knee)
  const imageComplete = !Object.keys(validateDemoImage(createDemoImageInput(existing), existing, true)).length
  const crop = getDemoCrop(existing.image?.sampleId ?? '', existing.knee)
  let reviewState: 'missing' | 'pending' | 'accepted' | 'rejected' | 'outdated' = 'missing'
  if (existing.review) reviewState = !imageComplete || existing.review.sourceKey !== crop?.id ? 'outdated' : existing.review.decision
  return { baselinesComplete, imageComplete, reviewState, preparationComplete: baselinesComplete && imageComplete && reviewState === 'accepted' }
}
