import { computed, ref, watch } from 'vue'
import { apiRequest, ApiError } from '../api/http'
import { expectedStatus, parseEvaluation, parseEvaluationPage, strongEtag, type ClinicalDto, type EvaluationDto } from '../api/contracts'
import { apiMessage, currentCsrf, sessionGeneration } from './session'
import type { Evaluation } from '../demo/demoEvaluations'

export interface DemoDraftInput { code: string; knee: '' | 'right' | 'left'; age: string; sex: '' | 'F' | 'M'; bmi: string }
export interface EvaluationView extends Evaluation { preparationId: string | null; reviewId: string | null; revisionId: string }

const records = ref<EvaluationDto[]>([])
const etags = new Map<string, string>()
export const evaluationIssue = ref('')
export const listLoading = ref(false)
export const listFailed = ref(false)
export const legacyDraftsPresent = ref(false)
try { legacyDraftsPresent.value = localStorage.getItem('kneeoai:demo-evaluations:v1') !== null } catch { /* Never delete or replace a legacy copy. */ }

export const evaluations = computed<EvaluationView[]>(() => records.value.map(item => ({
  id: item.id, code: item.code, knee: item.clinical.side, age: item.clinical.age, sex: item.clinical.sex, bmi: item.clinical.bmi,
  updatedAt: item.updatedAt, status: item.reviewId ? 'review' : 'draft', progressStep: item.reviewId ? 4 : item.preparationId ? 2 : 1,
  image: null, review: null, preparationId: item.preparationId, reviewId: item.reviewId, revisionId: item.currentRevisionId,
})))
export function getCanonicalEvaluation(id: string): EvaluationDto | undefined { return records.value.find(item => item.id === id) }
export function getEvaluationEtag(id: string): string {
  const value = etags.get(id)
  if (!value) throw new ApiError({ status: 428, code: 'VERSION_REQUIRED', message: 'Recupera el caso antes de enviar una imagen.' })
  return value
}

export function resetWorkspace(): void {
  records.value = []; etags.clear(); evaluationIssue.value = ''; listFailed.value = false; listLoading.value = false
}
watch(sessionGeneration, resetWorkspace, { flush: 'sync' })
function assertCurrentSession(generation: number): void {
  if (generation !== sessionGeneration.value) {
    throw new ApiError({ status: 0, code: 'SESSION_CHANGED', message: 'La sesión cambió mientras se procesaba la solicitud. Se descartó la respuesta anterior.' })
  }
}
function accept(item: EvaluationDto, etag?: string): void {
  const previous = records.value.find(record => record.id === item.id)
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
    etags.delete(id)
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
    const prior = new Map(records.value.map(item => [item.id, item.currentRevisionId]))
    for (const item of items) if (prior.get(item.id) !== item.currentRevisionId) { etags.delete(item.id) }
    records.value = items.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  } catch (error) {
    if (generation !== sessionGeneration.value) throw error
    records.value = []; etags.clear(); listFailed.value = true
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
  if (requireSide && !input.knee) errors.knee = 'Selecciona la rodilla antes de cargar el DICOM.'
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
  return evaluations.value.find(record => record.id === item.id)!
}
