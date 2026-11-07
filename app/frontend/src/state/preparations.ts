import { ref, watch } from 'vue'
import { apiBlobRequest, apiMultipartRequest, apiRequest, ApiError } from '../api/http'
import { expectedStatus, strongEtag } from '../api/contracts'
import { parsePreparation, parseQualityReview, parseReference, type PreparationDto, type PreparationView, type QualityReviewDto, type ReferenceDto, type ReferenceSubmission, type ReviewSubmission } from '../api/preparationContracts'
import { currentCsrf, sessionGeneration } from './session'
import { getCanonicalEvaluation, getEvaluationEtag, loadEvaluation } from './evaluations'

export interface StoredPreparation { data: PreparationDto; etag: string }
export const preparationRecords = ref<Record<string, StoredPreparation>>({})
export function getActivePreparation(evaluationId: string): PreparationDto | undefined {
  const stored = preparationRecords.value[evaluationId]?.data, evaluation = getCanonicalEvaluation(evaluationId)
  return stored && evaluation?.preparationId === stored.id && evaluation.clinical.side === stored.side && stored.isCurrent ? stored : undefined
}
const requests = new Map<string, number>()
function changed(): never { throw new ApiError({ status: 0, code: 'INPUT_CHANGED', message: 'La sesión o la preparación cambió. Se descartó la respuesta anterior; recupera el caso actual.' }) }
function begin(evaluationId: string) { const request = (requests.get(evaluationId) ?? 0) + 1; requests.set(evaluationId, request); return { generation: sessionGeneration.value, request } }
function check(evaluationId: string, operation: ReturnType<typeof begin>) { if (operation.generation !== sessionGeneration.value || requests.get(evaluationId) !== operation.request) changed() }
function active(evaluationId: string, preparation: PreparationDto) {
  const evaluation = getCanonicalEvaluation(evaluationId)
  if (!evaluation || evaluation.preparationId !== preparation.id || preparation.evaluationId !== evaluationId || preparation.side !== evaluation.clinical.side || !preparation.isCurrent) changed()
}
watch(sessionGeneration, () => { preparationRecords.value = {}; requests.clear() }, { flush: 'sync' })

export async function loadActivePreparation(evaluationId: string, signal?: AbortSignal): Promise<PreparationDto | null> {
  const operation = begin(evaluationId)
  const preparationId = getCanonicalEvaluation(evaluationId)?.preparationId
  if (!preparationId) { delete preparationRecords.value[evaluationId]; return null }
  const response = await apiRequest(`/preparations/${preparationId}`, { signal })
  check(evaluationId, operation); expectedStatus(response.status, 200)
  const preparation = parsePreparation(response.data)
  if (preparation.id !== preparationId) changed()
  active(evaluationId, preparation)
  preparationRecords.value[evaluationId] = { data: preparation, etag: strongEtag(response.etag) }
  return preparation
}
function current(evaluationId: string): StoredPreparation {
  const stored = preparationRecords.value[evaluationId]
  if (!stored) throw new ApiError({ status: 428, code: 'VERSION_REQUIRED', message: 'Recupera la preparación actual antes de continuar.' })
  active(evaluationId, stored.data)
  return stored
}
export async function uploadPreparation(evaluationId: string, file: File, idempotencyKey: string, signal?: AbortSignal): Promise<PreparationDto> {
  const operation = begin(evaluationId)
  const form = new FormData(); form.set('file', file)
  const response = await apiMultipartRequest(`/evaluations/${evaluationId}/preparations`, form, { csrfToken: currentCsrf(), ifMatch: getEvaluationEtag(evaluationId), idempotencyKey, signal })
  check(evaluationId, operation)
  if (response.status !== 200 && response.status !== 202) expectedStatus(response.status, 202)
  const preparation = parsePreparation(response.data)
  if (preparation.evaluationId !== evaluationId) changed()
  await loadEvaluation(evaluationId); check(evaluationId, operation)
  active(evaluationId, preparation)
  preparationRecords.value[evaluationId] = { data: preparation, etag: strongEtag(response.etag) }
  return preparation
}
export async function sealSourceReference(evaluationId: string, payload: ReferenceSubmission, idempotencyKey: string, signal?: AbortSignal): Promise<ReferenceDto> {
  const stored = current(evaluationId), operation = begin(evaluationId)
  if (payload.preparationRevisionId !== stored.data.preparationRevisionId) changed()
  const response = await apiRequest(`/preparations/${stored.data.id}/reference`, { method: 'POST', body: payload, csrfToken: currentCsrf(), ifMatch: stored.etag, idempotencyKey, signal })
  check(evaluationId, operation)
  if (response.status !== 200 && response.status !== 201) expectedStatus(response.status, 201)
  const reference = parseReference(response.data)
  if (reference.preparationId !== stored.data.id || reference.preparationRevisionId !== payload.preparationRevisionId || reference.confirmedSide !== payload.confirmedSide) changed()
  return reference
}
export async function retryPreparation(evaluationId: string, idempotencyKey: string, signal?: AbortSignal): Promise<PreparationDto> {
  const stored = current(evaluationId), operation = begin(evaluationId)
  const response = await apiRequest(`/preparations/${stored.data.id}/retry`, { method: 'POST', body: {}, csrfToken: currentCsrf(), ifMatch: stored.etag, idempotencyKey, signal })
  check(evaluationId, operation)
  if (response.status !== 200 && response.status !== 202) expectedStatus(response.status, 202)
  const preparation = parsePreparation(response.data); active(evaluationId, preparation)
  preparationRecords.value[evaluationId] = { data: preparation, etag: strongEtag(response.etag) }
  return preparation
}
export async function recordQualityReview(evaluationId: string, payload: ReviewSubmission, idempotencyKey: string, signal?: AbortSignal): Promise<QualityReviewDto> {
  const stored = current(evaluationId), operation = begin(evaluationId)
  if (payload.preparationRevisionId !== stored.data.preparationRevisionId || payload.confirmedSide !== stored.data.side) changed()
  const response = await apiRequest(`/preparations/${stored.data.id}/reviews`, { method: 'POST', body: payload, csrfToken: currentCsrf(), ifMatch: stored.etag, idempotencyKey, signal })
  check(evaluationId, operation)
  if (response.status !== 200 && response.status !== 201) expectedStatus(response.status, 201)
  const review = parseQualityReview(response.data)
  if (review.evaluationId !== evaluationId || review.preparationId !== stored.data.id || review.preparationRevisionId !== stored.data.preparationRevisionId || review.decision !== payload.decision || review.side !== stored.data.side) changed()
  return review
}
export async function loadPreparationImage(preparation: PreparationDto, view: PreparationView, signal?: AbortSignal): Promise<Blob> {
  const generation = sessionGeneration.value
  const url = view.startsWith('crop_') ? preparation.reviewContext?.views[view as 'crop_a' | 'crop_b'] : preparation.sourceContext?.views[view as 'full_a' | 'full_b' | 'half_a' | 'half_b']
  if (!url) throw new ApiError({ status: 409, code: 'VIEW_NOT_READY', message: 'Esta vista todavía no está disponible.' })
  const parsed = new URL(url, 'https://internal.invalid')
  if (parsed.pathname !== `/api/v1/preparations/${preparation.id}/views/${view}` || parsed.searchParams.getAll('revision').length !== 1 || parsed.searchParams.get('revision') !== preparation.preparationRevisionId || [...parsed.searchParams.keys()].some(key => key !== 'revision')) changed()
  if (view.startsWith('crop_') && (!preparation.sourceReference || preparation.status !== 'ready')) changed()
  const response = await apiBlobRequest(url, { signal })
  if (generation !== sessionGeneration.value) changed()
  active(preparation.evaluationId, preparation)
  const latest = preparationRecords.value[preparation.evaluationId]?.data
  if (!latest || latest.id !== preparation.id || latest.preparationRevisionId !== preparation.preparationRevisionId) changed()
  return response.data
}
