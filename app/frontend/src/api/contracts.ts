import { ApiError } from './http'

export type Side = 'right' | 'left'
export interface UserDto { id: string; displayName: string; role: 'doctor' | 'administrator' }
export interface SessionDto { authenticated: boolean; user: UserDto | null; expiresAt: string | null; csrfToken: string }
export interface ClinicalDto { age: number | null; sex: 'F' | 'M' | null; bmi: number | null; side: Side | null }
export interface EvaluationDto {
  id: string; code: string; currentRevisionId: string; clinical: ClinicalDto
  preparationId: string | null; reviewId: string | null; currentExecutionId: string | null; updatedAt: string
}
export interface EvaluationPageDto { items: EvaluationDto[]; nextCursor: string | null }

function invalid(): never {
  throw new ApiError({ status: 200, code: 'INVALID_RESPONSE', message: 'No pudimos verificar la respuesta del servidor. No se confirmó el guardado; recupera el caso antes de intentarlo otra vez.' })
}
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return invalid()
  return value as Record<string, unknown>
}
function text(value: unknown): string { return typeof value === 'string' && Boolean(value.trim()) ? value : invalid() }
function id(value: unknown): string { const parsed = text(value); return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(parsed) ? parsed : invalid() }
function nullableId(value: unknown): string | null { return value === null ? null : id(value) }
function timestamp(value: unknown): string { const parsed = text(value); return Number.isFinite(Date.parse(parsed)) ? parsed : invalid() }
function positive(value: unknown, integer = false): number | null {
  if (value === null) return null
  return typeof value === 'number' && Number.isFinite(value) && value > 0 && (!integer || Number.isInteger(value)) ? value : invalid()
}
export function parseSession(value: unknown): SessionDto {
  const data = record(value)
  if (typeof data.authenticated !== 'boolean') return invalid()
  const csrfToken = text(data.csrfToken)
  if (!data.authenticated) {
    if (data.user !== null || data.expiresAt !== null) return invalid()
    return { authenticated: false, user: null, expiresAt: null, csrfToken }
  }
  const user = record(data.user)
  if (user.role !== 'doctor' && user.role !== 'administrator') return invalid()
  return { authenticated: true, csrfToken, expiresAt: timestamp(data.expiresAt), user: { id: id(user.id), displayName: text(user.displayName), role: user.role } }
}
export function parseEvaluation(value: unknown): EvaluationDto {
  const data = record(value)
  const clinical = record(data.clinical)
  if (clinical.sex !== null && clinical.sex !== 'F' && clinical.sex !== 'M') return invalid()
  if (clinical.side !== null && clinical.side !== 'right' && clinical.side !== 'left') return invalid()
  return {
    id: id(data.id), code: text(data.code), currentRevisionId: id(data.currentRevisionId),
    clinical: { age: positive(clinical.age, true), sex: clinical.sex, bmi: positive(clinical.bmi), side: clinical.side },
    preparationId: nullableId(data.preparationId), reviewId: nullableId(data.reviewId), currentExecutionId: nullableId(data.currentExecutionId), updatedAt: timestamp(data.updatedAt),
  }
}
export function parseEvaluationPage(value: unknown): EvaluationPageDto {
  const data = record(value)
  if (!Array.isArray(data.items) || (data.nextCursor !== null && (typeof data.nextCursor !== 'string' || !data.nextCursor))) return invalid()
  const items = data.items.map(parseEvaluation)
  if (new Set(items.map(item => item.id)).size !== items.length) return invalid()
  return { items, nextCursor: data.nextCursor as string | null }
}
export function strongEtag(value: string | null): string { return value && /^"[^"\r\n]+"$/.test(value) ? value : invalid() }
export function expectedStatus(actual: number, expected: number): void { if (actual !== expected) invalid() }
