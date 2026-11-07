import { ApiError } from './http'
import type { Side } from './contracts'

export type Point = [number, number]
export type Shape = [number, number]
export type SourceView = 'full_a' | 'full_b' | 'half_a' | 'half_b'
export type CropView = 'crop_a' | 'crop_b'
export type PreparationView = SourceView | CropView
export type Visibility = 'visible' | 'obscured' | 'uncertain'
export type Presence = 'absent' | 'present' | 'uncertain'
export type Mark = 'peripheral_ruler' | 'fine_line' | 'isolated_spot' | 'hardware' | 'text_or_mask'
export interface QcObservation {
  side_supported: boolean; coverage_complete: boolean
  visibility: { medial_joint_borders: Visibility; lateral_joint_borders: Visibility; subchondral_bone: Visibility }
  hardware: Presence; text_or_mask: Presence; marks: Mark[]
  mark_effect: 'none' | 'non_obscuring' | 'obscuring' | 'uncertain'
  physical_origin_known: boolean; note: string
  original_half_inspected: boolean; both_windows_inspected: boolean; outcome_blinded: boolean
  lateral_margins_visible: boolean; unclassified_mark_present: boolean
  ruler_outside_bone: boolean | null; ruler_locations_xy: Point[]
}
export interface ReferenceSubmission {
  preparationRevisionId: string; confirmedSide: Side; centerResolved: boolean; centerXY: Point | null
  originalAndHalfInspected: boolean; outcomeBlinded: boolean; note: string
}
export interface ReferenceDto extends ReferenceSubmission { id: string; preparationId: string; reviewerId: string; createdAt: string }
export interface ReviewSubmission {
  preparationRevisionId: string; confirmedSide: Side; decision: 'accepted' | 'rejected'; confirmedCheckIds: string[]; reason: string | null; observation: QcObservation | null
}
export interface QualityReviewDto extends ReviewSubmission {
  id: string; evaluationId: string; preparationId: string; image: { imageId: string; imageVersion: string }
  side: Side; qcPolicyVersion: string; reviewerId: string; createdAt: string
  assessment: { accepted: boolean; reasons: string[]; [key: string]: unknown } | null
}
export interface PreparationDto {
  id: string; evaluationId: string; sourceEvaluationRevisionId: string; preparationRevisionId: string
  image: { imageId: string; imageVersion: string }; side: Side
  status: 'queued' | 'processing' | 'awaiting_reference' | 'ready' | 'failed'
  pipelineVersion: string | null; createdAt: string; updatedAt: string; isCurrent: boolean
  error: { code: string; message: string; [key: string]: unknown } | null
  nativeHalfShape: Shape | null
  sourceContext: { views: Record<SourceView, string>; fullNativeShape: Shape } | null
  sourceReference: ReferenceDto | null
  cropGeometry: { nativeHalfBox: [number, number, number, number]; previewShapes: Record<CropView, Shape> } | null
  reviewContext: { qcPolicyVersion: string; canReview: boolean; blockerCodes: string[]; checks: { id: string; label: string; required: boolean }[]; originalPreviewUrl: string; cropPreviewUrl: string; views: Record<CropView, string> } | null
  activeReview: QualityReviewDto | null
}

function invalid(): never { throw new ApiError({ status: 200, code: 'INVALID_RESPONSE', message: 'La respuesta de preparación no corresponde al formato esperado. Recupera el caso; no se confirmó ningún cambio.' }) }
function obj(value: unknown): Record<string, unknown> { return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : invalid() }
function str(value: unknown): string { return typeof value === 'string' && value.trim() ? value : invalid() }
function id(value: unknown): string { const parsed = str(value); return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(parsed) ? parsed : invalid() }
function bool(value: unknown): boolean { return typeof value === 'boolean' ? value : invalid() }
function one<T extends string>(value: unknown, choices: readonly T[]): T { return typeof value === 'string' && choices.includes(value as T) ? value as T : invalid() }
function date(value: unknown): string { const parsed = str(value); return Number.isFinite(Date.parse(parsed)) ? parsed : invalid() }
function strings(value: unknown): string[] { return Array.isArray(value) ? value.map(str) : invalid() }
function point(value: unknown): Point { return Array.isArray(value) && value.length === 2 && value.every(n => typeof n === 'number' && Number.isFinite(n)) ? [value[0], value[1]] : invalid() }
function shape(value: unknown): Shape { const result = point(value); return result.every(n => Number.isInteger(n) && n > 0) ? result : invalid() }
function image(value: unknown) { const data = obj(value); return { imageId: id(data.imageId), imageVersion: str(data.imageVersion) } }
function viewUrl(value: unknown): string {
  const parsed = str(value)
  if (!parsed.startsWith('/api/v1/preparations/') || parsed.startsWith('//') || /[\\#\s\u0000-\u001f\u007f]/.test(parsed) || parsed.split('?', 1)[0]!.includes('%')) return invalid()
  const url = new URL(parsed, 'https://internal.invalid')
  if (url.origin !== 'https://internal.invalid' || !/^\/api\/v1\/preparations\/[A-Za-z0-9._:-]+\/views\/(full_a|full_b|half_a|half_b|crop_a|crop_b|original|crop)$/.test(url.pathname)) return invalid()
  return `${url.pathname}${url.search}`
}
export function parseReference(value: unknown): ReferenceDto {
  const data = obj(value)
  const resolved = bool(data.centerResolved)
  const center = data.centerXY === null ? null : point(data.centerXY)
  if ((resolved && (!center || !center.every(Number.isInteger))) || (!resolved && center !== null) || data.originalAndHalfInspected !== true || data.outcomeBlinded !== true || str(data.note).trim().length < 35) return invalid()
  return {
    id: id(data.id), preparationId: id(data.preparationId), preparationRevisionId: id(data.preparationRevisionId),
    confirmedSide: one(data.confirmedSide, ['right', 'left']), centerResolved: resolved, centerXY: center,
    originalAndHalfInspected: bool(data.originalAndHalfInspected), outcomeBlinded: bool(data.outcomeBlinded), note: str(data.note),
    reviewerId: id(data.reviewerId), createdAt: date(data.createdAt),
  }
}
export function parseObservation(value: unknown): QcObservation {
  const data = obj(value), visibility = obj(data.visibility)
  if (!Array.isArray(data.marks) || !Array.isArray(data.ruler_locations_xy)) return invalid()
  return {
    side_supported: bool(data.side_supported), coverage_complete: bool(data.coverage_complete),
    visibility: { medial_joint_borders: one(visibility.medial_joint_borders, ['visible', 'obscured', 'uncertain']), lateral_joint_borders: one(visibility.lateral_joint_borders, ['visible', 'obscured', 'uncertain']), subchondral_bone: one(visibility.subchondral_bone, ['visible', 'obscured', 'uncertain']) },
    hardware: one(data.hardware, ['absent', 'present', 'uncertain']), text_or_mask: one(data.text_or_mask, ['absent', 'present', 'uncertain']),
    marks: data.marks.map(mark => one(mark, ['peripheral_ruler', 'fine_line', 'isolated_spot', 'hardware', 'text_or_mask'])),
    mark_effect: one(data.mark_effect, ['none', 'non_obscuring', 'obscuring', 'uncertain']), physical_origin_known: bool(data.physical_origin_known), note: str(data.note),
    original_half_inspected: bool(data.original_half_inspected), both_windows_inspected: bool(data.both_windows_inspected), outcome_blinded: bool(data.outcome_blinded),
    lateral_margins_visible: bool(data.lateral_margins_visible), unclassified_mark_present: bool(data.unclassified_mark_present),
    ruler_outside_bone: data.ruler_outside_bone === null ? null : bool(data.ruler_outside_bone), ruler_locations_xy: data.ruler_locations_xy.map(point),
  }
}
export function parseQualityReview(value: unknown): QualityReviewDto {
  const data = obj(value), decision = one(data.decision, ['accepted', 'rejected'])
  const observation = data.observation === null ? null : parseObservation(data.observation)
  const assessmentData = data.assessment === null ? null : obj(data.assessment)
  const assessment = assessmentData ? { ...assessmentData, accepted: bool(assessmentData.accepted), reasons: strings(assessmentData.reasons) } : null
  if (decision === 'accepted' && (!observation || !assessment?.accepted)) return invalid()
  const side = one(data.side, ['right', 'left'])
  const reason = data.reason === null ? null : str(data.reason)
  if (decision === 'rejected' && !reason) return invalid()
  return {
    id: id(data.id), evaluationId: id(data.evaluationId), preparationId: id(data.preparationId), preparationRevisionId: id(data.preparationRevisionId),
    image: image(data.image), side, confirmedSide: side, qcPolicyVersion: str(data.qcPolicyVersion), decision,
    confirmedCheckIds: strings(data.confirmedCheckIds), reason, observation, assessment, reviewerId: id(data.reviewerId), createdAt: date(data.createdAt),
  }
}
export function parsePreparation(value: unknown): PreparationDto {
  const data = obj(value), status = one(data.status, ['queued', 'processing', 'awaiting_reference', 'ready', 'failed'])
  const source = data.sourceContext === null ? null : obj(data.sourceContext)
  const sourceViews = source ? obj(source.views) : null
  const review = data.reviewContext === null ? null : obj(data.reviewContext)
  const cropViews = review ? obj(review.views) : null
  const geometry = data.cropGeometry === null ? null : obj(data.cropGeometry)
  const previewShapes = geometry ? obj(geometry.previewShapes) : null
  const box = geometry?.nativeHalfBox
  if (geometry && (!Array.isArray(box) || box.length !== 4 || !box.every(n => typeof n === 'number' && Number.isFinite(n)) || box[2] <= box[0] || box[3] <= box[1])) return invalid()
  const errorData = data.error === null ? null : obj(data.error)
  const activeReview = data.activeReview === null ? null : parseQualityReview(data.activeReview)
  const sourceReference = data.sourceReference === null ? null : parseReference(data.sourceReference)
  const pipelineVersion = data.pipelineVersion === null ? null : str(data.pipelineVersion)
  if (status === 'ready' && (!review || !pipelineVersion || !sourceReference || errorData)) return invalid()
  if (status === 'awaiting_reference' && (!source || !data.nativeHalfShape || sourceReference?.centerResolved === true || review || activeReview || errorData)) return invalid()
  if (status === 'failed' && (!errorData || review || activeReview)) return invalid()
  if ((status === 'queued' || status === 'processing') && (review || activeReview || errorData)) return invalid()
  const checks = review ? (Array.isArray(review.checks) ? review.checks.map(value => { const check = obj(value); return { id: id(check.id), label: str(check.label), required: bool(check.required) } }) : invalid()) : []
  if (new Set(checks.map(check => check.id)).size !== checks.length) return invalid()
  const result: PreparationDto = {
    id: id(data.id), evaluationId: id(data.evaluationId), sourceEvaluationRevisionId: id(data.sourceEvaluationRevisionId), preparationRevisionId: id(data.preparationRevisionId), image: image(data.image), side: one(data.side, ['right', 'left']),
    status, pipelineVersion, createdAt: date(data.createdAt), updatedAt: date(data.updatedAt), isCurrent: bool(data.isCurrent),
    error: errorData ? { ...errorData, code: str(errorData.code), message: str(errorData.message) } : null,
    nativeHalfShape: data.nativeHalfShape === null ? null : shape(data.nativeHalfShape),
    sourceContext: source && sourceViews ? { fullNativeShape: shape(source.fullNativeShape), views: { full_a: viewUrl(sourceViews.full_a), full_b: viewUrl(sourceViews.full_b), half_a: viewUrl(sourceViews.half_a), half_b: viewUrl(sourceViews.half_b) } } : null,
    sourceReference,
    cropGeometry: geometry && previewShapes ? { nativeHalfBox: box as [number, number, number, number], previewShapes: { crop_a: shape(previewShapes.crop_a), crop_b: shape(previewShapes.crop_b) } } : null,
    reviewContext: review && cropViews ? { qcPolicyVersion: str(review.qcPolicyVersion), canReview: bool(review.canReview), blockerCodes: strings(review.blockerCodes), checks, originalPreviewUrl: viewUrl(review.originalPreviewUrl), cropPreviewUrl: viewUrl(review.cropPreviewUrl), views: { crop_a: viewUrl(cropViews.crop_a), crop_b: viewUrl(cropViews.crop_b) } } : null,
    activeReview,
  }
  if (sourceReference && (sourceReference.preparationId !== result.id || sourceReference.confirmedSide !== result.side)) return invalid()
  if (activeReview && (activeReview.preparationId !== result.id || activeReview.evaluationId !== result.evaluationId || activeReview.preparationRevisionId !== result.preparationRevisionId || activeReview.image.imageId !== result.image.imageId || activeReview.image.imageVersion !== result.image.imageVersion || activeReview.side !== result.side || activeReview.qcPolicyVersion !== result.reviewContext?.qcPolicyVersion)) return invalid()
  return result
}
