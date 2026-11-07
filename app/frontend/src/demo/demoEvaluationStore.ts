import { ref } from 'vue'
import type { Evaluation } from './demoEvaluations'
import { demoImages, type DemoImageState } from './demoImages'
import { getDemoCrop, demoRejectReasons, type DemoReviewInput } from './demoCrops'
import { loadSavedDemoEvaluations, persistDemoEvaluations } from './demoEvaluationPersistence'

export type DemoImageInput = DemoImageState
export type { DemoReviewInput } from './demoCrops'

export function isEditableDemoEvaluation(existing: Evaluation | undefined): boolean {
  return Boolean(existing && (existing.status === 'draft' || ((existing.status === 'crop_accepted' || existing.status === 'crop_rejected') && existing.review)))
}

// Fictional cases only. Saved explicitly in this browser; no files, cohort or network calls.
export const evaluations = ref<Evaluation[]>(loadSavedDemoEvaluations())

function commitDemoEvaluations(next: Evaluation[]): void {
  persistDemoEvaluations(next)
  evaluations.value = next
}

export interface DemoDraftInput {
  code: string
  knee: '' | 'right' | 'left'
  age: string
  sex: '' | 'F' | 'M'
  bmi: string
}

export function createDraftInput(existing?: Evaluation): DemoDraftInput {
  const largest = Math.max(0, ...evaluations.value.map(item => Number(item.code.match(/^DEMO-(\d+)$/)?.[1] ?? 0)))
  return {
    code: existing?.code ?? `DEMO-${String(largest + 1).padStart(3, '0')}`,
    knee: existing?.knee ?? '',
    age: existing?.age == null ? '' : String(existing.age),
    sex: existing?.sex ?? '',
    bmi: existing?.bmi == null ? '' : String(existing.bmi),
  }
}

function parseDecimal(value: string): number | null {
  if (!value.trim()) return null
  const normalized = value.trim().replace(',', '.')
  return /^\d+(?:\.\d+)?$/.test(normalized) ? Number(normalized) : NaN
}

export function validateDemoDraft(input: DemoDraftInput, requireComplete: boolean): Partial<Record<keyof DemoDraftInput, string>> {
  const errors: Partial<Record<keyof DemoDraftInput, string>> = {}
  if (!/^DEMO-\d{3,}$/.test(input.code)) errors.code = 'El código debe corresponder a un caso de demostración.'
  const age = parseDecimal(input.age)
  const bmi = parseDecimal(input.bmi)
  if (age === null) { if (requireComplete) errors.age = 'Escribe la edad en la fecha de la radiografía.' }
  else if (!Number.isFinite(age) || !Number.isInteger(age) || age <= 0) errors.age = 'Escribe una edad entera mayor que cero.'
  if (bmi === null) { if (requireComplete) errors.bmi = 'Escribe el IMC basal.' }
  else if (!Number.isFinite(bmi) || bmi <= 0) errors.bmi = 'Escribe un IMC numérico mayor que cero.'
  if (input.sex === '') { if (requireComplete) errors.sex = 'Selecciona el sexo registrado.' }
  else if (input.sex !== 'F' && input.sex !== 'M') errors.sex = 'Selecciona una opción disponible.'
  if (input.knee === '') { if (requireComplete) errors.knee = 'Selecciona la rodilla que se evaluará.' }
  else if (input.knee !== 'right' && input.knee !== 'left') errors.knee = 'Selecciona una opción disponible.'
  return errors
}

export function saveDemoDraft(input: DemoDraftInput, existingId?: string, options: { invalidateImageSide?: boolean } = {}): Evaluation {
  if (Object.keys(validateDemoDraft(input, false)).length) throw new Error('draft_validation_required')
  const existing = existingId ? evaluations.value.find(item => item.id === existingId) : undefined
  if (existingId && !isEditableDemoEvaluation(existing)) throw new Error('editable_demo_draft_required')
  if (evaluations.value.some(item => item.code === input.code && item.id !== existingId)) throw new Error('unique_demo_code_required')
  const updated: Evaluation = {
    id: existing?.id ?? `draft-${crypto.randomUUID()}`,
    code: input.code,
    knee: input.knee || null,
    age: parseDecimal(input.age),
    sex: input.sex || null,
    bmi: parseDecimal(input.bmi),
    updatedAt: new Date().toISOString(),
    status: 'draft',
    progressStep: existing && existing.progressStep >= 2 && !Object.keys(validateDemoDraft(input, true)).length ? 2 : 1,
    image: existing?.image ? {
      ...existing.image,
      knee: input.knee,
      sideConfirmed: !options.invalidateImageSide && existing.knee === input.knee && existing.image.sideConfirmed,
    } : null,
    review: options.invalidateImageSide || existing?.knee !== input.knee ? null : existing?.review ?? null,
  }
  if (updated.review && existing && !Object.keys(validateDemoDraft(input, true)).length) {
    updated.status = existing.status
    updated.progressStep = existing.progressStep
  }
  if (Object.keys(validateDemoDraft(input, true)).length) updated.review = null
  commitDemoEvaluations([updated, ...evaluations.value.filter(item => item.id !== updated.id)])
  return updated
}

export function createDemoImageInput(existing: Evaluation): DemoImageInput {
  return {
    sampleId: existing.image?.sampleId ?? '',
    knee: existing.image?.knee ?? existing.knee ?? '',
    sideConfirmed: Boolean(existing.image?.sampleId && existing.image.knee === existing.knee && existing.image.sideConfirmed),
  }
}

export function validateDemoImage(input: DemoImageInput, existing: Evaluation, requireComplete: boolean): Partial<Record<keyof DemoImageInput, string>> {
  const errors: Partial<Record<keyof DemoImageInput, string>> = {}
  if (Object.keys(validateDemoDraft(createDraftInput(existing), true)).length) errors.sampleId = 'Completa primero los datos basales.'
  if (input.sampleId && !demoImages.some(item => item.id === input.sampleId)) errors.sampleId = 'Selecciona una de las imágenes de demostración.'
  if (requireComplete && !input.sampleId) errors.sampleId = 'Selecciona una imagen de ejemplo.'
  if (input.knee && input.knee !== existing.knee) errors.knee = 'El lado no coincide con los datos basales. Revísalo o vuelve a esos datos.'
  if (requireComplete && !input.knee) errors.knee = 'Selecciona la rodilla del paciente.'
  if (requireComplete && !input.sideConfirmed) errors.sideConfirmed = 'Confirma expresamente el lado de este ejemplo.'
  if (input.sideConfirmed && (!input.sampleId || !input.knee || input.knee !== existing.knee)) errors.sideConfirmed = 'Selecciona la imagen y el lado correcto antes de confirmar.'
  return errors
}

export function saveDemoImage(input: DemoImageInput, id: string, options: { invalidateReview?: boolean } = {}): Evaluation {
  const existing = evaluations.value.find(item => item.id === id)
  if (!existing || !isEditableDemoEvaluation(existing)) throw new Error('editable_demo_draft_required')
  if (Object.keys(validateDemoImage(input, existing, false)).length) throw new Error('demo_image_validation_required')
  const updated: Evaluation = {
    ...existing,
    image: { ...input, sideConfirmed: Boolean(input.sampleId && input.knee === existing.knee && input.sideConfirmed) },
    updatedAt: new Date().toISOString(),
    progressStep: 2,
    status: 'draft',
    review: !options.invalidateReview && input.sampleId === existing.image?.sampleId && input.knee === existing.image?.knee && input.sideConfirmed ? existing.review ?? null : null,
  }
  if (updated.review) { updated.status = existing.status; updated.progressStep = existing.progressStep }
  commitDemoEvaluations([updated, ...evaluations.value.filter(item => item.id !== id)])
  return updated
}

export function createDemoReviewInput(existing: Evaluation): DemoReviewInput {
  const crop = getDemoCrop(existing.image?.sampleId ?? '', existing.knee)
  if (existing.review && crop?.id === existing.review.sourceKey && existing.image?.sideConfirmed) {
    return { decision: existing.review.decision, reason: existing.review.reason, reviewed: existing.review.reviewed }
  }
  return { decision: 'pending', reason: '', reviewed: false }
}

export function saveDemoReview(input: DemoReviewInput, id: string, options: { reachSummary?: boolean } = {}): Evaluation {
  const existing = evaluations.value.find(item => item.id === id)
  if (!existing || !isEditableDemoEvaluation(existing)) throw new Error('editable_demo_draft_required')
  const image = createDemoImageInput(existing)
  const crop = getDemoCrop(image.sampleId, image.knee)
  if (Object.keys(validateDemoImage(image, existing, true)).length || !crop) throw new Error('confirmed_demo_image_required')
  if (!['pending', 'accepted', 'rejected'].includes(input.decision)) throw new Error('demo_review_decision_required')
  if (input.decision === 'accepted' && !input.reviewed) throw new Error('explicit_demo_review_required')
  if (input.decision === 'rejected' && !demoRejectReasons.some(reason => reason.id === input.reason)) throw new Error('demo_rejection_reason_required')
  if (input.reason && !demoRejectReasons.some(reason => reason.id === input.reason)) throw new Error('demo_rejection_reason_required')
  const updated: Evaluation = {
    ...existing,
    review: { ...input, reason: input.decision === 'accepted' ? '' : input.reason, sourceKey: crop.id, decidedAt: input.decision === 'pending' ? null : new Date().toISOString() },
    status: input.decision === 'accepted' ? 'crop_accepted' : input.decision === 'rejected' ? 'crop_rejected' : 'draft',
    progressStep: 3,
    updatedAt: new Date().toISOString(),
  }
  if (options.reachSummary) {
    if (!getDemoSummaryState(updated).preparationComplete) throw new Error('accepted_demo_preparation_required')
    updated.progressStep = 4
  }
  commitDemoEvaluations([updated, ...evaluations.value.filter(item => item.id !== id)])
  return updated
}

export function getDemoSummaryState(existing: Evaluation): {
  baselinesComplete: boolean
  imageComplete: boolean
  reviewState: 'missing' | 'pending' | 'accepted' | 'rejected' | 'outdated'
  preparationComplete: boolean
} {
  const baselinesComplete = !Object.keys(validateDemoDraft(createDraftInput(existing), true)).length
  const imageComplete = !Object.keys(validateDemoImage(createDemoImageInput(existing), existing, true)).length
  const crop = getDemoCrop(existing.image?.sampleId ?? '', existing.knee)
  let reviewState: 'missing' | 'pending' | 'accepted' | 'rejected' | 'outdated' = 'missing'
  if (existing.review) {
    reviewState = !imageComplete || !crop || existing.review.sourceKey !== crop.id
      ? 'outdated'
      : existing.review.decision === 'accepted' && !existing.review.reviewed
        ? 'pending'
        : existing.review.decision
  }
  return { baselinesComplete, imageComplete, reviewState, preparationComplete: baselinesComplete && imageComplete && reviewState === 'accepted' }
}

export function markDemoSummaryReached(id: string): void {
  const existing = evaluations.value.find(item => item.id === id)
  if (!existing || !getDemoSummaryState(existing).preparationComplete) throw new Error('accepted_demo_preparation_required')
  commitDemoEvaluations(evaluations.value.map(item => item.id === id ? { ...item, progressStep: 4 as const } : item))
}
