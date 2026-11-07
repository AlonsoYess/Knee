import { ref } from 'vue'
import { demoEvaluations, type Evaluation, type EvaluationStatus } from './demoEvaluations'
import { demoImages, type DemoImageState } from './demoImages'
import { demoRejectReasons, type DemoReviewState } from './demoCrops'

export const DEMO_EVALUATIONS_KEY = 'kneeoai:demo-evaluations:v1'
export const demoStorageIssue = ref<{
  kind: 'read' | 'invalid' | 'write' | 'conflict'
  message: string
} | null>(null)

interface SavedDemoEvaluations {
  schemaVersion: 1
  fixtureVersion: 'ui-fixtures-v1'
  revision: number
  savedAt: string
  evaluations: Evaluation[]
}

const fixtureVersion = 'ui-fixtures-v1'
const fixtureById = new Map(demoEvaluations.map(item => [item.id, item]))
const largestFixtureCode = Math.max(...demoEvaluations.map(item => Number(item.code.slice(5))))
const imageIds = new Set<string>(demoImages.map(item => item.id))
const rejectionReasons = new Set<string>(demoRejectReasons.map(item => item.id))
const statuses = new Set<EvaluationStatus>(['draft', 'review', 'rejected', 'model_pending', 'crop_accepted', 'crop_rejected'])
let lastRaw: string | null | undefined
let revision = 0
let storageState: 'uninitialized' | 'ready' | 'protected' | 'conflict' = 'uninitialized'

function invalid(): never { throw new Error('invalid_demo_snapshot') }
function objectWithKeys(value: unknown, required: readonly string[], optional: readonly string[] = []): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return invalid()
  const record = value as Record<string, unknown>
  const allowed = new Set([...required, ...optional])
  if (required.some(key => !Object.prototype.hasOwnProperty.call(record, key)) || Object.keys(record).some(key => !allowed.has(key))) return invalid()
  return record
}
function validDate(value: unknown): value is string {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(value)) return false
  const parsed = new Date(value)
  return Number.isFinite(parsed.getTime()) && parsed.toISOString().replace('.000Z', 'Z') === value.replace('.000Z', 'Z')
}
function positiveNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}
function baselineComplete(item: Evaluation): boolean {
  return item.age !== null && item.sex !== null && item.bmi !== null && item.knee !== null
}
function parseImage(value: unknown, knee: Evaluation['knee']): DemoImageState | null {
  if (value === null) return null
  const record = objectWithKeys(value, ['sampleId', 'knee', 'sideConfirmed'])
  if (typeof record.sampleId !== 'string' || (record.sampleId !== '' && !imageIds.has(record.sampleId))) return invalid()
  if (record.knee !== '' && record.knee !== 'right' && record.knee !== 'left') return invalid()
  if (record.knee !== '' && record.knee !== knee) return invalid()
  if (typeof record.sideConfirmed !== 'boolean') return invalid()
  if (record.sideConfirmed && (!record.sampleId || !record.knee || record.knee !== knee)) return invalid()
  return { sampleId: record.sampleId as DemoImageState['sampleId'], knee: record.knee, sideConfirmed: record.sideConfirmed }
}
function parseReview(value: unknown): DemoReviewState | null {
  if (value === null) return null
  const record = objectWithKeys(value, ['decision', 'reason', 'reviewed', 'sourceKey', 'decidedAt'])
  if (record.decision !== 'pending' && record.decision !== 'accepted' && record.decision !== 'rejected') return invalid()
  if (typeof record.reason !== 'string' || (record.reason !== '' && !rejectionReasons.has(record.reason))) return invalid()
  if (typeof record.reviewed !== 'boolean') return invalid()
  // Preserve the recorded source/version. A previous version remains outdated in the summary.
  if (typeof record.sourceKey !== 'string' || !/^bilateral-[ab]:(?:right|left):ui-crop-v\d{1,9}$/.test(record.sourceKey)) return invalid()
  if (record.decision === 'pending' ? record.decidedAt !== null : !validDate(record.decidedAt)) return invalid()
  if (record.decision === 'accepted' && (!record.reviewed || record.reason !== '')) return invalid()
  if (record.decision === 'rejected' && !rejectionReasons.has(record.reason)) return invalid()
  return {
    decision: record.decision, reason: record.reason, reviewed: record.reviewed,
    sourceKey: record.sourceKey, decidedAt: record.decidedAt as string | null,
  }
}
function parseEvaluation(value: unknown): Evaluation {
  const record = objectWithKeys(value,
    ['id', 'code', 'knee', 'updatedAt', 'status', 'age', 'sex', 'bmi', 'progressStep'], ['image', 'review'])
  if (typeof record.id !== 'string' || typeof record.code !== 'string' || !/^DEMO-\d{3,}$/.test(record.code)) return invalid()
  const codeNumber = Number(record.code.slice(5))
  if (!Number.isSafeInteger(codeNumber) || codeNumber <= 0 || codeNumber >= Number.MAX_SAFE_INTEGER) return invalid()
  const fixture = fixtureById.get(record.id)
  if (fixture ? record.code !== fixture.code : !/^draft-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(record.id) || codeNumber <= largestFixtureCode) return invalid()
  if (record.knee !== null && record.knee !== 'right' && record.knee !== 'left') return invalid()
  if (record.sex !== null && record.sex !== 'F' && record.sex !== 'M') return invalid()
  if (record.age !== null && (!positiveNumber(record.age) || !Number.isInteger(record.age))) return invalid()
  if (record.bmi !== null && !positiveNumber(record.bmi)) return invalid()
  if (!validDate(record.updatedAt) || !statuses.has(record.status as EvaluationStatus)) return invalid()
  if (record.progressStep !== 1 && record.progressStep !== 2 && record.progressStep !== 3 && record.progressStep !== 4) return invalid()
  const item: Evaluation = {
    id: record.id, code: record.code, knee: record.knee, updatedAt: record.updatedAt,
    status: record.status as EvaluationStatus, age: record.age as number | null,
    sex: record.sex, bmi: record.bmi as number | null, progressStep: record.progressStep,
  }
  if (Object.prototype.hasOwnProperty.call(record, 'image')) item.image = parseImage(record.image, item.knee)
  if (Object.prototype.hasOwnProperty.call(record, 'review')) item.review = parseReview(record.review)
  if (fixture && fixture.status !== 'draft') {
    if (JSON.stringify(item) !== JSON.stringify(fixture)) return invalid()
    return item
  }
  if (item.review) {
    if (!baselineComplete(item) || !item.image?.sideConfirmed) return invalid()
    const decision = item.review.decision
    if (decision === 'pending' && (item.status !== 'draft' || item.progressStep !== 3)) return invalid()
    if (decision === 'accepted' && (item.status !== 'crop_accepted' || (item.progressStep !== 3 && item.progressStep !== 4))) return invalid()
    if (decision === 'rejected' && (item.status !== 'crop_rejected' || item.progressStep !== 3)) return invalid()
  } else {
    if (item.status !== 'draft' || (item.progressStep !== 1 && item.progressStep !== 2)) return invalid()
    if (item.progressStep === 2 && !baselineComplete(item)) return invalid()
  }
  return item
}
function parseEvaluations(value: unknown): Evaluation[] {
  if (!Array.isArray(value)) return invalid()
  const items = value.map(parseEvaluation)
  const ids = new Set(items.map(item => item.id))
  const codes = new Set(items.map(item => Number(item.code.slice(5))))
  if (ids.size !== items.length || codes.size !== items.length || demoEvaluations.some(item => !ids.has(item.id))) return invalid()
  return items
}
function parseSnapshot(value: unknown): SavedDemoEvaluations {
  const record = objectWithKeys(value, ['schemaVersion', 'fixtureVersion', 'revision', 'savedAt', 'evaluations'])
  if (record.schemaVersion !== 1 || record.fixtureVersion !== fixtureVersion) return invalid()
  if (typeof record.revision !== 'number' || !Number.isSafeInteger(record.revision) || record.revision < 0 || record.revision >= Number.MAX_SAFE_INTEGER) return invalid()
  if (!validDate(record.savedAt)) return invalid()
  return {
    schemaVersion: 1, fixtureVersion, revision: record.revision, savedAt: record.savedAt,
    evaluations: parseEvaluations(record.evaluations),
  }
}
function freshFixtures(): Evaluation[] {
  return JSON.parse(JSON.stringify(demoEvaluations)) as Evaluation[]
}

export function loadSavedDemoEvaluations(): Evaluation[] {
  let raw: string | null
  try { raw = globalThis.localStorage.getItem(DEMO_EVALUATIONS_KEY) } catch {
    storageState = 'protected'
    lastRaw = undefined
    demoStorageIssue.value = { kind: 'read', message: 'No pudimos acceder al guardado local. Se muestran los ejemplos iniciales y no se guardarán cambios para proteger cualquier copia anterior.' }
    return freshFixtures()
  }
  lastRaw = raw
  revision = 0
  if (raw === null) {
    storageState = 'ready'
    demoStorageIssue.value = null
    return freshFixtures()
  }
  try {
    const snapshot = parseSnapshot(JSON.parse(raw) as unknown)
    revision = snapshot.revision
    storageState = 'ready'
    demoStorageIssue.value = null
    return snapshot.evaluations
  } catch {
    storageState = 'protected'
    demoStorageIssue.value = { kind: 'invalid', message: 'No pudimos recuperar la copia local porque sus datos o su versión no son compatibles. La copia se conserva sin reemplazar; se muestran los ejemplos iniciales y el guardado está bloqueado.' }
    return freshFixtures()
  }
}

export function persistDemoEvaluations(next: readonly Evaluation[]): void {
  if (storageState === 'uninitialized') loadSavedDemoEvaluations()
  if (storageState === 'protected') throw new Error('demo_storage_protected')
  if (storageState === 'conflict') throw new Error('demo_storage_conflict')
  let raw: string
  try {
    const snapshot = parseSnapshot({
      schemaVersion: 1, fixtureVersion, revision: revision + 1,
      savedAt: new Date().toISOString(), evaluations: next,
    })
    raw = JSON.stringify(snapshot)
  } catch {
    demoStorageIssue.value = { kind: 'write', message: 'Los cambios no se guardaron porque el estado del ejemplo no pudo verificarse. La copia anterior se conserva; revisa el caso antes de intentarlo otra vez.' }
    throw new Error('demo_storage_invalid_state')
  }
  try {
    if (globalThis.localStorage.getItem(DEMO_EVALUATIONS_KEY) !== lastRaw) {
      storageState = 'conflict'
      demoStorageIssue.value = { kind: 'conflict', message: 'La copia local cambió en otra pestaña o ventana. No se sobrescribió. Recarga para recuperar esa copia; antes revisa los cambios pendientes de esta pantalla.' }
      throw new Error('demo_storage_conflict')
    }
    globalThis.localStorage.setItem(DEMO_EVALUATIONS_KEY, raw)
  } catch (error) {
    if (storageState === 'conflict') throw error
    demoStorageIssue.value = { kind: 'write', message: 'No pudimos guardar los cambios en este navegador. La copia anterior se conserva y los cambios de esta pantalla siguen pendientes. Puedes intentar guardarlos otra vez.' }
    throw new Error('demo_storage_write_failed')
  }
  lastRaw = raw
  revision += 1
  demoStorageIssue.value = null
}
