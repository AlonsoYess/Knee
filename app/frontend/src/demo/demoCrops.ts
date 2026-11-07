import { demoImages, type DemoImageId } from './demoImages'

export interface DemoReviewInput {
  decision: 'pending' | 'accepted' | 'rejected'
  reason: string
  reviewed: boolean
}
export interface DemoReviewState extends DemoReviewInput {
  sourceKey: string
  decidedAt: string | null
}
export interface DemoCropFixture {
  id: string
  sampleId: DemoImageId
  knee: 'right' | 'left'
  version: 'ui-crop-v1'
  sourceSrc: string
  box: { x: number; y: number; width: number; height: number }
  label: string
}

// Fixed viewports of original synthetic SVGs, for UI illustration only.
// These boxes are neither a localization algorithm nor scientific QC parameters.
export function getDemoCrop(sampleId: string, knee: string | null): DemoCropFixture | undefined {
  const image = demoImages.find(item => item.id === sampleId)
  if (!image || (knee !== 'right' && knee !== 'left')) return undefined
  const x = sampleId === 'bilateral-a' ? (knee === 'right' ? 250 : 535) : (knee === 'right' ? 230 : 553)
  return {
    id: `${image.id}:${knee}:ui-crop-v1`, sampleId: image.id, knee,
    version: 'ui-crop-v1', sourceSrc: image.src,
    box: { x, y: 275, width: 225, height: 225 },
    label: `${image.label} · ${knee === 'right' ? 'Derecha del paciente' : 'Izquierda del paciente'}`,
  }
}

// Wording for exercising the rejection form; not an adopted QC checklist.
export const demoRejectReasons = [
  { id: 'incomplete-example', label: 'Región incompleta en el ejemplo' },
  { id: 'obstructed-example', label: 'Marcas que dificultan revisar el ejemplo' },
  { id: 'unclear-side-example', label: 'Lado no identificable en el ejemplo' },
  { id: 'other-example', label: 'Otro problema del ejemplo' },
] as const
