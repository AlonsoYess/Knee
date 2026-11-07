import type { DemoImageState } from './demoImages'
import type { DemoReviewState } from './demoCrops'

export type EvaluationStatus = 'draft' | 'review' | 'rejected' | 'model_pending' | 'crop_accepted' | 'crop_rejected'

export interface Evaluation {
  id: string
  code: string
  knee: 'right' | 'left' | null
  updatedAt: string
  status: EvaluationStatus
  age: number | null
  sex: 'F' | 'M' | null
  bmi: number | null
  progressStep: 1 | 2 | 3 | 4
  image?: DemoImageState | null
  review?: DemoReviewState | null
}

// Entirely invented examples. No patient, cohort, image, or prediction is imported.
export const demoEvaluations: readonly Evaluation[] = [
  { id: 'demo-001', code: 'DEMO-001', knee: 'right', updatedAt: '2026-11-06T15:20:00Z', status: 'draft', age: 62, sex: 'F', bmi: 28.4, progressStep: 1 },
  { id: 'demo-002', code: 'DEMO-002', knee: 'left', updatedAt: '2026-11-06T14:45:00Z', status: 'review', age: 58, sex: 'M', bmi: 26.2, progressStep: 3 },
  { id: 'demo-003', code: 'DEMO-003', knee: 'right', updatedAt: '2026-11-05T20:10:00Z', status: 'model_pending', age: 67, sex: 'F', bmi: 30.1, progressStep: 4 },
  { id: 'demo-004', code: 'DEMO-004', knee: 'left', updatedAt: '2026-11-05T17:35:00Z', status: 'rejected', age: 64, sex: 'M', bmi: 27.8, progressStep: 3 },
  { id: 'demo-005', code: 'DEMO-005', knee: 'right', updatedAt: '2026-11-04T16:00:00Z', status: 'draft', age: 55, sex: 'F', bmi: 25.6, progressStep: 2 },
  { id: 'demo-006', code: 'DEMO-006', knee: 'left', updatedAt: '2026-11-04T14:25:00Z', status: 'review', age: 71, sex: 'F', bmi: 29.3, progressStep: 3 },
  { id: 'demo-007', code: 'DEMO-007', knee: 'right', updatedAt: '2026-11-03T19:40:00Z', status: 'model_pending', age: 60, sex: 'M', bmi: 27.1, progressStep: 4 },
]
