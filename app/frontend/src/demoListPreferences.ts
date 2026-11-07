import { reactive } from 'vue'
import type { EvaluationStatus } from './demo/demoEvaluations'

// UI preferences only: kept in memory while navigating and cleared on demo logout.
export const demoListPreferences = reactive({
  search: '',
  statusFilter: 'all' as EvaluationStatus | 'all',
})

export function resetDemoListPreferences() {
  demoListPreferences.search = ''
  demoListPreferences.statusFilter = 'all'
}
