import { ref } from 'vue'

// This client-side flag only supports the mock UI. It grants no server access.
const STORAGE_KEY = 'kneeoai:ui-demo:v1'
export const DEMO_EMAIL = 'medico@kneeoai.invalid'
export const DEMO_PASSWORD = 'DemoKnee2026!'

function restored(): boolean {
  try { return sessionStorage.getItem(STORAGE_KEY) === 'active' } catch { return false }
}

export const demoSessionActive = ref(restored())

export function enterDemo(): void {
  demoSessionActive.value = true
  try { sessionStorage.setItem(STORAGE_KEY, 'active') } catch { /* Memory-only fallback. */ }
}

export function leaveDemo(): void {
  demoSessionActive.value = false
  try { sessionStorage.removeItem(STORAGE_KEY) } catch { /* Memory state is already cleared. */ }
}
