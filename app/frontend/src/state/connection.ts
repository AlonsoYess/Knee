import { ref } from 'vue'

export const pendingMutations = ref(0)
export const serverReachability = ref<'unknown' | 'available' | 'unavailable'>('unknown')
export const browserOnline = ref(typeof navigator === 'undefined' ? true : navigator.onLine)
let observationSequence = 0
let latestObservation = 0

export function beginApiObservation(): number { return ++observationSequence }
export function observeApiResponse(sequence: number, response: Response): void {
  if (sequence < latestObservation) return
  // A backend request ID distinguishes its response from a gateway/static fallback.
  if (response.headers.get('X-Request-Id')?.trim()) {
    latestObservation = sequence; serverReachability.value = 'available'
  } else if (response.status >= 500) {
    latestObservation = sequence; serverReachability.value = 'unavailable'
  }
}
export function observeApiNetworkFailure(sequence: number): void {
  if (sequence < latestObservation) return
  latestObservation = sequence; serverReachability.value = 'unavailable'
}
export function confirmServerConnection(): void {
  latestObservation = ++observationSequence; serverReachability.value = 'available'
}
export function updateBrowserOnline(): void { browserOnline.value = navigator.onLine }

/** Covers the entire response body, not merely receipt of response headers. */
export async function trackMutation<T>(method: string, operation: () => Promise<T>): Promise<T> {
  const mutation = ['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)
  if (mutation) pendingMutations.value += 1
  try { return await operation() }
  finally { if (mutation) pendingMutations.value -= 1 }
}
