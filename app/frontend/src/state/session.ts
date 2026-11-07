import { ref } from 'vue'
import { apiRequest, ApiError } from '../api/http'
import { expectedStatus, parseSession, type SessionDto } from '../api/contracts'

export const serverSession = ref<SessionDto | null>(null)
export const sessionIssue = ref('')
export const sessionGeneration = ref(0)
let checking: Promise<SessionDto> | null = null
let checkingIntent = -1
let sessionIntent = 0

function changedIntent(): ApiError {
  return new ApiError({ status: 0, code: 'SESSION_CHANGED', message: 'La intención de acceso cambió. Se descartó la respuesta de la operación anterior.' })
}
function assertCurrentIntent(intent: number): void {
  if (intent !== sessionIntent) throw changedIntent()
}
function beginSessionIntent(): number {
  sessionIntent += 1
  checking = null
  checkingIntent = -1
  return sessionIntent
}

function accept(session: SessionDto): SessionDto {
  if (serverSession.value?.user?.id !== session.user?.id || serverSession.value?.authenticated !== session.authenticated) sessionGeneration.value += 1
  serverSession.value = session
  sessionIssue.value = ''
  return session
}
export function apiMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return 'La sesión caducó o no se pudo autenticar. Inicia sesión de nuevo; no se confirmó ningún cambio.'
    if (error.status === 403) return 'No se pudo autorizar la operación. Recupera la sesión antes de volver a intentarlo.'
    if (error.status === 412 || error.status === 428) return 'El caso cambió o falta su versión. Tus campos siguen en pantalla; vuelve a abrir el caso para recuperar la versión actual antes de guardar.'
    if (error.code === 'NETWORK_ERROR') return 'No pudimos conectar con el servidor. No se confirmó ningún cambio. Revisa la conexión e inténtalo otra vez.'
    return error.message
  }
  return 'No pudimos completar la operación. No se confirmó ningún cambio.'
}
export function refreshSession(): Promise<SessionDto> {
  if (checking && checkingIntent === sessionIntent) return checking
  const intent = sessionIntent
  let pending!: Promise<SessionDto>
  pending = (async () => {
    try {
      const response = await apiRequest('/session')
      assertCurrentIntent(intent)
      expectedStatus(response.status, 200)
      return accept(parseSession(response.data))
    } catch (error) {
      if (intent !== sessionIntent) throw changedIntent()
      sessionIssue.value = apiMessage(error)
      throw error
    } finally {
      if (checking === pending) { checking = null; checkingIntent = -1 }
    }
  })()
  checking = pending
  checkingIntent = intent
  return pending
}
export async function authenticate(email: string, password: string): Promise<void> {
  const intent = beginSessionIntent()
  const context = await refreshSession()
  assertCurrentIntent(intent)
  const response = await apiRequest('/session', { method: 'POST', csrfToken: context.csrfToken, body: { email, password } })
  assertCurrentIntent(intent)
  expectedStatus(response.status, 200)
  const session = parseSession(response.data)
  if (!session.authenticated) throw new ApiError({ status: 200, code: 'INVALID_RESPONSE', message: 'El servidor no confirmó una sesión autenticada.' })
  // Also invalidate reads started while the authentication request was in flight.
  beginSessionIntent()
  accept(session)
}
export function currentCsrf(): string {
  if (!serverSession.value?.authenticated) throw new ApiError({ status: 401, code: 'AUTHENTICATION_REQUIRED', message: 'Inicia sesión antes de guardar.' })
  return serverSession.value.csrfToken
}
export async function closeSession(): Promise<void> {
  const csrfToken = currentCsrf()
  const intent = beginSessionIntent()
  try {
    // Revocation is always requested, even when other session work was pending.
    const response = await apiRequest('/session', { method: 'DELETE', csrfToken })
    if (intent !== sessionIntent) return
    expectedStatus(response.status, 204)
  } catch (error) {
    if (intent !== sessionIntent) return
    throw error
  }
  beginSessionIntent()
  serverSession.value = null
  sessionGeneration.value += 1
  sessionIssue.value = ''
}
