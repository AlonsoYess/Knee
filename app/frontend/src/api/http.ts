import { beginApiObservation, observeApiNetworkFailure, observeApiResponse, trackMutation } from '../state/connection'

export const API_PREFIX = '/api/v1'

export type ApiMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'

export interface ApiFieldError {
  path: string
  code: string
  message: string
}

export interface ApiRequestOptions {
  method?: ApiMethod
  body?: unknown
  csrfToken?: string
  ifMatch?: string
  idempotencyKey?: string
  signal?: AbortSignal
}
export type ApiMultipartOptions = Omit<ApiRequestOptions, 'body' | 'method'> & { method?: Exclude<ApiMethod, 'GET'> }

export interface ApiResponse<T> {
  data: T | undefined
  status: number
  requestId: string | null
  etag: string | null
  retryAfter: string | null
}

interface ApiErrorDetails {
  status: number
  code: string
  message: string
  requestId?: string | null
  fields?: ApiFieldError[]
  retryable?: boolean
  etag?: string | null
  retryAfter?: string | null
}

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly requestId: string | null
  readonly fields: ApiFieldError[]
  readonly retryable: boolean
  readonly etag: string | null
  readonly retryAfter: string | null

  constructor(details: ApiErrorDetails) {
    super(details.message)
    this.name = 'ApiError'
    this.status = details.status
    this.code = details.code
    this.requestId = details.requestId ?? null
    this.fields = details.fields ?? []
    this.retryable = details.retryable ?? false
    this.etag = details.etag ?? null
    this.retryAfter = details.retryAfter ?? null
  }
}

function invalidRequest(code: string, message: string): never {
  throw new ApiError({ status: 0, code, message })
}

function apiPath(path: string): string {
  if (typeof path !== 'string' || !path.startsWith('/') || path.startsWith('//') || /[\\#\s\u0000-\u001f\u007f]/.test(path)) {
    return invalidRequest('INVALID_REQUEST_PATH', 'La solicitud debe usar una ruta interna de la API.')
  }
  // API route segments/UUIDs need no percent escapes. Query values may still be encoded.
  // Rejecting them before URL normalization also blocks encoded and double-encoded traversal.
  if (path.split('?', 1)[0]!.includes('%')) {
    return invalidRequest('INVALID_REQUEST_PATH', 'La ruta de la API no admite segmentos codificados.')
  }
  const hasPrefix = path === API_PREFIX || path.startsWith(`${API_PREFIX}/`) || path.startsWith(`${API_PREFIX}?`)
  if (path.startsWith('/api/') && !hasPrefix) {
    return invalidRequest('INVALID_REQUEST_PATH', 'La solicitud no corresponde a la versión de la API.')
  }
  const base = 'https://internal.invalid'
  const url = new URL(hasPrefix ? path : `${API_PREFIX}${path}`, base)
  if (url.origin !== base || (url.pathname !== API_PREFIX && !url.pathname.startsWith(`${API_PREFIX}/`))) {
    return invalidRequest('INVALID_REQUEST_PATH', 'La solicitud debe permanecer dentro de la API.')
  }
  return `${url.pathname}${url.search}`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function errorDetails(value: unknown): Omit<ApiErrorDetails, 'status'> | null {
  if (!isRecord(value) || !isRecord(value.error)) return null
  const error = value.error
  if (typeof error.code !== 'string' || !error.code.trim() || typeof error.message !== 'string' || !error.message.trim()
    || typeof error.requestId !== 'string' || !error.requestId.trim() || typeof error.retryable !== 'boolean' || !Array.isArray(error.fields)) return null
  const fields: ApiFieldError[] = []
  for (const field of error.fields) {
    if (!isRecord(field) || typeof field.path !== 'string' || typeof field.code !== 'string' || typeof field.message !== 'string') return null
    fields.push({ path: field.path, code: field.code, message: field.message })
  }
  return { code: error.code, message: error.message, requestId: error.requestId, fields, retryable: error.retryable }
}

function hasJsonContent(response: Response): boolean {
  const mediaType = response.headers.get('Content-Type')?.split(';', 1)[0]?.trim().toLowerCase()
  return mediaType === 'application/json' || Boolean(mediaType?.startsWith('application/') && mediaType.endsWith('+json'))
}

function metadata(response: Response) {
  return { status: response.status, requestId: response.headers.get('X-Request-Id'), etag: response.headers.get('ETag'), retryAfter: response.headers.get('Retry-After') }
}
function interrupted(signal?: AbortSignal): ApiError {
  return new ApiError({ status: 0, code: signal?.aborted ? 'REQUEST_ABORTED' : 'NETWORK_ERROR', message: signal?.aborted ? 'La solicitud se canceló. No se confirmó ningún cambio en el servidor.' : 'No pudimos completar la solicitud. No se confirmó ningún cambio en el servidor.' })
}
async function send(target: string, init: RequestInit): Promise<Response> {
  const sequence = beginApiObservation()
  try {
    const response = await fetch(target, { ...init, credentials: 'same-origin', mode: 'same-origin', redirect: 'error', cache: 'no-store' })
    observeApiResponse(sequence, response)
    return response
  } catch {
    if (!init.signal?.aborted) observeApiNetworkFailure(sequence)
    throw interrupted(init.signal ?? undefined)
  }
}

async function jsonResponse<T>(response: Response, signal?: AbortSignal): Promise<ApiResponse<T>> {
  const details = metadata(response)
  if (response.ok && (response.status === 204 || response.status === 205)) return { data: undefined, ...details }
  let data: unknown
  let validJson = false
  if (hasJsonContent(response)) {
    try { data = await response.json(); validJson = true } catch {
      if (signal?.aborted) throw new ApiError({ ...details, code: 'REQUEST_ABORTED', message: 'La solicitud se canceló. No se confirmó ningún cambio en el servidor.' })
    }
  }
  if (!response.ok) {
    throw new ApiError({ ...details, code: 'HTTP_ERROR', message: 'El servidor no pudo completar la solicitud. No se confirmó ningún cambio.', ...(validJson ? errorDetails(data) ?? {} : {}) })
  }
  if (!validJson) throw new ApiError({ ...details, code: 'INVALID_RESPONSE', message: 'La respuesta del servidor no tiene el formato esperado. No se confirmó ningún cambio.' })
  const unexpectedError = errorDetails(data)
  if (unexpectedError) throw new ApiError({ ...details, requestId: unexpectedError.requestId, fields: unexpectedError.fields, code: 'INVALID_RESPONSE', message: 'El servidor devolvió un error con un estado de éxito. No se confirmó ningún cambio.' })
  return { data: data as T, ...details }
}

/**
 * JSON transport. Endpoint DTOs are validated by their services, independently of this layer.
 * T describes the caller's expected payload and does not validate endpoint schemas at runtime.
 * Contract schemas/examples are checked separately; this module never changes demo state or retries.
 */
export async function apiRequest<T = unknown>(path: string, options: ApiRequestOptions = {}): Promise<ApiResponse<T>> {
  const target = apiPath(path)
  const method = options.method ?? 'GET'
  if (!['GET', 'POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
    return invalidRequest('INVALID_REQUEST', 'El método de la solicitud no está permitido.')
  }
  if (method === 'GET' && options.body !== undefined) {
    return invalidRequest('INVALID_REQUEST', 'Una consulta no puede incluir cambios para guardar.')
  }
  if (method !== 'GET' && (typeof options.csrfToken !== 'string' || !options.csrfToken.trim())) {
    return invalidRequest('CSRF_TOKEN_REQUIRED', 'Se necesita el token de la sesión antes de enviar cambios.')
  }
  let headers: Headers
  let body: string | undefined
  try {
    headers = new Headers({ Accept: 'application/json' })
    if (method !== 'GET') {
      headers.set('X-CSRF-Token', options.csrfToken!)
      if (options.ifMatch !== undefined) headers.set('If-Match', options.ifMatch)
      if (options.idempotencyKey !== undefined) headers.set('Idempotency-Key', options.idempotencyKey)
      if (options.body !== undefined) {
        body = JSON.stringify(options.body)
        if (body === undefined) throw new Error('non_json_body')
        headers.set('Content-Type', 'application/json')
      }
    }
  } catch {
    return invalidRequest('INVALID_REQUEST', 'No pudimos preparar la solicitud. Revisa los datos antes de enviarla.')
  }
  return trackMutation(method, async () => jsonResponse<T>(await send(target, { method, headers, body, signal: options.signal }), options.signal))
}

/** Multipart boundary is generated by the browser; no file or token is stored here. */
export async function apiMultipartRequest<T = unknown>(path: string, form: FormData, options: ApiMultipartOptions = {}): Promise<ApiResponse<T>> {
  const target = apiPath(path)
  const method = options.method ?? 'POST'
  if (!['POST', 'PUT', 'PATCH', 'DELETE'].includes(method) || !(form instanceof FormData)) return invalidRequest('INVALID_REQUEST', 'La carga debe usar un formulario válido.')
  if (typeof options.csrfToken !== 'string' || !options.csrfToken.trim()) return invalidRequest('CSRF_TOKEN_REQUIRED', 'Se necesita el token de la sesión antes de enviar cambios.')
  let headers: Headers
  try {
    headers = new Headers({ Accept: 'application/json', 'X-CSRF-Token': options.csrfToken })
    if (options.ifMatch !== undefined) headers.set('If-Match', options.ifMatch)
    if (options.idempotencyKey !== undefined) headers.set('Idempotency-Key', options.idempotencyKey)
  } catch { return invalidRequest('INVALID_REQUEST', 'No pudimos preparar la solicitud. Revisa los datos antes de enviarla.') }
  return trackMutation(method, async () => jsonResponse<T>(await send(target, { method, headers, body: form, signal: options.signal }), options.signal))
}

/** Authenticated server PNG only: HTML, JSON, SVG and empty bodies cannot become previews. */
export async function apiBlobRequest(path: string, options: { signal?: AbortSignal } = {}): Promise<ApiResponse<Blob> & { data: Blob }> {
  const response = await send(apiPath(path), { method: 'GET', headers: { Accept: 'image/png' }, signal: options.signal })
  if (!response.ok) await jsonResponse(response, options.signal)
  const details = metadata(response)
  const mediaType = response.headers.get('Content-Type')?.split(';', 1)[0]?.trim().toLowerCase()
  if (response.status !== 200 || mediaType !== 'image/png') throw new ApiError({ ...details, code: 'INVALID_RESPONSE', message: 'El servidor no devolvió una vista PNG válida.' })
  let blob: Blob
  try { blob = await response.blob() } catch { throw interrupted(options.signal) }
  if (options.signal?.aborted) throw interrupted(options.signal)
  const signature = new Uint8Array(await blob.slice(0, 8).arrayBuffer())
  if (signature.length !== 8 || ![137, 80, 78, 71, 13, 10, 26, 10].every((value, index) => signature[index] === value)) {
    throw new ApiError({ ...details, code: 'INVALID_RESPONSE', message: 'La vista recibida no es un PNG válido.' })
  }
  return { data: blob, ...details }
}
