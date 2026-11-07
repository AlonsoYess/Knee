import { test, expect, type Page } from '@playwright/test'
import type { ApiRequestOptions } from '../src/api/http'

async function request(page: Page, path: string, options: Omit<ApiRequestOptions, 'signal'> = {}) {
  return page.evaluate(async ({ path, options }) => {
    const modulePath = '/src/api/http.ts'
    const { apiRequest, ApiError } = await import(/* @vite-ignore */ modulePath)
    try { return { ok: true, result: await apiRequest(path, options) } } catch (error) {
      if (!(error instanceof ApiError)) throw error
      return { ok: false, error: {
        name: error.name, status: error.status, code: error.code, message: error.message,
        requestId: error.requestId, fields: error.fields, retryable: error.retryable,
        etag: error.etag, retryAfter: error.retryAfter,
      } }
    }
  }, { path, options })
}

test.beforeEach(async ({ page }) => {
  // These tests exercise only intercepted fictional responses, never a backend or model.
  await page.route('**/api/v1/**', route => route.abort())
  await page.goto('/acceso')
})

test('GET receives direct JSON and metadata without mutation headers', async ({ page }) => {
  let captured: Record<string, string> = {}
  await page.route('**/api/v1/session', async route => {
    captured = route.request().headers()
    await route.fulfill({ status: 200, contentType: 'application/json', headers: {
      ETag: '"fictional-r1"', 'X-Request-Id': 'req-fictional-read', 'Retry-After': '60',
    }, body: JSON.stringify({ authenticated: false, user: null, expiresAt: null, csrfToken: 'csrf-fictional' }) })
  })
  const result = await request(page, '/session', { csrfToken: 'must-not-send', ifMatch: 'must-not-send', idempotencyKey: 'must-not-send' })
  expect(result).toEqual({ ok: true, result: {
    data: { authenticated: false, user: null, expiresAt: null, csrfToken: 'csrf-fictional' }, status: 200,
    requestId: 'req-fictional-read', etag: '"fictional-r1"', retryAfter: '60',
  } })
  expect(captured.accept).toBe('application/json')
  for (const header of ['x-csrf-token', 'if-match', 'idempotency-key', 'content-type']) expect(captured[header]).toBeUndefined()
})

test('mutation sends supplied CSRF, version and idempotency without storing them', async ({ page }) => {
  let captured: { headers: Record<string, string>; method: string; body: unknown } | undefined
  await page.route('**/api/v1/evaluations/fictional-001', async route => {
    captured = { headers: route.request().headers(), method: route.request().method(), body: route.request().postDataJSON() }
    await route.fulfill({ status: 200, contentType: 'application/json', headers: { ETag: '"fictional-r2"' }, body: '{"saved":true}' })
  })
  const result = await request(page, '/api/v1/evaluations/fictional-001', {
    method: 'PATCH', body: { clinical: { age: 62 } }, csrfToken: 'csrf-fictional',
    ifMatch: '"fictional-r1"', idempotencyKey: 'idem-fictional-001',
  })
  expect(result.ok).toBe(true)
  expect(captured?.method).toBe('PATCH')
  expect(captured?.body).toEqual({ clinical: { age: 62 } })
  expect(captured?.headers).toMatchObject({
    accept: 'application/json', 'content-type': 'application/json',
    'x-csrf-token': 'csrf-fictional', 'if-match': '"fictional-r1"', 'idempotency-key': 'idem-fictional-001',
  })
  expect(await page.evaluate(() => JSON.stringify([localStorage, sessionStorage]))).not.toContain('csrf-fictional')
})

test('transport uses same-origin credentials and rejects redirections', async ({ page }) => {
  await page.route('**/api/v1/session', route => route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }))
  const captured = await page.evaluate(async () => {
    const original = window.fetch
    let settings: Pick<RequestInit, 'credentials' | 'mode' | 'redirect' | 'cache'> | undefined
    window.fetch = (input, init) => {
      settings = { credentials: init?.credentials, mode: init?.mode, redirect: init?.redirect, cache: init?.cache }
      return original(input, init)
    }
    const modulePath = '/src/api/http.ts'
    try { await (await import(/* @vite-ignore */ modulePath)).apiRequest('/session') } finally { window.fetch = original }
    return settings
  })
  expect(captured).toEqual({ credentials: 'same-origin', mode: 'same-origin', redirect: 'error', cache: 'no-store' })
})

for (const scenario of [
  { status: 401, code: 'AUTH_REQUIRED', retryable: false },
  { status: 409, code: 'STATE_CONFLICT', retryable: false },
  { status: 503, code: 'SERVICE_UNAVAILABLE', retryable: true },
]) {
  test(`HTTP ${scenario.status} preserves the error contract without retrying`, async ({ page }) => {
    let attempts = 0
    await page.route('**/api/v1/evaluations', async route => {
      attempts += 1
      await route.fulfill({ status: scenario.status, contentType: 'application/json', headers: { 'Retry-After': '30' }, body: JSON.stringify({ error: {
        code: scenario.code, message: 'Mensaje ficticio del contrato.', requestId: 'req-fictional-error',
        fields: [{ path: 'clinical.age', code: 'INVALID_VALUE', message: 'Detalle ficticio.' }], retryable: scenario.retryable,
      } }) })
    })
    const result = await request(page, '/evaluations', { method: 'POST', csrfToken: 'csrf-fictional', body: {} })
    expect(result).toEqual({ ok: false, error: {
      name: 'ApiError', status: scenario.status, code: scenario.code, message: 'Mensaje ficticio del contrato.',
      requestId: 'req-fictional-error', fields: [{ path: 'clinical.age', code: 'INVALID_VALUE', message: 'Detalle ficticio.' }],
      retryable: scenario.retryable, etag: null, retryAfter: '30',
    } })
    expect(attempts).toBe(1)
  })
}

test('HTML 404 becomes a uniform error and never exposes the HTML as a message', async ({ page }) => {
  await page.route('**/api/v1/missing', route => route.fulfill({ status: 404, contentType: 'text/html', headers: { 'X-Request-Id': 'req-fictional-html' }, body: '<html>PRIVATE SERVER TRACE</html>' }))
  const result = await request(page, '/missing')
  expect(result.ok).toBe(false)
  expect(result.error).toMatchObject({ status: 404, code: 'HTTP_ERROR', requestId: 'req-fictional-html', fields: [] })
  expect(JSON.stringify(result)).not.toContain('PRIVATE SERVER TRACE')
})

test('network failure rejects once without changing the demo snapshot', async ({ page }) => {
  let attempts = 0
  await page.route('**/api/v1/evaluations', route => { attempts += 1; return route.abort('failed') })
  const before = await page.evaluate(() => localStorage.getItem('kneeoai:demo-evaluations:v1'))
  const result = await request(page, '/evaluations', { method: 'POST', csrfToken: 'csrf-fictional', body: {} })
  expect(result.ok).toBe(false)
  expect(result.error).toMatchObject({ status: 0, code: 'NETWORK_ERROR', retryable: false })
  expect(attempts).toBe(1)
  expect(await page.evaluate(() => localStorage.getItem('kneeoai:demo-evaluations:v1'))).toBe(before)
})

test('cancellation before sending or during JSON body reading remains REQUEST_ABORTED', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const modulePath = '/src/api/http.ts'
    const { apiRequest, ApiError } = await import(/* @vite-ignore */ modulePath)
    const original = window.fetch
    const controller = new AbortController()
    let bodyReading = false
    let attempts = 0
    // Native Response.json consumes a partial browser stream; cancellation interrupts that read.
    const body = new ReadableStream<Uint8Array>({
      start(stream) {
        controller.signal.addEventListener('abort', () => stream.error(new DOMException('Fictional cancellation', 'AbortError')), { once: true })
        stream.enqueue(new TextEncoder().encode('{"saved":'))
      },
      pull() { bodyReading = true; controller.abort() },
    })
    window.fetch = async () => {
      attempts += 1
      return new Response(body, { status: 200, headers: { 'Content-Type': 'application/json', 'X-Request-Id': 'req-fictional-body-aborted' } })
    }
    let duringBody
    try {
      await apiRequest('/evaluations', { method: 'POST', csrfToken: 'csrf-fictional', body: {}, signal: controller.signal })
      duringBody = { success: true }
    } catch (error) {
      if (!(error instanceof ApiError)) throw error
      duringBody = { success: false, code: error.code, status: error.status, requestId: error.requestId }
    } finally { window.fetch = original }
    const cancelledBefore = new AbortController()
    cancelledBefore.abort()
    let beforeSending
    try {
      await apiRequest('/session', { signal: cancelledBefore.signal })
      beforeSending = { success: true }
    } catch (error) {
      if (!(error instanceof ApiError)) throw error
      beforeSending = { success: false, code: error.code, status: error.status }
    }
    return { duringBody, beforeSending, bodyReading, attempts }
  })
  expect(result).toEqual({
    duringBody: { success: false, code: 'REQUEST_ABORTED', status: 200, requestId: 'req-fictional-body-aborted' },
    beforeSending: { success: false, code: 'REQUEST_ABORTED', status: 0 }, bodyReading: true, attempts: 1,
  })
})

test('204 succeeds without parsing a body or inventing data', async ({ page }) => {
  await page.route('**/api/v1/session', route => route.fulfill({ status: 204, headers: { 'X-Request-Id': 'req-fictional-delete' } }))
  const result = await request(page, '/session', { method: 'DELETE', csrfToken: 'csrf-fictional' })
  expect(result.ok).toBe(true)
  expect(result.result).toMatchObject({ status: 204, requestId: 'req-fictional-delete' })
  expect(result.result?.data).toBeUndefined()
})

test('malformed JSON success rejects instead of reporting a saved case', async ({ page }) => {
  await page.route('**/api/v1/evaluations', route => route.fulfill({ status: 200, contentType: 'application/json', body: '{broken' }))
  const result = await request(page, '/evaluations', { method: 'POST', csrfToken: 'csrf-fictional', body: {} })
  expect(result.ok).toBe(false)
  expect(result.error).toMatchObject({ status: 200, code: 'INVALID_RESPONSE' })
})

test('a canonical error envelope under HTTP 200 cannot report a successful save', async ({ page }) => {
  await page.route('**/api/v1/evaluations', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ error: {
    code: 'VERSION_CONFLICT', message: 'Error ficticio bajo estado incorrecto.', requestId: 'req-fictional-invalid-status', fields: [], retryable: false,
  } }) }))
  const result = await request(page, '/evaluations', { method: 'POST', csrfToken: 'csrf-fictional', body: {} })
  expect(result.ok).toBe(false)
  expect(result.error).toMatchObject({ status: 200, code: 'INVALID_RESPONSE', requestId: 'req-fictional-invalid-status', fields: [] })
})

test('unsafe paths, missing CSRF and invalid headers are rejected before a request', async ({ page }) => {
  let attempts = 0
  await page.route('**/api/v1/**', route => { attempts += 1; return route.abort() })
  for (const path of [
    'https://outside.invalid/evaluations', '//outside.invalid/evaluations', '/../../outside', '/api/v1/../../../outside',
    '/%2e%2e/%2e%2e/outside', '/api/v1/%2e%2e%2f%2e%2e%2foutside', '/api/v1/..%5coutside',
    '/api/v1/%252e%252e/%252foutside', '/api/v1/%2foutside', '/api/v1/%5coutside', '/api/v1/%00outside',
    '/api/v1\\outside', '/api/v2/evaluations', '/evaluations#fragment', 'evaluations',
  ]) {
    const result = await request(page, path, { method: 'POST', csrfToken: 'csrf-fictional', body: {} })
    expect(result.error?.code, path).toBe('INVALID_REQUEST_PATH')
  }
  expect((await request(page, '/evaluations', { method: 'POST', body: {} })).error?.code).toBe('CSRF_TOKEN_REQUIRED')
  expect((await request(page, '/evaluations', { method: 'POST', body: {}, csrfToken: 'bad\nheader' })).error?.code).toBe('INVALID_REQUEST')
  expect((await request(page, '/evaluations', { body: {} })).error?.code).toBe('INVALID_REQUEST')
  expect(attempts).toBe(0)
  await page.route(/\/api\/v1\/evaluations\?search=a%2Fb$/, route => {
    attempts += 1
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{"items":[]}' })
  })
  expect((await request(page, '/evaluations?search=a%2Fb')).ok).toBe(true)
  expect(attempts).toBe(1)
})

test('an external redirect cannot receive the caller CSRF header', async ({ page }) => {
  let externalAttempts = 0
  await page.route('https://outside.invalid/**', route => { externalAttempts += 1; return route.abort() })
  await page.route('**/api/v1/evaluations', route => route.fulfill({ status: 307, headers: { Location: 'https://outside.invalid/collect' } }))
  const result = await request(page, '/evaluations', { method: 'POST', csrfToken: 'csrf-fictional', body: {} })
  expect(result.error?.code).toBe('NETWORK_ERROR')
  expect(externalAttempts).toBe(0)
})
