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
  // A static synthetic SVG supplies the origin without bootstrapping Vue/auth requests.
  await page.goto('/demo/radiografia-demo-a.svg')
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

test('multipart preserves the browser boundary and versioned mutation headers', async ({ page }) => {
  let headers: Record<string, string> = {}, body = ''
  await page.route('**/api/v1/evaluations/test-only/preparations', async route => {
    headers = route.request().headers(); body = route.request().postDataBuffer()?.toString('utf8') ?? ''
    await route.fulfill({ status: 202, contentType: 'application/json', headers: { ETag: '"preparation-test-v1"' }, body: '{"id":"preparation-test"}' })
  })
  const result = await page.evaluate(async () => {
    const modulePath = '/src/api/http.ts'
    const { apiMultipartRequest } = await import(/* @vite-ignore */ modulePath)
    const form = new FormData(); form.set('file', new File(['SYNTHETIC-TRANSPORT-ONLY'], 'transport-only.dcm', { type: 'application/dicom' }))
    return apiMultipartRequest('/evaluations/test-only/preparations', form, { csrfToken: 'csrf-test-only', ifMatch: '"case-v1"', idempotencyKey: 'transport-upload-intent' })
  })
  expect(result.status).toBe(202)
  expect(result.etag).toBe('"preparation-test-v1"')
  expect(headers['content-type']).toMatch(/^multipart\/form-data; boundary=/)
  expect(headers).toMatchObject({ 'x-csrf-token': 'csrf-test-only', 'if-match': '"case-v1"', 'idempotency-key': 'transport-upload-intent' })
  expect(body).toContain('name="file"; filename="transport-only.dcm"')
  expect(body).toContain('SYNTHETIC-TRANSPORT-ONLY')
})

test('multipart rejects a missing CSRF token before sending bytes', async ({ page }) => {
  let requests = 0
  await page.route('**/api/v1/**', route => { requests += 1; return route.abort() })
  const code = await page.evaluate(async () => {
    const modulePath = '/src/api/http.ts'
    const { apiMultipartRequest } = await import(/* @vite-ignore */ modulePath)
    try { await apiMultipartRequest('/evaluations/test-only/preparations', new FormData()); return null }
    catch (error) { return (error as { code: string }).code }
  })
  expect(code).toBe('CSRF_TOKEN_REQUIRED')
  expect(requests).toBe(0)
})

test('authenticated PNG transport returns a blob without mutation headers', async ({ page }) => {
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6mMAAAAASUVORK5CYII=', 'base64')
  let headers: Record<string, string> = {}
  await page.route('**/api/v1/preparations/test-only/views/half_a?*', route => {
    headers = route.request().headers()
    return route.fulfill({ status: 200, contentType: 'image/png', headers: { 'X-Request-Id': 'png-test-only' }, body: png })
  })
  const result = await page.evaluate(async () => {
    const modulePath = '/src/api/http.ts'
    const { apiBlobRequest } = await import(/* @vite-ignore */ modulePath)
    const response = await apiBlobRequest('/preparations/test-only/views/half_a?revision=version-test')
    return { size: response.data.size, type: response.data.type, status: response.status, requestId: response.requestId }
  })
  expect(result).toEqual({ size: png.length, type: 'image/png', status: 200, requestId: 'png-test-only' })
  expect(headers.accept).toBe('image/png')
  expect(headers['x-csrf-token']).toBeUndefined()
  expect(headers['content-type']).toBeUndefined()
})

test('a JSON error or disguised SVG cannot become a PNG preview', async ({ page }) => {
  for (const scenario of [
    { contentType: 'application/json', body: '{"error":{"code":"TEST","message":"Only a test","requestId":"test","fields":[],"retryable":false}}' },
    { contentType: 'image/png', body: '<svg xmlns="http://www.w3.org/2000/svg"></svg>' },
    { contentType: 'image/png', body: '' },
  ]) {
    await page.route('**/api/v1/preparations/test-only/views/crop_a?*', route => route.fulfill({ status: 200, ...scenario }))
    const code = await page.evaluate(async () => {
      const modulePath = '/src/api/http.ts'
      const { apiBlobRequest } = await import(/* @vite-ignore */ modulePath)
      try { await apiBlobRequest('/preparations/test-only/views/crop_a?revision=version-test'); return null }
      catch (error) { return (error as { code: string }).code }
    })
    expect(code).toBe('INVALID_RESPONSE')
  }
})

test('PNG failures preserve the JSON error contract instead of returning a blob', async ({ page }) => {
  await page.route('**/api/v1/preparations/test-only/views/crop_b?*', route => route.fulfill({ status: 403, contentType: 'application/json', body: JSON.stringify({ error: { code: 'ACCESS_DENIED', message: 'Solicitud de prueba no autorizada.', requestId: 'png-forbidden-test', fields: [], retryable: false } }) }))
  const result = await page.evaluate(async () => {
    const modulePath = '/src/api/http.ts'
    const { apiBlobRequest } = await import(/* @vite-ignore */ modulePath)
    try { await apiBlobRequest('/preparations/test-only/views/crop_b?revision=version-test'); return null }
    catch (error) { const failure = error as { status: number; code: string; requestId: string }; return { status: failure.status, code: failure.code, requestId: failure.requestId } }
  })
  expect(result).toEqual({ status: 403, code: 'ACCESS_DENIED', requestId: 'png-forbidden-test' })
})

test('a mutation remains pending until its response body is consumed', async ({ page }) => {
  const count = await page.evaluate(async () => {
    const modulePath = '/src/api/http.ts', connectionPath = '/src/state/connection.ts'
    const { apiRequest } = await import(/* @vite-ignore */ modulePath)
    const { pendingMutations } = await import(/* @vite-ignore */ connectionPath)
    const original = window.fetch
    const state = window as Window & { finishTransportBody?: () => void; pendingTransportResult?: Promise<unknown> }
    window.fetch = async () => new Response(new ReadableStream({ start(controller) {
      state.finishTransportBody = () => { controller.enqueue(new TextEncoder().encode('{"saved":true}')); controller.close() }
    } }), { status: 200, headers: { 'Content-Type': 'application/json', 'X-Request-Id': 'pending-body-test' } })
    state.pendingTransportResult = apiRequest('/evaluations/test-only', { method: 'PATCH', csrfToken: 'csrf-test-only', body: { clinical: { age: 62 } } }).finally(() => { window.fetch = original })
    await Promise.resolve()
    return pendingMutations.value
  })
  expect(count).toBe(1)
  const result = await page.evaluate(async () => {
    const state = window as Window & { finishTransportBody?: () => void; pendingTransportResult?: Promise<unknown> }
    state.finishTransportBody!()
    await state.pendingTransportResult
    const modulePath = '/src/state/connection.ts'
    return (await import(/* @vite-ignore */ modulePath)).pendingMutations.value
  })
  expect(result).toBe(0)
})

test('a failed request clears the pending counter and an actual server response restores connectivity', async ({ page }) => {
  await page.route('**/api/v1/connection-test', route => route.abort('failed'))
  const failure = await request(page, '/connection-test', { method: 'POST', csrfToken: 'csrf-test-only', body: {} })
  expect(failure.error?.code).toBe('NETWORK_ERROR')
  const state = await page.evaluate(async () => {
    const modulePath = '/src/state/connection.ts'
    const { pendingMutations, serverReachability } = await import(/* @vite-ignore */ modulePath)
    return { pending: pendingMutations.value, server: serverReachability.value }
  })
  expect(state).toEqual({ pending: 0, server: 'unavailable' })
  await page.route('**/api/v1/connection-test', route => route.fulfill({ status: 200, contentType: 'application/json', headers: { 'X-Request-Id': 'connection-restored-test' }, body: '{}' }))
  expect((await request(page, '/connection-test')).ok).toBe(true)
  expect(await page.evaluate(async () => {
    const modulePath = '/src/state/connection.ts'
    return (await import(/* @vite-ignore */ modulePath)).serverReachability.value
  })).toBe('available')
})

test('a browser offline flag does not prevent a working loopback API request', async ({ page }) => {
  let requests = 0
  await page.route('**/api/v1/loopback-test', route => { requests += 1; return route.fulfill({ status: 200, contentType: 'application/json', headers: { 'X-Request-Id': 'local-server-test' }, body: '{"local":true}' }) })
  await page.evaluate(async () => {
    Object.defineProperty(navigator, 'onLine', { configurable: true, get: () => false })
    const modulePath = '/src/state/connection.ts'
    ;(await import(/* @vite-ignore */ modulePath)).updateBrowserOnline()
  })
  const result = await request(page, '/loopback-test', { method: 'POST', csrfToken: 'csrf-test-only', idempotencyKey: 'loopback-intent-test', body: { clinical: { age: null } } })
  expect(result.ok).toBe(true)
  expect(requests).toBe(1)
  expect(await page.evaluate(async () => {
    const modulePath = '/src/state/connection.ts'
    const { browserOnline, serverReachability } = await import(/* @vite-ignore */ modulePath)
    return { browserOnline: browserOnline.value, server: serverReachability.value }
  })).toEqual({ browserOnline: false, server: 'available' })
})
