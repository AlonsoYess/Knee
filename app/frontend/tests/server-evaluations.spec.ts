import { test, expect, type Page } from '@playwright/test'

const emailA = 'browser-a@kneeoai.invalid'
const emailB = 'browser-b@kneeoai.invalid'
const password = 'BrowserTestsOnly-2026!'

async function login(page: Page, email = emailA) {
  await page.goto('/acceso')
  await page.locator('#login-email').fill(email)
  await page.locator('#login-password').fill(password)
  await page.getByRole('button', { name: 'Iniciar sesión', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toBeVisible()
}

async function createCase(page: Page, age = '61', next = false) {
  await page.getByRole('button', { name: 'Nueva evaluación', exact: true }).click()
  await page.locator('#draft-age').fill(age)
  await page.locator('#draft-sex').selectOption('F')
  await page.locator('#draft-bmi').fill('28.4')
  await page.locator('#draft-knee-right').check()
  const saved = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/v1/evaluations')
  await page.getByRole('button', { name: next ? 'Guardar y continuar' : 'Guardar y salir', exact: true }).click()
  const response = await saved
  expect(response.status()).toBe(201)
  const evaluation = await response.json() as { id: string; code: string; clinical: { age: number }; currentRevisionId: string }
  await expect(page).not.toHaveURL(/\/nueva$/)
  return evaluation
}

test('real login rejects incorrect credentials, restores session and stores no password/token', async ({ page }) => {
  await page.addInitScript(() => sessionStorage.setItem('kneeoai:ui-demo:v1', 'active'))
  await page.goto('/evaluaciones')
  await expect(page).toHaveURL(/\/acceso$/)
  await page.locator('#login-email').fill(emailA)
  await page.locator('#login-password').fill('Incorrect-password-2026!')
  await page.getByRole('button', { name: 'Iniciar sesión', exact: true }).click()
  await expect(page.locator('.login-error[role="alert"]').last()).toBeVisible()
  await page.locator('#login-password').fill(password)
  await page.getByRole('button', { name: 'Iniciar sesión', exact: true }).click()
  await expect(page).toHaveURL(/\/evaluaciones$/)
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toBeVisible()
  const storage = await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } }))
  expect(storage).not.toContain(password)
  expect(storage).not.toContain('csrfToken')
  expect(await page.evaluate(() => document.cookie)).not.toContain('kneeoai_session')
  const session = await page.request.get('/api/v1/session')
  expect((await session.json()).authenticated).toBe(true)
})

test('basal evaluations survive reload and another browser session using SQLite', async ({ page, browser, baseURL }) => {
  await login(page)
  const evaluation = await createCase(page)
  await page.reload()
  await expect(page.getByRole('link', { name: `Ver resumen del caso ${evaluation.code}`, exact: true })).toBeVisible()
  const other = await browser.newContext({ baseURL })
  try {
    const second = await other.newPage()
    await login(second)
    await expect(second.getByRole('link', { name: `Ver resumen del caso ${evaluation.code}`, exact: true })).toBeVisible()
    const response = await second.request.get(`/api/v1/evaluations/${evaluation.id}`)
    expect(response.status()).toBe(200)
    expect((await response.json()).clinical.age).toBe(61)
  } finally { await other.close() }
})

test('partial drafts preserve null values on the server', async ({ page }) => {
  await login(page)
  await page.getByRole('button', { name: 'Nueva evaluación', exact: true }).click()
  await page.locator('#draft-age').fill('62')
  const saved = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/v1/evaluations')
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  const response = await saved
  expect(response.status()).toBe(201)
  const evaluation = await response.json()
  expect(evaluation.clinical).toEqual({ age: 62, bmi: null, sex: null, side: null })
  await expect(page).toHaveURL(/\/evaluaciones$/)
  await page.reload()
  await page.getByRole('button', { name: `Retomar ${evaluation.code}`, exact: true }).click()
  await expect(page.locator('#draft-age')).toHaveValue('62')
  await expect(page.locator('#draft-bmi')).toHaveValue('')
})

test('another account cannot read, update or see the first account case', async ({ page, browser, baseURL }) => {
  await login(page)
  const evaluation = await createCase(page, '63')
  const other = await browser.newContext({ baseURL })
  try {
    const second = await other.newPage()
    await login(second, emailB)
    await expect(second.getByRole('link', { name: `Ver resumen del caso ${evaluation.code}`, exact: true })).toHaveCount(0)
    const response = await second.request.get(`/api/v1/evaluations/${evaluation.id}`)
    expect(response.status()).toBe(404)
    await second.goto(`/evaluaciones/${evaluation.id}/resumen`)
    await expect(second.getByText('63 años', { exact: true })).toHaveCount(0)
    await expect(second.getByRole('button', { name: 'Calcular predicción' })).toHaveCount(0)
  } finally { await other.close() }
})

test('a failed write keeps the typed values and does not change the saved record', async ({ page }) => {
  await login(page)
  const evaluation = await createCase(page, '64')
  await page.getByRole('button', { name: `Retomar ${evaluation.code}`, exact: true }).click()
  await page.locator('#draft-age').fill('65')
  await page.route(`**/api/v1/evaluations/${evaluation.id}`, async route => {
    if (route.request().method() !== 'PATCH') return route.continue()
    await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: {
      code: 'DATABASE_UNAVAILABLE', message: 'No se pudo guardar. Reintenta.', requestId: 'req-test-write', fields: [], retryable: true,
    } }) })
  })
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  await expect(page.locator('.new-save-error')).toBeVisible()
  await expect(page.locator('#draft-age')).toHaveValue('65')
  const response = await page.request.get(`/api/v1/evaluations/${evaluation.id}`)
  expect((await response.json()).clinical.age).toBe(64)
  await page.unroute(`**/api/v1/evaluations/${evaluation.id}`)
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  await expect(page).toHaveURL(/\/evaluaciones$/)
  expect((await (await page.request.get(`/api/v1/evaluations/${evaluation.id}`)).json()).clinical.age).toBe(65)
})

test('a concurrent update prevents an older editor from overwriting it', async ({ page, browser, baseURL }) => {
  await login(page)
  const evaluation = await createCase(page, '66')
  await page.getByRole('button', { name: `Retomar ${evaluation.code}`, exact: true }).click()
  await page.locator('#draft-age').fill('67')
  const other = await browser.newContext({ baseURL })
  try {
    const second = await other.newPage()
    await login(second)
    await second.getByRole('button', { name: `Retomar ${evaluation.code}`, exact: true }).click()
    await second.locator('#draft-age').fill('68')
    await second.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
    await expect(second).toHaveURL(/\/evaluaciones$/)
    await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
    await expect(page.locator('.new-save-error')).toBeVisible()
    await expect(page.locator('#draft-age')).toHaveValue('67')
    expect((await (await page.request.get(`/api/v1/evaluations/${evaluation.id}`)).json()).clinical.age).toBe(68)
  } finally { await other.close() }
})

test('legacy drafts stay untouched and are not silently uploaded', async ({ page }) => {
  const raw = '{"schemaVersion":1,"fictionalLegacySentinel":"preserve-this-copy"}'
  await page.addInitScript(value => localStorage.setItem('kneeoai:demo-evaluations:v1', value), raw)
  let creates = 0
  page.on('request', request => { if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/v1/evaluations') creates++ })
  await login(page)
  await expect(page.locator('#legacy-drafts-notice')).toBeVisible()
  expect(await page.evaluate(() => localStorage.getItem('kneeoai:demo-evaluations:v1'))).toBe(raw)
  expect(creates).toBe(0)
})

test('server unavailable shows an error instead of a mock list', async ({ page }) => {
  await page.route('**/api/v1/session', route => route.abort())
  await page.goto('/acceso')
  await expect(page.locator('#api-session-error')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toHaveCount(0)
})

test('logout revokes the server session and protects direct navigation', async ({ page, browser, baseURL }) => {
  await login(page)
  const oldCookies = await page.context().cookies()
  await page.getByRole('button', { name: 'Cerrar sesión', exact: true }).click()
  await expect(page).toHaveURL(/\/acceso$/)
  await page.goto('/evaluaciones')
  await expect(page).toHaveURL(/\/acceso$/)
  const replay = await browser.newContext({ baseURL })
  try {
    await replay.addCookies(oldCookies)
    const response = await replay.request.get('/api/v1/session')
    expect((await response.json()).authenticated).toBe(false)
  } finally { await replay.close() }
})

test('a delayed session read cannot restore the UI authentication after logout', async ({ page }) => {
  await login(page)
  let release: () => void = () => {}
  const held = new Promise<void>(resolve => { release = resolve })
  let captured = false
  let reads = 0
  await page.route('**/api/v1/session', async route => {
    if (route.request().method() !== 'GET' || ++reads !== 1) return route.continue()
    const response = await route.fetch()
    expect((await response.json()).authenticated).toBe(true)
    captured = true
    await held
    await route.fulfill({ response })
  })
  await page.evaluate(async () => {
    const path = '/src/state/session.ts'
    const session = await import(/* @vite-ignore */ path)
    ;(window as unknown as { oldSessionRead: Promise<void> }).oldSessionRead = session.refreshSession().catch(() => {})
  })
  await expect.poll(() => captured).toBe(true)
  try {
    const closed = page.waitForResponse(response => response.request().method() === 'DELETE' && new URL(response.url()).pathname === '/api/v1/session')
    await page.getByRole('button', { name: 'Cerrar sesión', exact: true }).click()
    expect((await closed).status()).toBe(204)
    release()
    await page.evaluate(() => (window as unknown as { oldSessionRead: Promise<void> }).oldSessionRead)
    await expect(page).toHaveURL(/\/acceso$/)
    const authenticated = await page.evaluate(async () => {
      const path = '/src/state/session.ts'
      return (await import(/* @vite-ignore */ path)).serverSession.value?.authenticated ?? false
    })
    expect(authenticated).toBe(false)
    expect((await (await page.request.get('/api/v1/session')).json()).authenticated).toBe(false)
  } finally { release() }
})

test('mobile layout retains the server case and keeps prediction blocked', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 })
  await login(page)
  const evaluation = await createCase(page, '69')
  await page.getByRole('link', { name: `Ver resumen del caso ${evaluation.code}`, exact: true }).click()
  await expect(page.getByRole('button', { name: 'Calcular predicción', exact: true })).toBeDisabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.screenshot({ path: 'test-results/server-summary-mobile.png', fullPage: true })
})

test('lost creation response can be retried without duplicating the server case', async ({ page }) => {
  await login(page)
  await page.getByRole('button', { name: 'Nueva evaluación', exact: true }).click()
  await page.locator('#draft-age').fill('70')
  let attempts = 0
  let original: { id: string; code: string } | undefined
  await page.route('**/api/v1/evaluations', async route => {
    if (route.request().method() !== 'POST' || ++attempts > 1) return route.continue()
    const response = await route.fetch()
    expect(response.status()).toBe(201)
    original = await response.json()
    await route.abort('connectionreset')
  })
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  await expect(page.locator('.new-save-error')).toBeVisible()
  expect(attempts).toBe(1)
  await expect(page.locator('#draft-age')).toHaveValue('70')
  const retry = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/v1/evaluations')
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  const response = await retry
  expect(response.status()).toBe(200)
  expect((await response.json()).id).toBe(original!.id)
  await expect(page).toHaveURL(/\/evaluaciones$/)
  await expect(page.getByRole('link', { name: `Ver resumen del caso ${original!.code}`, exact: true })).toHaveCount(1)
})

test('a delayed response from account A cannot repopulate account B workspace', async ({ page }) => {
  await login(page)
  const owned = await createCase(page, '71')
  let release: () => void = () => {}
  const held = new Promise<void>(resolve => { release = resolve })
  let captured = false
  let queries = 0
  await page.route('**/api/v1/evaluations?**', async route => {
    if (++queries !== 1) return route.continue()
    const response = await route.fetch()
    captured = true
    await held
    await route.fulfill({ response })
  })
  await page.evaluate(async () => {
    const path = '/src/state/evaluations.ts'
    const store = await import(/* @vite-ignore */ path)
    ;(window as unknown as { oldLoad: Promise<void> }).oldLoad = store.loadEvaluations().catch(() => {})
  })
  await expect.poll(() => captured).toBe(true)
  try {
    await page.evaluate(async ({ email, password }) => {
      const sessionPath = '/src/state/session.ts', storePath = '/src/state/evaluations.ts'
      const session = await import(/* @vite-ignore */ sessionPath)
      await session.closeSession()
      await session.authenticate(email, password)
      const store = await import(/* @vite-ignore */ storePath)
      await store.loadEvaluations()
    }, { email: emailB, password })
    release()
    await page.evaluate(() => (window as unknown as { oldLoad: Promise<void> }).oldLoad)
    const state = await page.evaluate(async () => {
      const sessionPath = '/src/state/session.ts', storePath = '/src/state/evaluations.ts'
      const session = await import(/* @vite-ignore */ sessionPath), store = await import(/* @vite-ignore */ storePath)
      return { name: session.serverSession.value.user.displayName, ids: store.evaluations.value.map((item: { id: string }) => item.id) }
    })
    expect(state.name).toBe('Médico B de pruebas')
    expect(state.ids).not.toContain(owned.id)
  } finally { release() }
})

test('synthetic image acceptance stays temporary and never becomes server QC or prediction', async ({ page }) => {
  await login(page)
  let scientificRequests = 0
  page.on('request', request => { if (/\/api\/v1\/.*\/(preparations|executions)/.test(request.url())) scientificRequests++ })
  const evaluation = await createCase(page, '72', true)
  await expect(page.getByRole('heading', { name: 'Radiografía y lado', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Seleccionar imagen A', exact: true }).click()
  await page.locator('#radiograph-confirmation').check()
  await page.getByRole('button', { name: 'Continuar con el ejemplo', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Revisión del recorte', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Aceptar recorte de ejemplo', exact: true }).click()
  await expect(page.locator('#crop-reviewed-error')).toBeVisible()
  await page.locator('#crop-reviewed').check()
  await page.getByRole('button', { name: 'Aceptar recorte de ejemplo', exact: true }).click()
  await page.locator('#crop-save-summary').click()
  await expect(page.getByRole('heading', { name: 'Resumen del caso', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Calcular predicción', exact: true })).toBeDisabled()
  const actual = await (await page.request.get(`/api/v1/evaluations/${evaluation.id}`)).json()
  expect(actual.preparationId).toBeNull()
  expect(actual.reviewId).toBeNull()
  expect(actual.currentExecutionId).toBeNull()
  expect(scientificRequests).toBe(0)
  await page.screenshot({ path: 'test-results/server-summary-desktop.png', fullPage: true })
  await page.reload()
  await expect(page.getByText('No se ha seleccionado una imagen de ejemplo.', { exact: true })).toBeVisible()
  expect((await (await page.request.get(`/api/v1/evaluations/${evaluation.id}`)).json()).clinical.age).toBe(72)
})

test('synthetic rejection requires a reason and blocks continuing to a result', async ({ page }) => {
  await login(page)
  await createCase(page, '73', true)
  await page.getByRole('button', { name: 'Seleccionar imagen B', exact: true }).click()
  await page.locator('#radiograph-confirmation').check()
  await page.getByRole('button', { name: 'Continuar con el ejemplo', exact: true }).click()
  await page.getByRole('button', { name: 'Rechazar recorte de ejemplo', exact: true }).click()
  await expect(page.locator('#crop-reason-error')).toBeVisible()
  await page.locator('#crop-reason').selectOption('incomplete-example')
  await page.getByRole('button', { name: 'Rechazar recorte de ejemplo', exact: true }).click()
  await expect(page.locator('#crop-save-summary')).toBeDisabled()
  await page.getByRole('button', { name: 'Volver a evaluaciones', exact: true }).click()
  await expect(page).toHaveURL(/\/evaluaciones$/)
})
