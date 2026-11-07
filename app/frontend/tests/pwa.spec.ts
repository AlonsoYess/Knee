import { test, expect, type Page, type APIRequestContext } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

const backend = fileURLToPath(new URL('../../backend/', import.meta.url))
const python = process.env.KNEEOAI_TEST_PYTHON ?? join(backend, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
let syntheticDicom: string

async function switchBuild(request: APIRequestContext, version: 'a' | 'b') {
  const response = await request.post(`/__pwa_test__/version/${version}`, { headers: { 'X-Pwa-Test-Control': 'synthetic-only-not-production' } })
  expect(response.status()).toBe(200)
}
async function workerVersion(page: Page, waiting = false) {
  return page.evaluate(async waiting => {
    const registration = await navigator.serviceWorker.ready
    const worker = waiting ? registration.waiting : navigator.serviceWorker.controller ?? registration.active
    if (!worker) return null
    return new Promise<{ buildId: string; contentSha256: string }>((resolve, reject) => {
      const channel = new MessageChannel()
      const timer = setTimeout(() => { channel.port1.close(); reject(new Error('Worker did not identify itself')) }, 5_000)
      channel.port1.onmessage = event => { clearTimeout(timer); channel.port1.close(); resolve(event.data) }
      worker.postMessage({ type: 'KNEEOAI_GET_VERSION' }, [channel.port2])
    })
  }, waiting)
}
async function warm(page: Page) {
  await page.goto('/acceso')
  await page.evaluate(async () => { await navigator.serviceWorker.ready })
  await page.reload()
  await expect.poll(async () => (await workerVersion(page))?.buildId).toBe('pwa-test-a')
}
async function login(page: Page) {
  await page.locator('#login-email').fill('browser-a@kneeoai.invalid')
  await page.locator('#login-password').fill('BrowserTestsOnly-2026!')
  await page.getByRole('button', { name: 'Iniciar sesión', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toBeVisible()
}
async function startDraft(page: Page, age = '61') {
  await page.getByRole('button', { name: 'Nueva evaluación', exact: true }).click()
  await page.locator('#draft-age').fill(age)
  await page.locator('#draft-sex').selectOption('F')
  await page.locator('#draft-knee-right').check()
}
async function waitingBuild(page: Page, request: APIRequestContext) {
  await switchBuild(request, 'b')
  await page.evaluate(async () => { await (await navigator.serviceWorker.ready).update() })
  await expect(page.locator('#pwa-update-notice')).toBeVisible()
  await expect.poll(async () => (await workerVersion(page, true))?.buildId).toBe('pwa-test-b')
}

test.beforeAll(() => {
  syntheticDicom = execFileSync(python, ['-c', 'import sys,base64;sys.path.insert(0,sys.argv[1]);from synthetic_dicoms import dicom_bytes;print(base64.b64encode(dicom_bytes()).decode())', join(backend, 'tests')], { cwd: backend, maxBuffer: 2_000_000 }).toString().trim()
})
test.beforeEach(async ({ request }) => { await switchBuild(request, 'a') })

test('the production manifest, icons and browser installability are valid', async ({ page, context }) => {
  const foreign: string[] = []
  page.on('request', request => { if (new URL(request.url()).origin !== new URL(page.url() === 'about:blank' ? request.url() : page.url()).origin) foreign.push(request.url()) })
  await warm(page)
  const manifest = await (await page.request.get('/manifest.webmanifest')).json()
  expect(manifest.name).toContain('KneeOAI')
  expect(manifest.short_name).toBe('KneeOAI')
  expect(manifest.display).toBe('standalone')
  expect(manifest.start_url).toBe('/acceso')
  expect(manifest.scope).toBe('/')
  for (const size of [192, 512]) {
    const icon = manifest.icons.find((icon: { sizes: string; type: string }) => icon.sizes === `${size}x${size}` && icon.type === 'image/png')
    expect(icon).toBeTruthy()
    const dimensions = await page.evaluate(async url => {
      const image = await createImageBitmap(await (await fetch(url)).blob())
      return [image.width, image.height]
    }, icon.src)
    expect(dimensions).toEqual([size, size])
  }
  const cdp = await context.newCDPSession(page)
  const parsed = await cdp.send('Page.getAppManifest')
  expect(parsed.errors.filter(error => error.critical)).toEqual([])
  const installability = await cdp.send('Page.getInstallabilityErrors')
  expect(installability.installabilityErrors).toEqual([])
  expect(foreign).toEqual([])
  await page.screenshot({ path: 'test-results/pwa-ready-desktop.png', fullPage: true })
})

test('authenticated JSON, images, decisions and errors never enter the static cache', async ({ page }) => {
  await warm(page); await login(page); await startDraft(page)
  const saved = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/v1/evaluations')
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  const evaluation = await (await saved).json()
  const checked = await page.evaluate(async ({ id, data }) => {
    const session = await (await fetch('/api/v1/session', { cache: 'no-store' })).json()
    const current = await fetch(`/api/v1/evaluations/${id}`, { cache: 'no-store' })
    const bytes = Uint8Array.from(atob(data), character => character.charCodeAt(0))
    const form = new FormData(); form.set('file', new File([bytes], 'synthetic-only.dcm', { type: 'application/dicom' }))
    const response = await fetch(`/api/v1/evaluations/${id}/preparations`, { method: 'POST', body: form,
      headers: { 'X-CSRF-Token': session.csrfToken, 'If-Match': current.headers.get('etag')!, 'Idempotency-Key': crypto.randomUUID() }, cache: 'no-store' })
    const preparation = await response.json()
    if (response.status !== 202) throw new Error(JSON.stringify(preparation))
    const pngStatuses = await Promise.all(Object.values(preparation.sourceContext.views as Record<string, string>).map(async url => (await fetch(url, { cache: 'no-store' })).status))
    const blocked = await fetch(`/api/v1/evaluations/${id}/executions`, { method: 'POST', headers: { 'X-CSRF-Token': session.csrfToken }, cache: 'no-store' })
    const forbidden = await fetch('/api/v1/evaluations/missing-synthetic-case', { cache: 'no-store' })
    const version = await (await fetch('/pwa-version.json', { cache: 'no-store' })).json()
    const declared = new Map<string,string>(version.assets.map((entry: { url: string; sha256: string }) => [entry.url,entry.sha256]))
    const entries = []
    for (const name of await caches.keys()) {
      const cache = await caches.open(name)
      for (const request of await cache.keys()) {
        const url = new URL(request.url), response = await cache.match(request)
        const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', await response!.arrayBuffer()))].map(value => value.toString(16).padStart(2,'0')).join('')
        entries.push({ path: url.pathname, knownStatic: declared.get(url.pathname) === hash })
      }
    }
    return { pngStatuses, blocked: blocked.status, forbidden: forbidden.status, entries,
      localState: JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } }) }
  }, { id: evaluation.id, data: syntheticDicom })
  expect(checked.pngStatuses).toEqual([200,200,200,200])
  expect(checked.blocked).toBe(503); expect(checked.forbidden).toBe(404)
  expect(checked.entries.length).toBeGreaterThan(5)
  expect(checked.entries.every(entry => entry.knownStatic && !entry.path.startsWith('/api/'))).toBe(true)
  expect(checked.localState).not.toContain(evaluation.id)
  expect(checked.localState).not.toContain('BrowserTestsOnly-2026!')
})

test('offline reopening shows the shell without an authenticated cached case', async ({ page, context }) => {
  await warm(page); await login(page)
  await context.setOffline(true)
  await page.reload()
  await expect(page).toHaveURL(/\/acceso$/)
  await expect(page.locator('#server-connection-notice')).toBeVisible()
  await expect(page.locator('#internet-connection-notice')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toHaveCount(0)
  await page.screenshot({ path: 'test-results/pwa-offline.png', fullPage: true })
  await context.setOffline(false)
  await page.locator('#retry-server-connection').click()
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toBeVisible()
})

test('a disconnected save keeps typed values and a later retry saves exactly once', async ({ page, context }) => {
  await warm(page); await login(page); await startDraft(page,'67')
  await context.setOffline(true)
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  await expect(page.locator('.new-save-error')).toBeVisible()
  await expect(page.locator('#draft-age')).toHaveValue('67')
  await expect(page.locator('#server-connection-notice')).toBeVisible()
  await context.setOffline(false)
  const saved = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/v1/evaluations')
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  const evaluation = await (await saved).json()
  await expect(page.getByRole('link', { name: `Ver resumen del caso ${evaluation.code}`, exact: true })).toHaveCount(1)
})

test('a new version waits and cancellation keeps an unsaved draft unchanged', async ({ page, request }) => {
  await warm(page); await login(page); await startDraft(page,'68')
  await waitingBuild(page,request)
  expect((await workerVersion(page))?.buildId).toBe('pwa-test-a')
  await expect(page.locator('#draft-age')).toHaveValue('68')
  await page.locator('#pwa-update').click()
  await page.getByRole('button', { name: 'Seguir editando', exact: true }).click()
  await expect(page.locator('#draft-age')).toHaveValue('68')
  expect((await workerVersion(page))?.buildId).toBe('pwa-test-a')
  await expect(page.locator('#pwa-update-notice')).toBeVisible()
  const saved = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/v1/evaluations')
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  const evaluation = await (await saved).json()
  await page.locator('#pwa-update').click()
  await expect.poll(async () => (await workerVersion(page))?.buildId).toBe('pwa-test-b')
  await expect(page.getByRole('link', { name: `Ver resumen del caso ${evaluation.code}`, exact: true })).toBeVisible()
  expect((await (await page.request.get(`/api/v1/evaluations/${evaluation.id}`)).json()).clinical.age).toBe(68)
})

test('an update cannot interrupt a pending save or reload another open window', async ({ page, request, context }) => {
  await warm(page); await login(page); await startDraft(page,'69')
  await waitingBuild(page,request)
  let release: () => void = () => {}, captured = false
  const held = new Promise<void>(resolve => { release = resolve })
  await page.route('**/api/v1/evaluations', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    captured = true; await held; await route.continue()
  })
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  await expect.poll(() => captured).toBe(true)
  try {
    await expect(page.locator('#pwa-update')).toBeDisabled()
    expect((await workerVersion(page))?.buildId).toBe('pwa-test-a')
  } finally { release() }
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toBeVisible()
  const other = await context.newPage()
  await other.goto('/evaluaciones/nueva')
  await other.locator('#draft-age').fill('70')
  await page.locator('#pwa-update').click()
  await expect(page.locator('#pwa-error')).toContainText('otras ventanas')
  await expect(other.locator('#draft-age')).toHaveValue('70')
  expect((await workerVersion(page))?.buildId).toBe('pwa-test-a')
  other.on('dialog', dialog => dialog.accept())
  await other.close({ runBeforeUnload: false })
  await page.locator('#pwa-update').click()
  await expect.poll(async () => (await workerVersion(page))?.buildId).toBe('pwa-test-b')
})

test('a filled login requires update consent, preserves cancellation and clears fields only after confirmed reload', async ({ page, request }) => {
  await warm(page)
  let loginPosts = 0
  page.on('request', sent => { if (sent.method() === 'POST' && new URL(sent.url()).pathname === '/api/v1/session') loginPosts += 1 })
  await page.locator('#login-email').fill('not-submitted@kneeoai.invalid')
  await page.locator('#login-password').fill('NotSubmitted-TestOnly-2026!')
  await waitingBuild(page, request)
  await page.locator('#pwa-update').click()
  await expect(page.locator('#pwa-update-confirm')).toBeVisible()
  await page.getByRole('button', { name: 'Seguir aquí', exact: true }).click()
  await expect(page.locator('#login-email')).toHaveValue('not-submitted@kneeoai.invalid')
  await expect(page.locator('#login-password')).toHaveValue('NotSubmitted-TestOnly-2026!')
  expect((await workerVersion(page))?.buildId).toBe('pwa-test-a')
  expect((await workerVersion(page, true))?.buildId).toBe('pwa-test-b')
  await expect(page.locator('#pwa-update-notice')).toBeVisible()
  expect(loginPosts).toBe(0)
  await page.locator('#pwa-update').click()
  await expect(page.locator('#pwa-update-confirm')).toBeVisible()
  await page.getByRole('button', { name: 'Descartar y actualizar', exact: true }).click()
  await expect.poll(async () => (await workerVersion(page))?.buildId).toBe('pwa-test-b')
  await expect(page).toHaveURL(/\/acceso$/)
  await expect(page.locator('#login-email')).toBeEmpty()
  await expect(page.locator('#login-password')).toBeEmpty()
  expect(loginPosts).toBe(0)
})

test('an unadded QC ruler point requires update consent even when the normal beforeunload guard is clean', async ({ page, request }) => {
  await warm(page); await login(page); await startDraft(page)
  const saved = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/v1/evaluations')
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  const evaluation = await (await saved).json()
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toBeVisible()
  // Arrange a persisted synthetic review; the DICOM/reference/QC UI flow is covered
  // by preparations.spec.ts. No production or scientific data is used here.
  const stored = await page.evaluate(async ({ evaluationId, dicom }) => {
    const session = await (await fetch('/api/v1/session', { cache: 'no-store' })).json()
    const caseResponse = await fetch(`/api/v1/evaluations/${evaluationId}`, { cache: 'no-store' })
    const bytes = Uint8Array.from(atob(dicom), character => character.charCodeAt(0))
    const form = new FormData(); form.set('file', new File([bytes], 'synthetic-only.dcm', { type: 'application/dicom' }))
    const upload = await fetch(`/api/v1/evaluations/${evaluationId}/preparations`, { method: 'POST', body: form,
      headers: { 'X-CSRF-Token': session.csrfToken, 'If-Match': caseResponse.headers.get('etag')!, 'Idempotency-Key': crypto.randomUUID() }, cache: 'no-store' })
    if (upload.status !== 202) throw new Error(`Synthetic upload failed: ${upload.status}`)
    const preparation = await upload.json()
    const sourceResponse = await fetch(`/api/v1/preparations/${preparation.id}`, { cache: 'no-store' })
    const source = await sourceResponse.json()
    const reference = await fetch(`/api/v1/preparations/${preparation.id}/reference`, { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': session.csrfToken, 'If-Match': sourceResponse.headers.get('etag')!, 'Idempotency-Key': crypto.randomUUID() },
      body: JSON.stringify({ preparationRevisionId: source.preparationRevisionId, confirmedSide: 'right', centerResolved: true, centerXY: [200, 120], originalAndHalfInspected: true, outcomeBlinded: true,
        note: 'Synthetic original and native half inspected before ROI for this isolated update-consent test.' }), cache: 'no-store' })
    if (reference.status !== 201) throw new Error(`Synthetic reference failed: ${reference.status}`)
    const preparedResponse = await fetch(`/api/v1/preparations/${preparation.id}`, { cache: 'no-store' })
    const prepared = await preparedResponse.json()
    const review = await fetch(`/api/v1/preparations/${preparation.id}/reviews`, { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': session.csrfToken, 'If-Match': preparedResponse.headers.get('etag')!, 'Idempotency-Key': crypto.randomUUID() },
      body: JSON.stringify({ preparationRevisionId: prepared.preparationRevisionId, confirmedSide: 'right', decision: 'accepted', confirmedCheckIds: [], reason: null,
        observation: { side_supported: true, coverage_complete: true, visibility: { medial_joint_borders: 'visible', lateral_joint_borders: 'visible', subchondral_bone: 'visible' },
          hardware: 'absent', text_or_mask: 'absent', marks: ['peripheral_ruler'], mark_effect: 'non_obscuring', physical_origin_known: true,
          note: 'Synthetic ruler observation with both ROI windows inspected solely to test an unadded coordinate draft.', original_half_inspected: true, both_windows_inspected: true, outcome_blinded: true,
          lateral_margins_visible: true, unclassified_mark_present: false, ruler_outside_bone: true, ruler_locations_xy: [[200, 120]] } }), cache: 'no-store' })
    if (review.status !== 201) throw new Error(`Synthetic review failed: ${review.status}`)
    return { preparationId: preparation.id, review: await review.json() }
  }, { evaluationId: evaluation.id, dicom: syntheticDicom })
  await page.goto(`/evaluaciones/${evaluation.id}/revision`)
  await page.getByRole('button', { name: 'Registrar nueva decisión', exact: true }).click()
  await page.locator('#ruler-point-x').fill('220')
  await page.locator('#ruler-point-y').fill('130')
  const guardIsDirty = await page.evaluate(() => {
    const probe = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(probe); return probe.defaultPrevented
  })
  expect(guardIsDirty).toBe(false)
  await waitingBuild(page, request)
  await page.locator('#pwa-update').click()
  await expect(page.locator('#pwa-update-confirm')).toBeVisible()
  await page.getByRole('button', { name: 'Seguir aquí', exact: true }).click()
  await expect(page).toHaveURL(new RegExp(`/evaluaciones/${evaluation.id}/revision$`))
  await expect(page.locator('#ruler-point-x')).toHaveValue('220')
  await expect(page.locator('#ruler-point-y')).toHaveValue('130')
  await expect(page.locator('#ruler-x-0')).toHaveValue('200')
  await expect(page.locator('#ruler-y-0')).toHaveValue('120')
  expect((await workerVersion(page))?.buildId).toBe('pwa-test-a')
  expect((await workerVersion(page, true))?.buildId).toBe('pwa-test-b')
  const current = await (await page.request.get(`/api/v1/preparations/${stored.preparationId}`)).json()
  expect(current.activeReview.id).toBe(stored.review.id)
  expect(current.activeReview.observation.ruler_locations_xy).toEqual([[200, 120]])
})
