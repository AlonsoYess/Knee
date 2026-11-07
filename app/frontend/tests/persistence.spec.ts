import { test, expect, type Page } from '@playwright/test'

const storageKey = 'kneeoai:demo-evaluations:v1'

type SavedEnvelope = {
  schemaVersion: number
  fixtureVersion: string
  revision: unknown
  savedAt: string
  evaluations: { id: string; code: string; [field: string]: unknown }[]
}

async function login(page: Page) {
  await page.goto('/acceso')
  await page.getByRole('button', { name: 'Completar datos de ejemplo' }).click()
  await page.getByRole('button', { name: 'Entrar a la demostración' }).click()
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toBeVisible()
}

async function rawSnapshot(page: Page): Promise<string | null> {
  return page.evaluate(key => localStorage.getItem(key), storageKey)
}

async function newExample(page: Page) {
  await page.getByRole('button', { name: 'Nueva evaluación', exact: true }).click()
  await page.getByRole('button', { name: 'Completar ejemplo', exact: true }).click()
}

async function saveBaselines(page: Page) {
  await login(page)
  await newExample(page)
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  await expect(page.locator('tbody tr').filter({ hasText: 'DEMO-008' })).toBeVisible()
}

async function prepareReview(page: Page, knee: 'right' | 'left' = 'right', image: 'A' | 'B' = 'A') {
  await login(page)
  await newExample(page)
  await page.locator(`#draft-knee-${knee}`).check()
  await page.getByRole('button', { name: 'Guardar y continuar', exact: true }).click()
  await page.getByRole('button', { name: `Seleccionar imagen ${image}`, exact: true }).click()
  await page.locator('#radiograph-confirmation').check()
  await page.getByRole('button', { name: 'Guardar y continuar', exact: true }).click()
  await expect(page.locator('#crop-reviewed')).toBeEnabled()
}

async function failSnapshotWrites(page: Page) {
  await page.evaluate(key => {
    const original = Storage.prototype.setItem
    Storage.prototype.setItem = function (name: string, value: string) {
      if (this === window.localStorage && name === key) {
        throw new DOMException('Simulated quota failure for fictional cases', 'QuotaExceededError')
      }
      return original.call(this, name, value)
    }
  }, storageKey)
}

async function expectStorageAlert(page: Page) {
  const alert = page.locator('#demo-storage-alert')
  await expect(alert).toHaveAttribute('role', 'alert')
  await expect(alert).toBeVisible()
  await expect(alert).not.toHaveText('')
}

async function discardToList(page: Page) {
  await page.getByRole('link', { name: 'Mis evaluaciones', exact: true }).click()
  await expect(page.getByRole('dialog', { name: 'Tienes cambios sin guardar' })).toBeVisible()
  await page.getByRole('button', { name: 'Salir sin guardar', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toBeVisible()
}

test('partial cases save explicitly and reload discards only later unsaved fields', async ({ page }) => {
  await login(page)
  await page.getByRole('button', { name: 'Nueva evaluación', exact: true }).click()
  const before = await rawSnapshot(page)
  await page.locator('#draft-age').fill('67')
  expect(await rawSnapshot(page)).toBe(before)
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  const saved = await rawSnapshot(page)
  expect(saved).not.toBeNull()
  const envelope = JSON.parse(saved!) as SavedEnvelope
  expect(envelope.schemaVersion).toBe(1)
  expect(envelope.fixtureVersion).toBe('ui-fixtures-v1')
  expect(envelope.revision).toBeTruthy()
  expect(Number.isFinite(Date.parse(envelope.savedAt))).toBe(true)
  expect(envelope.evaluations.filter(item => item.code === 'DEMO-008')).toHaveLength(1)
  await page.getByRole('button', { name: 'Retomar DEMO-008', exact: true }).click()
  await page.locator('#draft-age').fill('70')
  expect(await rawSnapshot(page)).toBe(saved)
  page.once('dialog', dialog => dialog.accept())
  await page.reload()
  await expect(page.locator('#draft-code')).toHaveValue('DEMO-008')
  await expect(page.locator('#draft-age')).toHaveValue('67')
  await expect(page.locator('#draft-sex')).toHaveValue('')
  await expect(page.locator('#draft-bmi')).toBeEmpty()
  await expect(page.locator('#draft-knee-right')).not.toBeChecked()
  await expect(page.locator('#draft-knee-left')).not.toBeChecked()
  expect(await rawSnapshot(page)).toBe(saved)
})

test('a saved pending review retains its checked state and reason after reload', async ({ page }) => {
  await prepareReview(page)
  const before = await rawSnapshot(page)
  await page.locator('#crop-reviewed').check()
  await page.locator('#crop-reason').selectOption('other-example')
  expect(await rawSnapshot(page)).toBe(before)
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  await page.getByRole('button', { name: 'Ver revisión DEMO-008', exact: true }).click()
  const saved = await rawSnapshot(page)
  await page.reload()
  await expect(page.locator('.crop-code')).toContainText('DEMO-008')
  await expect(page.locator('#crop-reviewed')).toBeChecked()
  await expect(page.locator('#crop-reason')).toHaveValue('other-example')
  const reviewUrl = page.url()
  await page.getByRole('button', { name: 'Guardar y ver resumen', exact: true }).click()
  await expect(page).toHaveURL(reviewUrl)
  await expect(page.locator('.crop-action-error')).toContainText('Selecciona una aceptación explícita')
  expect(await rawSnapshot(page)).toBe(saved)
})

test('accepted summary saves once and survives reload and logout without enabling prediction', async ({ page }) => {
  await prepareReview(page)
  await page.locator('#crop-reviewed').check()
  await page.getByRole('button', { name: 'Aceptar recorte de ejemplo', exact: true }).click()
  await page.evaluate(key => {
    const countedWindow = window as Window & { demoSnapshotWriteCount?: number }
    countedWindow.demoSnapshotWriteCount = 0
    const original = Storage.prototype.setItem
    Storage.prototype.setItem = function (name: string, value: string) {
      const result = original.call(this, name, value)
      if (this === window.localStorage && name === key) countedWindow.demoSnapshotWriteCount! += 1
      return result
    }
  }, storageKey)
  await page.getByRole('button', { name: 'Guardar y ver resumen', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Resumen del caso', exact: true })).toBeVisible()
  expect(await page.evaluate(() => (window as Window & { demoSnapshotWriteCount?: number }).demoSnapshotWriteCount)).toBe(1)
  const saved = await rawSnapshot(page)
  await page.reload()
  await expect(page.locator('.summary-data-grid')).toContainText('DEMO-008')
  await expect(page.locator('#summary-review-status')).toHaveText('Aceptado en demostración')
  await expect(page.locator('.summary-review-details')).toContainText('ui-crop-v1')
  await expect(page.getByRole('button', { name: 'Calcular predicción', exact: true })).toBeDisabled()
  expect(await rawSnapshot(page)).toBe(saved)
  await page.getByRole('button', { name: 'Cerrar sesión de demostración' }).click()
  await expect(page).toHaveURL(/\/acceso$/)
  expect(await rawSnapshot(page)).toBe(saved)
  await login(page)
  await page.getByRole('button', { name: 'Ver resumen DEMO-008', exact: true }).click()
  await expect(page.locator('#summary-review-status')).toHaveText('Aceptado en demostración')
  await expect(page.locator('.summary-image-identity')).toContainText('radiografia-demo-a.svg')
  await expect(page.getByRole('button', { name: 'Calcular predicción', exact: true })).toBeDisabled()
  expect(await rawSnapshot(page)).toBe(saved)
})

test('a rejected left knee from image B reloads with its case, reason and crop version intact', async ({ page }) => {
  await prepareReview(page, 'left', 'B')
  await page.locator('#crop-reason').selectOption('incomplete-example')
  await page.getByRole('button', { name: 'Rechazar recorte de ejemplo', exact: true }).click()
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  await page.getByRole('link', { name: 'Ver resumen del caso DEMO-008', exact: true }).click()
  const saved = await rawSnapshot(page)
  await page.reload()
  await expect(page.locator('.summary-data-grid')).toContainText('DEMO-008')
  await expect(page.locator('.summary-data-grid')).toContainText('Izquierda del paciente')
  await expect(page.locator('.summary-image-identity')).toContainText('radiografia-demo-b.svg')
  await expect(page.locator('#summary-review-status')).toHaveText('Rechazado en demostración')
  await expect(page.locator('.summary-rejection')).toContainText('Región incompleta en el ejemplo')
  await expect(page.locator('.summary-review-details')).toContainText('ui-crop-v1')
  await expect(page.getByRole('button', { name: 'Calcular predicción', exact: true })).toBeDisabled()
  expect(await rawSnapshot(page)).toBe(saved)
})

test('a quota failure cannot create a case or replace the last saved snapshot', async ({ page }) => {
  await saveBaselines(page)
  const saved = await rawSnapshot(page)
  await newExample(page)
  const newUrl = page.url()
  await failSnapshotWrites(page)
  await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
  await expect(page).toHaveURL(newUrl)
  await expectStorageAlert(page)
  await expect(page.locator('#draft-code')).toHaveValue('DEMO-009')
  await expect(page.locator('#draft-age')).toHaveValue('62')
  await expect(page.locator('.new-bottom-note')).toContainText('cambios sin guardar')
  expect(await rawSnapshot(page)).toBe(saved)
  await discardToList(page)
  await expect(page.locator('tbody tr')).toHaveCount(8)
  await expect(page.locator('tbody tr').filter({ hasText: 'DEMO-009' })).toHaveCount(0)
})

test('a quota failure keeps an edited form dirty and leaves the saved case unchanged in memory', async ({ page }) => {
  await saveBaselines(page)
  const saved = await rawSnapshot(page)
  await page.getByRole('button', { name: 'Retomar DEMO-008', exact: true }).click()
  const editUrl = page.url()
  await page.locator('#draft-age').fill('64')
  await failSnapshotWrites(page)
  await page.getByRole('button', { name: 'Guardar y continuar', exact: true }).click()
  await expect(page).toHaveURL(editUrl)
  await expectStorageAlert(page)
  await expect(page.locator('#draft-age')).toHaveValue('64')
  await expect(page.locator('.new-bottom-note')).toContainText('cambios sin guardar')
  expect(await rawSnapshot(page)).toBe(saved)
  await discardToList(page)
  await page.getByRole('button', { name: 'Retomar DEMO-008', exact: true }).click()
  await expect(page.locator('#draft-age')).toHaveValue('62')
})

test('a quota failure prevents summary navigation without losing the provisional acceptance', async ({ page }) => {
  await prepareReview(page)
  const saved = await rawSnapshot(page)
  const reviewUrl = page.url()
  await page.locator('#crop-reviewed').check()
  await page.getByRole('button', { name: 'Aceptar recorte de ejemplo', exact: true }).click()
  await failSnapshotWrites(page)
  await page.getByRole('button', { name: 'Guardar y ver resumen', exact: true }).click()
  await expect(page).toHaveURL(reviewUrl)
  await expectStorageAlert(page)
  await expect(page.getByRole('button', { name: 'Reabrir revisión', exact: true })).toBeVisible()
  await expect(page.locator('.crop-dirty-note')).toContainText('cambios sin guardar')
  expect(await rawSnapshot(page)).toBe(saved)
  await discardToList(page)
  await expect(page.locator('tbody tr').filter({ hasText: 'DEMO-008' })).toContainText('Borrador')
  await page.getByRole('link', { name: 'Ver resumen del caso DEMO-008', exact: true }).click()
  await expect(page.locator('#summary-review-status')).toHaveText('Sin revisión asociada')
  await expect(page.getByRole('button', { name: 'Calcular predicción', exact: true })).toBeDisabled()
})

const invalidSnapshots = [
  { name: 'malformed JSON', corrupt: (_raw: string) => '{"schemaVersion":' },
  { name: 'an unknown schema version', corrupt: (raw: string) => JSON.stringify({ ...JSON.parse(raw), schemaVersion: 999 }) },
  {
    name: 'duplicate case identities',
    corrupt: (raw: string) => {
      const envelope = JSON.parse(raw) as SavedEnvelope
      envelope.evaluations.push({ ...envelope.evaluations[0]! })
      return JSON.stringify(envelope)
    },
  },
  {
    name: 'duplicate case codes',
    corrupt: (raw: string) => {
      const envelope = JSON.parse(raw) as SavedEnvelope
      envelope.evaluations.push({ ...envelope.evaluations[0]!, id: 'draft-00000000-0000-4000-8000-000000000099' })
      return JSON.stringify(envelope)
    },
  },
]

for (const sample of invalidSnapshots) {
  test(`${sample.name} blocks overwrite and preserves the original stored bytes`, async ({ page, browser }) => {
    await saveBaselines(page)
    const saved = await rawSnapshot(page)
    expect(saved).not.toBeNull()
    const corruptRaw = sample.corrupt(saved!)
    const isolated = await browser.newContext({ baseURL: new URL(page.url()).origin })
    try {
      await isolated.addInitScript(({ key, raw }) => {
        if (location.origin !== 'null' && localStorage.getItem(key) === null) localStorage.setItem(key, raw)
      }, { key: storageKey, raw: corruptRaw })
      const damagedPage = await isolated.newPage()
      await login(damagedPage)
      await expectStorageAlert(damagedPage)
      expect(await rawSnapshot(damagedPage)).toBe(corruptRaw)
      await newExample(damagedPage)
      const formUrl = damagedPage.url()
      await damagedPage.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
      await expect(damagedPage).toHaveURL(formUrl)
      await expectStorageAlert(damagedPage)
      await expect(damagedPage.locator('#draft-age')).toHaveValue('62')
      await expect(damagedPage.locator('.new-bottom-note')).toContainText('cambios sin guardar')
      expect(await rawSnapshot(damagedPage)).toBe(corruptRaw)
    } finally {
      await isolated.close()
    }
  })
}

test('a stale tab cannot overwrite a case saved by another tab', async ({ page }) => {
  await saveBaselines(page)
  await page.getByRole('button', { name: 'Retomar DEMO-008', exact: true }).click()
  const editUrl = page.url()
  await page.locator('#draft-age').fill('63')
  const other = await page.context().newPage()
  try {
    await login(other)
    await other.getByRole('button', { name: 'Retomar DEMO-008', exact: true }).click()
    await other.locator('#draft-age').fill('65')
    await other.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
    await expect(other.getByRole('heading', { name: 'Mis evaluaciones', exact: true })).toBeVisible()
    const winningSnapshot = await rawSnapshot(other)
    await page.getByRole('button', { name: 'Guardar y salir', exact: true }).click()
    await expect(page).toHaveURL(editUrl)
    await expectStorageAlert(page)
    await expect(page.locator('#draft-age')).toHaveValue('63')
    await expect(page.locator('.new-bottom-note')).toContainText('cambios sin guardar')
    expect(await rawSnapshot(page)).toBe(winningSnapshot)
    page.once('dialog', dialog => dialog.accept())
    await page.reload()
    await expect(page.locator('#draft-age')).toHaveValue('65')
    expect(await rawSnapshot(page)).toBe(winningSnapshot)
  } finally {
    await other.close()
  }
})
