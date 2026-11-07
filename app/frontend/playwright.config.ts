import { defineConfig } from '@playwright/test'
import { fileURLToPath } from 'node:url'

const e2ePort = Number(process.env.E2E_PORT ?? '5174')
const apiPort = Number(process.env.E2E_API_PORT ?? '8011')
if (!Number.isInteger(e2ePort) || e2ePort < 1 || e2ePort > 65535) {
  throw new Error('E2E_PORT debe ser un puerto entero entre 1 y 65535.')
}
const baseURL = `http://127.0.0.1:${e2ePort}`
const isCI = Boolean(process.env.CI)
if (!Number.isInteger(apiPort) || apiPort < 1024 || apiPort > 65535 || apiPort === e2ePort) throw new Error('E2E_API_PORT debe ser un puerto distinto entre 1024 y 65535.')
const python = process.env.KNEEOAI_TEST_PYTHON ?? fileURLToPath(new URL(
  process.platform === 'win32' ? '../backend/.venv/Scripts/python.exe' : '../backend/.venv/bin/python', import.meta.url,
))

export default defineConfig({
  testDir: './tests',
  // These historical suites exercise the retired mock login and browser-only persistence.
  testIgnore: ['**/history-v08/**', '**/pwa.spec.ts'],
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL,
    viewport: { width: 1440, height: 1000 },
    browserName: 'chromium',
    launchOptions: process.platform === 'win32' && !isCI ? { channel: 'msedge' } : {},
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: `"${python}" tests/support/api_server.py`,
      url: `http://127.0.0.1:${apiPort}/api/v1/session`,
      reuseExistingServer: false,
      timeout: 30000,
      env: { E2E_PORT: String(e2ePort), E2E_API_PORT: String(apiPort) },
    },
    {
      command: `npm run dev -- --port ${e2ePort}`,
      url: `${baseURL}/acceso`,
      reuseExistingServer: false,
      timeout: 30000,
      env: { KNEEOAI_API_PROXY_TARGET: `http://127.0.0.1:${apiPort}` },
    },
  ],
})
