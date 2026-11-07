import { defineConfig } from '@playwright/test'

const e2ePort = Number(process.env.E2E_PORT ?? '5173')
if (!Number.isInteger(e2ePort) || e2ePort < 1 || e2ePort > 65535) {
  throw new Error('E2E_PORT debe ser un puerto entero entre 1 y 65535.')
}
const baseURL = `http://127.0.0.1:${e2ePort}`
const isCI = Boolean(process.env.CI)

export default defineConfig({
  testDir: './tests',
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
  webServer: {
    command: `npm run dev -- --port ${e2ePort}`,
    url: `${baseURL}/acceso`,
    reuseExistingServer: !isCI,
    timeout: 30000,
  },
})
