import { defineConfig } from '@playwright/test'
import { fileURLToPath } from 'node:url'

const port = Number(process.env.E2E_PWA_PORT ?? '8021')
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('Invalid PWA test port')
const python = process.env.KNEEOAI_TEST_PYTHON ?? fileURLToPath(new URL(
  process.platform === 'win32' ? '../backend/.venv/Scripts/python.exe' : '../backend/.venv/bin/python', import.meta.url))

export default defineConfig({
  testDir: './tests', testMatch: '**/pwa.spec.ts', fullyParallel: false, workers: 1,
  timeout: 45_000,
  outputDir: 'test-results/pwa-browser', reporter: 'list',
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: 'chromium',
    viewport: { width: 1440, height: 1000 }, trace: 'retain-on-failure' },
  webServer: { command: 'node tests/support/start-pwa-server.mjs', url: `http://127.0.0.1:${port}/api/v1/session`,
    reuseExistingServer: false, timeout: 90_000, env: { E2E_PWA_PORT: String(port), KNEEOAI_TEST_PYTHON: python } },
})
