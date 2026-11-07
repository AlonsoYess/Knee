import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { createPwaBuildPlugin } from './scripts/pwa-build.mjs'

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, '.', 'KNEEOAI_')
  return {
    plugins: [vue(), createPwaBuildPlugin({ buildId: environment.KNEEOAI_BUILD_ID || undefined })],
    server: {
      host: '127.0.0.1',
      port: 5173,
      strictPort: true,
      proxy: {
        '/api': {
          target: environment.KNEEOAI_API_PROXY_TARGET || 'http://127.0.0.1:8000',
        },
      },
    },
    preview: {
      host: '127.0.0.1',
      port: 4173,
      strictPort: true,
    },
  }
})
