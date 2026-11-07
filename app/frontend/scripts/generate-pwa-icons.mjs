// Raster copies of the existing Lucide Activity brand, using the already pinned browser.
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { chromium } from '@playwright/test'

const source = await readFile(new URL('../public/icons/kneeoai.svg', import.meta.url), 'utf8')
const browser = await chromium.launch({ headless: true })
try {
  for (const size of [192, 512]) {
    const page = await browser.newPage({ viewport: { width: size, height: size }, deviceScaleFactor: 1 })
    await page.setContent(`<style>html,body{margin:0;background:transparent}svg{width:100vw;height:100vh;display:block}</style>${source}`)
    await page.screenshot({ path: fileURLToPath(new URL(`../public/icons/kneeoai-${size}.png`, import.meta.url)), omitBackground: true })
    await page.close()
  }
} finally { await browser.close() }
