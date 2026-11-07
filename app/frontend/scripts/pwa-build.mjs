import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

export const publicPwaFiles = [
  'manifest.webmanifest', 'icons/kneeoai.svg', 'icons/kneeoai-maskable.svg',
  'icons/kneeoai-192.png', 'icons/kneeoai-512.png',
]
const digest = value => createHash('sha256').update(value).digest('hex')

export function safeBuildId(value) {
  if (value !== undefined && !/^[a-z0-9][a-z0-9.-]{0,63}$/.test(value)) {
    throw new Error('KNEEOAI_BUILD_ID must contain 1–64 lowercase letters, digits, dots or hyphens.')
  }
  return value
}

export function createPwaBundle(entries, appVersion, buildId, template) {
  safeBuildId(buildId)
  const assets = entries.map(([path, content]) => {
    if (!(path === 'index.html' || publicPwaFiles.includes(path) || /^assets\/[A-Za-z0-9_.-]+-[A-Za-z0-9_-]{8,}\.(?:js|css|woff2)$/.test(path))) {
      throw new Error(`PWA asset is not an approved public build resource: ${path}`)
    }
    return { url: `/${path}`, sha256: digest(content) }
  }).sort((a, b) => a.url.localeCompare(b.url, 'en'))
  if (!assets.some(asset => asset.url === '/index.html')) throw new Error('PWA requires its public HTML shell.')
  if (new Set(assets.map(asset => asset.url)).size !== assets.length) throw new Error('Duplicate PWA asset.')
  const contentSha256 = digest(JSON.stringify(assets))
  const metadata = { buildId: buildId ?? contentSha256.slice(0, 20), contentSha256, appVersion, assets }
  const config = { ...metadata, cacheName: `kneeoai-static-v1-${metadata.buildId}-${contentSha256}` }
  if (!template.includes('/*__PWA_CONFIG__*/')) throw new Error('PWA template marker missing.')
  return { metadata, worker: template.replace('/*__PWA_CONFIG__*/', JSON.stringify(config)) }
}

export function createPwaBuildPlugin({ buildId } = {}) {
  safeBuildId(buildId)
  let root
  let publicDir
  return {
    name: 'kneeoai-public-shell-pwa',
    apply: 'build',
    configResolved(config) {
      if (config.base !== '/') throw new Error('KneeOAI PWA requires the same-origin root deployment.')
      root = config.root
      publicDir = config.publicDir
    },
    transformIndexHtml: {
      order: 'post',
      handler() {
        return [
          { tag: 'link', attrs: { rel: 'manifest', href: '/manifest.webmanifest' }, injectTo: 'head' },
          { tag: 'link', attrs: { rel: 'icon', href: '/icons/kneeoai.svg', type: 'image/svg+xml' }, injectTo: 'head' },
          { tag: 'link', attrs: { rel: 'apple-touch-icon', href: '/icons/kneeoai-192.png' }, injectTo: 'head' },
        ]
      },
    },
    generateBundle: {
      order: 'post',
      async handler(_options, bundle) {
        const entries = []
        for (const [name, item] of Object.entries(bundle)) {
          if (name === 'index.html' || /^assets\/.*\.(?:js|css|woff2)$/.test(name)) {
            entries.push([name, item.type === 'chunk' ? item.code : item.source])
          }
        }
        for (const name of publicPwaFiles) entries.push([name, await readFile(resolve(publicDir, name))])
        const appVersion = JSON.parse(await readFile(resolve(root, 'package.json'), 'utf8')).version
        const template = await readFile(new URL('./pwa-worker.js', import.meta.url), 'utf8')
        const result = createPwaBundle(entries, appVersion, buildId, template)
        this.emitFile({ type: 'asset', fileName: 'sw.js', source: result.worker })
        this.emitFile({ type: 'asset', fileName: 'pwa-version.json', source: `${JSON.stringify(result.metadata, null, 2)}\n` })
      },
    },
  }
}
