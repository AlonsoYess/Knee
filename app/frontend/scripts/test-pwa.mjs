import assert from 'node:assert/strict'
import { createHash, webcrypto } from 'node:crypto'
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises'
import { resolve, dirname, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import vm from 'node:vm'
import test from 'node:test'
import { build } from 'vite'
import { createPwaBundle, createPwaBuildPlugin, publicPwaFiles, safeBuildId } from './pwa-build.mjs'

const frontend = fileURLToPath(new URL('..', import.meta.url))
const template = await readFile(new URL('./pwa-worker.js', import.meta.url), 'utf8')
const origin = 'https://kneeoai.example.test'
const originalEntries = [
  ['index.html', '<html><body>Public shell</body></html>'],
  ['assets/index-abcdefgh.js', 'console.log("public interface")'],
  ['assets/index-abcdefgh.css', 'body{color:black}'],
  ['manifest.webmanifest', '{"name":"KneeOAI"}'],
]
const sha = bytes => createHash('sha256').update(bytes).digest('hex')
const keyFor = key => typeof key === 'string' ? key : key.url

class CacheStorageFixture {
  stores = new Map()
  async open(name) {
    if (!this.stores.has(name)) this.stores.set(name, new Map())
    const store = this.stores.get(name)
    return {
      put: async (key, value) => { store.set(keyFor(key), value.clone()) },
      match: async key => store.get(keyFor(key))?.clone(),
    }
  }
  async keys() { return [...this.stores.keys()] }
  async delete(name) { return this.stores.delete(name) }
}

function fixture({ id = 'test-a', entries = originalEntries, storage = new CacheStorageFixture(), clients } = {}) {
  const result = createPwaBundle(entries, '0.0.0-test', id, template)
  const handlers = new Map()
  const known = new Map(entries.map(([path, body]) => [`${origin}/${path}`, body]))
  const requests = []
  const main = { id: 'window-main', type: 'window', url: `${origin}/acceso` }
  let openClients = clients ?? [main]
  let networkUnavailable = false
  let skipWaitingCalls = 0
  const fetch = async request => {
    requests.push(request)
    if (networkUnavailable) throw new TypeError('network offline')
    const body = known.get(keyFor(request))
    return new Response(body ?? 'unlisted network response', { status: body === undefined ? 404 : 200 })
  }
  const context = vm.createContext({
    URL, Request, Response, Set, Uint8Array, crypto: webcrypto,
    caches: storage, fetch,
    self: {
      location: { origin },
      addEventListener: (name, callback) => handlers.set(name, callback),
      clients: { matchAll: async () => openClients },
      skipWaiting: async () => { skipWaitingCalls++ },
    },
  })
  vm.runInContext(result.worker, context)
  async function dispatch(name, fields = {}) {
    let response
    const pending = []
    handlers.get(name)({ ...fields, waitUntil: promise => pending.push(promise), respondWith: promise => { response = promise } })
    await Promise.all(pending)
    return response
  }
  return {
    ...result, storage, requests, known, main,
    dispatch,
    install: () => dispatch('install'),
    activate: () => dispatch('activate'),
    get skipWaitingCalls() { return skipWaitingCalls },
    setClients: value => { openClients = value },
    offline: () => { networkUnavailable = true },
    async message(type, source = main) {
      const replies = []
      await dispatch('message', { data: { type }, source, ports: [{ postMessage: value => replies.push(value) }] })
      return replies[0]
    },
    async fetch(path, { method = 'GET', navigate = false, headers = {} } = {}) {
      const request = new Request(new URL(path, origin), { method, headers })
      if (navigate) Object.defineProperty(request, 'mode', { value: 'navigate' })
      return dispatch('fetch', { request })
    },
  }
}

test('release IDs are deterministic and overrides cannot alias changed build bytes', () => {
  const first = createPwaBundle(originalEntries, '0.0.0-test', undefined, template)
  const second = createPwaBundle(originalEntries, '0.0.0-test', undefined, template)
  assert.equal(first.worker, second.worker)
  const a = fixture({ id: 'release-a' })
  const b = fixture({ id: 'release-b' })
  assert.equal(a.metadata.contentSha256, b.metadata.contentSha256)
  assert.notEqual(a.worker, b.worker)
  const changed = fixture({ id: 'release-a', entries: [...originalEntries.slice(1), ['index.html', 'updated public shell']] })
  assert.notEqual(changed.metadata.contentSha256, a.metadata.contentSha256)
  for (const value of ['', 'UPPER', '../escape', 'secret value', 'a'.repeat(65), '<script>']) {
    assert.throws(() => safeBuildId(value))
  }
})

test('build rejects any private/unversioned/unrecognized precache resource', () => {
  for (const path of ['api/v1/session', 'images/case.png', 'demo/radiograph.svg', 'assets/model.onnx', 'assets/index.js', '../index.html']) {
    assert.throws(() => createPwaBundle([...originalEntries, [path, 'private']], '0.0.0-test', 'test', template))
  }
  assert.throws(() => createPwaBundle([...originalEntries, originalEntries[0]], '0.0.0-test', 'test', template))
})

test('install fetches only verified public build bytes without cookies and does not activate', async () => {
  const app = fixture()
  await app.install()
  assert.equal(app.skipWaitingCalls, 0)
  assert.equal(app.requests.length, originalEntries.length)
  assert.ok(app.requests.every(request => request.credentials === 'omit' && request.cache === 'no-store' && request.redirect === 'error'))
  const stores = [...app.storage.stores.values()]
  assert.equal(stores.length, 1)
  assert.deepEqual([...stores[0].keys()].sort(), originalEntries.map(([name]) => `${origin}/${name}`).sort())
})

test('mixed deployment or private HTML response fails integrity before changing cache', async () => {
  const app = fixture()
  app.known.set(`${origin}/index.html`, '<html>private session or a different release</html>')
  await assert.rejects(app.install(), /changed during installation/)
  assert.equal(app.storage.stores.size, 0)
  assert.equal(app.skipWaitingCalls, 0)
})

test('API, private views, mutations, other origins and authenticated requests bypass worker', async () => {
  const app = fixture()
  await app.install()
  const writes = [...app.storage.stores.values()][0].size
  for (const path of ['/api', '/api/v1/session', '/api/v1/evaluations', '/api/v1/preparations/fictional/views/full_a?revision=fictional', '/api/v1/results/fictional', '/demo/radiografia-demo-a.svg', '/private/example.dcm', '/unknown', 'https://elsewhere.example/assets/index-abcdefgh.js']) {
    assert.equal(await app.fetch(path), undefined, path)
    assert.equal(await app.fetch(path, { navigate: true }), undefined, `navigate ${path}`)
  }
  assert.equal(await app.fetch('/assets/index-abcdefgh.js', { method: 'POST' }), undefined)
  assert.equal(await app.fetch('/assets/index-abcdefgh.js', { headers: { Authorization: 'Bearer fictional' } }), undefined)
  assert.equal(await app.fetch('/assets/index-abcdefgh.js', { headers: { Range: 'bytes=0-9' } }), undefined)
  assert.equal(await app.fetch('/assets/index-abcdefgh.js?patient=fictional'), undefined)
  assert.equal([...app.storage.stores.values()][0].size, writes)
})

test('offline navigation uses only the public shell and never stores a case URL', async () => {
  const app = fixture()
  await app.install()
  app.offline()
  for (const route of ['/', '/acceso', '/evaluaciones', '/evaluaciones/nueva', '/evaluaciones/fictional/revision']) {
    const response = await app.fetch(route, { navigate: true })
    assert.equal(await response.text(), originalEntries[0][1])
  }
  assert.equal(await (await app.fetch('/assets/index-abcdefgh.js')).text(), originalEntries[1][1])
  assert.equal([...app.storage.stores.values()][0].size, originalEntries.length)
})

test('runtime asset misses are fetched without adding network responses to cache', async () => {
  const app = fixture()
  await app.install()
  const response = await app.fetch('/assets/unlisted-abcdefgh.js')
  assert.equal(response.status, 404)
  assert.equal([...app.storage.stores.values()][0].size, originalEntries.length)
})

test('explicit activation rejects a second window and succeeds only after it closes', async () => {
  const app = fixture()
  await app.install()
  app.setClients([app.main, { id: 'window-other', type: 'window', url: `${origin}/evaluaciones/nueva` }])
  const blocked = await app.message('KNEEOAI_ACTIVATE_UPDATE')
  assert.equal(blocked.ok, false)
  assert.equal(blocked.code, 'OTHER_WINDOWS_OPEN')
  assert.equal(app.skipWaitingCalls, 0)
  app.setClients([app.main])
  assert.equal((await app.message('KNEEOAI_ACTIVATE_UPDATE')).ok, true)
  assert.equal(app.skipWaitingCalls, 1)
})

test('unknown messages, non-window clients and off-origin sources cannot activate', async () => {
  const app = fixture()
  await app.message('SKIP_WAITING')
  await app.message('KNEEOAI_ACTIVATE_UPDATE', { ...app.main, type: 'worker' })
  await app.message('KNEEOAI_ACTIVATE_UPDATE', { ...app.main, url: 'https://other.example/' })
  await app.message('KNEEOAI_ACTIVATE_UPDATE', null)
  assert.equal(app.skipWaitingCalls, 0)
  const version = await app.message('KNEEOAI_GET_VERSION')
  assert.equal(version.buildId, 'test-a')
  assert.equal(version.contentSha256, app.metadata.contentSha256)
})

test('two versions retain earlier hashed assets while any window remains open', async () => {
  const storage = new CacheStorageFixture()
  const a = fixture({ id: 'test-a', storage })
  await a.install()
  const b = fixture({ id: 'test-b', storage, entries: [originalEntries[0], ['assets/index-ijklmnop.js', 'new public code']] })
  await b.install()
  await b.activate()
  assert.equal((await storage.keys()).length, 2)
  b.offline()
  assert.equal(await (await b.fetch('/assets/index-abcdefgh.js')).text(), originalEntries[1][1])
  assert.equal(await (await b.fetch('/assets/index-ijklmnop.js')).text(), 'new public code')
})

test('natural activation without windows prunes only this application static caches', async () => {
  const storage = new CacheStorageFixture()
  const a = fixture({ id: 'test-a', storage })
  await a.install()
  await storage.open('another-app-cache')
  const b = fixture({ id: 'test-b', storage, clients: [] })
  await b.install()
  await b.activate()
  const names = await storage.keys()
  assert.equal(names.length, 2)
  assert.ok(names.includes('another-app-cache'))
  assert.ok(names.some(name => name.startsWith('kneeoai-static-v1-test-b-')))
})

test('manifest contains installable local routes and correct 192/512 raster icons', async () => {
  const manifest = JSON.parse(await readFile(resolve(frontend, 'public/manifest.webmanifest'), 'utf8'))
  assert.equal(manifest.scope, '/')
  assert.equal(manifest.start_url, '/acceso')
  assert.equal(manifest.display, 'standalone')
  assert.equal(manifest.prefer_related_applications, false)
  for (const size of [192, 512]) {
    const icon = manifest.icons.find(item => item.type === 'image/png' && item.sizes === `${size}x${size}`)
    assert.ok(icon)
    const bytes = await readFile(resolve(frontend, 'public', icon.src.slice(1)))
    assert.equal(bytes.subarray(0, 8).toString('hex'), '89504e470d0a1a0a')
    assert.equal(bytes.readUInt32BE(16), size)
    assert.equal(bytes.readUInt32BE(20), size)
  }
})

test('two controlled production builds emit coherent manifests, IDs, HTML links and hashes', async () => {
  const parent = resolve(frontend, 'test-results')
  await mkdir(parent, { recursive: true })
  const fixtureRoot = await mkdtemp(resolve(parent, 'pwa-core-'))
  try {
    await writeFile(resolve(fixtureRoot, 'package.json'), JSON.stringify({ version: '0.0.0-test', type: 'module' }))
    await writeFile(resolve(fixtureRoot, 'index.html'), '<html><head><title>Fictional shell</title></head><body><script type="module" src="/main.js"></script></body></html>')
    await writeFile(resolve(fixtureRoot, 'main.js'), 'document.body.dataset.publicShell = "true"')
    const results = []
    for (const id of ['pwa-test-a', 'pwa-test-b']) {
      const outDir = resolve(fixtureRoot, id)
      await build({
        root: fixtureRoot, configFile: false, logLevel: 'silent', base: '/',
        publicDir: resolve(frontend, 'public'),
        plugins: [createPwaBuildPlugin({ buildId: id })],
        build: { outDir, emptyOutDir: true },
      })
      const metadata = JSON.parse(await readFile(resolve(outDir, 'pwa-version.json'), 'utf8'))
      const worker = await readFile(resolve(outDir, 'sw.js'), 'utf8')
      const html = await readFile(resolve(outDir, 'index.html'), 'utf8')
      assert.equal(metadata.buildId, id)
      assert.match(worker, new RegExp(`kneeoai-static-v1-${id}-`))
      assert.match(html, /rel="manifest" href="\/manifest.webmanifest"/)
      assert.match(html, /rel="icon" href="\/icons\/kneeoai.svg"/)
      for (const asset of metadata.assets) {
        assert.equal(sha(await readFile(resolve(outDir, asset.url.slice(1)))), asset.sha256)
      }
      assert.ok(publicPwaFiles.every(file => metadata.assets.some(item => item.url === `/${file}`)))
      assert.ok(metadata.assets.every(asset => !asset.url.startsWith('/demo/') && !asset.url.startsWith('/api/')))
      assert.ok(!metadata.assets.some(asset => asset.url === '/sw.js' || asset.url === '/pwa-version.json'))
      results.push({ metadata, worker })
    }
    assert.equal(results[0].metadata.contentSha256, results[1].metadata.contentSha256)
    assert.notEqual(results[0].worker, results[1].worker)
  } finally {
    const safe = resolve(fixtureRoot)
    assert.equal(dirname(safe), parent)
    assert.ok(safe.startsWith(`${parent}${sep}pwa-core-`))
    await rm(safe, { recursive: true, force: true })
  }
})
