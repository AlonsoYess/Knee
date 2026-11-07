/* This template is emitted only by the production build. No clinical data is cached. */
'use strict';
const CONFIG = /*__PWA_CONFIG__*/;
const CACHE_PREFIX = 'kneeoai-static-v1-';
const PUBLIC_URLS = new Set(CONFIG.assets.map(asset => asset.url));
const HASHED_ASSET = /^\/assets\/[A-Za-z0-9_.-]+-[A-Za-z0-9_-]{8,}\.(?:js|css|woff2)$/;

function isShellPath(path) {
  return path === '/' || path === '/index.html' || path === '/acceso' ||
    path === '/evaluaciones' || path === '/evaluaciones/nueva' ||
    /^\/evaluaciones\/[A-Za-z0-9_-]+\/(?:editar|radiografia|revision|resumen)$/.test(path);
}

async function verifiedPublicResponses() {
  const entries = [];
  for (const asset of CONFIG.assets) {
    const request = new Request(new URL(asset.url, self.location.origin), {
      method: 'GET', credentials: 'omit', cache: 'no-store', redirect: 'error',
    });
    const response = await fetch(request);
    if (response.status !== 200 || response.redirected || !['basic', 'default'].includes(response.type)) {
      throw new Error('The public build resource could not be verified.');
    }
    const bytes = await response.clone().arrayBuffer();
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), byte => byte.toString(16).padStart(2, '0')).join('');
    if (hash !== asset.sha256) throw new Error('The public build changed during installation.');
    entries.push([request, response]);
  }
  return entries;
}

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    // Verify the entire release before touching an existing cache of the same build.
    const entries = await verifiedPublicResponses();
    const cache = await caches.open(CONFIG.cacheName);
    for (const [request, response] of entries) await cache.put(request, response);
    // Intentionally no skipWaiting: an update stays pending until explicit consent.
  })());
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const windows = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    // Old releases stay usable while any page may still contain their JS or edits.
    // Cleanup is safe only on natural activation with no windows of this origin.
    if (!windows.length) {
      for (const name of await caches.keys()) {
        if (name.startsWith(CACHE_PREFIX) && name !== CONFIG.cacheName) await caches.delete(name);
      }
    }
    // No clients.claim: a new worker never commandeers an unrelated open page.
  })());
});

async function publicAsset(request, path) {
  const current = await caches.open(CONFIG.cacheName);
  const hit = await current.match(new URL(path, self.location.origin).href);
  if (hit) return hit;
  // Only immutable code/font assets from our own verified old builds can be reused.
  // This covers a new window racing with an explicitly requested update.
  if (HASHED_ASSET.test(path)) {
    for (const name of await caches.keys()) {
      if (name.startsWith(CACHE_PREFIX) && name !== CONFIG.cacheName) {
        const old = await (await caches.open(name)).match(new URL(path, self.location.origin).href);
        if (old) return old;
      }
    }
  }
  return fetch(request); // A miss is never written into any cache.
}

self.addEventListener('fetch', event => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin ||
      url.pathname === '/api' || url.pathname.startsWith('/api/') ||
      request.headers.has('Authorization') || request.headers.has('Range')) return;

  if (request.mode === 'navigate' && isShellPath(url.pathname)) {
    // The only HTML ever stored is the public build's index, not a fetched case page.
    event.respondWith((async () => {
      const shell = await (await caches.open(CONFIG.cacheName)).match(new URL('/index.html', self.location.origin).href);
      return shell || fetch(request);
    })());
    return;
  }
  if (url.search || url.hash || !(PUBLIC_URLS.has(url.pathname) || HASHED_ASSET.test(url.pathname))) return;
  event.respondWith(publicAsset(request, url.pathname));
});

self.addEventListener('message', event => {
  if (!['KNEEOAI_ACTIVATE_UPDATE', 'KNEEOAI_GET_VERSION'].includes(event.data?.type) || !event.ports?.[0]) return;
  const port = event.ports[0];
  event.waitUntil((async () => {
    const source = event.source;
    if (!source || source.type !== 'window' || new URL(source.url).origin !== self.location.origin) {
      port.postMessage({ ok: false, code: 'INVALID_CLIENT' });
      return;
    }
    if (event.data.type === 'KNEEOAI_GET_VERSION') {
      port.postMessage({ buildId: CONFIG.buildId, contentSha256: CONFIG.contentSha256, appVersion: CONFIG.appVersion });
      return;
    }
    const windows = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    if (!windows.some(client => client.id === source.id) || windows.some(client => client.id !== source.id)) {
      port.postMessage({ ok: false, code: 'OTHER_WINDOWS_OPEN' });
      return;
    }
    // User action is required by the page; the worker also enforces the window guard.
    await self.skipWaiting();
    port.postMessage({ ok: true, buildId: CONFIG.buildId });
  })().catch(() => port.postMessage({ ok: false, code: 'UPDATE_FAILED' })));
});
