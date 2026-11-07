# Public shell and explicit updates

The build plugin emits `sw.js` and `pwa-version.json` from the exact emitted HTML,
hashed JavaScript/CSS/fonts and the five explicitly listed manifest/icon files.
Only their SHA-256-verified bytes may be written to Cache Storage. The shell is
public; it does not contain a session, case, image or prediction. There is no runtime
cache and no background synchronization. `/api`, other origins, mutations,
authenticated requests, private images, `/demo` and unknown routes are outside
the worker's cache. API transport continues to use the server on every request.

`KNEEOAI_BUILD_ID` is optional, public and limited to 1–64 lowercase ASCII
letters/digits/dots/hyphens, starting with a letter or digit. Without it the ID is
derived deterministically from the build. Cache names also contain the full asset
digest, so reusing an override never aliases different contents. Do not place a
secret or patient identifier in this setting. `pwa-version.json` describes the
deployed distribution, not necessarily the version in an already open tab.

`src/pwa/controller.ts` exposes `pwaController.initialize/getState/subscribe`,
`requestInstall`, `checkForUpdate` and `activateUpdate(canActivate)`. The last
operation requires an explicit caller action and a synchronous callback that
allows navigation only after pending edits and mutations have been handled. The
callback is checked before the worker message and again before the sole reload.
No controller-change event reloads a tab. Installation is offered only after the
browser supplies its real `beforeinstallprompt` event; support varies by browser.

An update waits by default. The waiting worker rejects activation if any other
window of the origin is open. There is no automatic `skipWaiting` or
`clients.claim`. Older static caches remain while clients exist, including when a
new window races with activation. They are removed only during a natural
activation with no open windows. This deliberately favors retaining a small
public build over breaking an existing tab; no private data is retained with it.
The browser can independently evict site storage, so offline availability is not
a guarantee of persistent clinical data or of an available local server.

For diagnostics, send `KNEEOAI_GET_VERSION` with a `MessageChannel` port to
`navigator.serviceWorker.controller` or `registration.waiting`. The response is
`{buildId, contentSha256, appVersion}` for that exact worker. An activation request
uses `KNEEOAI_ACTIVATE_UPDATE`; its reply is `{ok:true, buildId}` or
`{ok:false, code}`. Only a same-origin window client can request these actions.

Run `node scripts/test-pwa.mjs` for worker/cache isolation tests and two controlled
production builds. `node scripts/generate-pwa-icons.mjs` reproduces PNG icons
from the SVG source using the already pinned Playwright browser. Browser lifecycle
and unsaved-edit tests are separate and must also pass for a release. The icon
reuses the application's existing Lucide Activity mark; its license is below.

The implementation follows the browser lifecycle and installability requirements:
[MDN installability](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Making_PWAs_installable),
[Chrome update lifecycle](https://developer.chrome.com/docs/workbox/handling-service-worker-updates/),
[MDN service workers](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API/Using_Service_Workers).

## Icon license

ISC License

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted, provided that the above copyright notice
and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
SOFTWARE.
