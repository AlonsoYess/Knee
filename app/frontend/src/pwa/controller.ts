/** Install/update lifecycle only. No case data, credentials or drafts enter this module. */
export interface PwaState {
  supported: boolean
  offline: boolean
  installAvailable: boolean
  installed: boolean
  updateAvailable: boolean
  installing: boolean
  updating: boolean
  error: string | null
}

interface InstallPromptEvent extends Event {
  prompt(): Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

type Listener = (state: Readonly<PwaState>) => void
let registration: ServiceWorkerRegistration | undefined
let initialization: Promise<void> | undefined
let installPrompt: InstallPromptEvent | undefined
const listeners = new Set<Listener>()
const state: PwaState = {
  supported: typeof window !== 'undefined' && window.isSecureContext && 'serviceWorker' in navigator,
  offline: typeof navigator !== 'undefined' && !navigator.onLine,
  installAvailable: false,
  installed: typeof window !== 'undefined' && window.matchMedia('(display-mode: standalone)').matches,
  updateAvailable: false,
  installing: false,
  updating: false,
  error: null,
}

function publish(patch: Partial<PwaState>) {
  Object.assign(state, patch)
  for (const listener of listeners) listener(Object.freeze({ ...state }))
}

function observeInstalling(worker: ServiceWorker | null) {
  if (!worker) return
  const changed = () => {
    if (worker.state === 'installed') {
      publish({ updateAvailable: Boolean(registration?.waiting && registration.active) })
    } else if (worker.state === 'redundant') {
      publish({ error: 'No se pudo preparar la actualización. Puedes comprobarla de nuevo.' })
    }
  }
  worker.addEventListener('statechange', changed)
  changed()
}

async function initialize(): Promise<void> {
  if (initialization) return initialization
  initialization = (async () => {
    window.addEventListener('online', () => publish({ offline: false }))
    window.addEventListener('offline', () => publish({ offline: true }))
    const mode = window.matchMedia('(display-mode: standalone)')
    mode.addEventListener('change', () => publish({ installed: mode.matches }))
    window.addEventListener('beforeinstallprompt', event => {
      event.preventDefault()
      installPrompt = event as InstallPromptEvent
      publish({ installAvailable: true })
    })
    window.addEventListener('appinstalled', () => {
      installPrompt = undefined
      publish({ installed: true, installAvailable: false, installing: false })
    })
    // Development remains free of persistent workers and production-cache surprises.
    if (!state.supported || !import.meta.env.PROD) return
    try {
      registration = await navigator.serviceWorker.register('/sw.js', { scope: '/', updateViaCache: 'none' })
      publish({ updateAvailable: Boolean(registration.waiting && registration.active) })
      observeInstalling(registration.installing)
      registration.addEventListener('updatefound', () => observeInstalling(registration?.installing ?? null))
      // Detect another tab's activation without ever reloading this tab.
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        publish({ updateAvailable: Boolean(registration?.waiting) })
      })
    } catch {
      publish({ error: 'La instalación sin conexión no está disponible. Puedes seguir usando el sitio conectado.' })
    }
  })()
  return initialization
}

async function requestInstall(): Promise<boolean> {
  const prompt = installPrompt
  if (!prompt || state.installing) return false
  installPrompt = undefined
  publish({ installing: true, installAvailable: false, error: null })
  try {
    await prompt.prompt()
    return (await prompt.userChoice).outcome === 'accepted'
  } catch {
    publish({ error: 'No se pudo completar la instalación. Usa también la opción de instalación de tu navegador.' })
    return false
  } finally { publish({ installing: false }) }
}

async function checkForUpdate(): Promise<void> {
  if (!registration) return
  try {
    await registration.update()
    publish({ updateAvailable: Boolean(registration.waiting && registration.active), error: null })
  } catch { publish({ error: 'No se pudo comprobar la actualización. Revisa la conexión e inténtalo otra vez.' }) }
}

function requestActivation(worker: ServiceWorker): Promise<{ ok: boolean; code?: string }> {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel()
    const finish = () => { window.clearTimeout(timer); channel.port1.close() }
    const timer = window.setTimeout(() => { finish(); reject(new Error('timeout')) }, 15_000)
    channel.port1.onmessage = event => {
      finish()
      const data: unknown = event.data
      if (!data || typeof data !== 'object' || !('ok' in data) || typeof data.ok !== 'boolean') {
        reject(new Error('invalid response'))
      } else {
        resolve({ ok: data.ok, code: 'code' in data && typeof data.code === 'string' ? data.code : undefined })
      }
    }
    try { worker.postMessage({ type: 'KNEEOAI_ACTIVATE_UPDATE' }, [channel.port2]) }
    catch (error) { finish(); reject(error) }
  })
}

function waitForActivation(worker: ServiceWorker): Promise<void> {
  if (worker.state === 'activated') return Promise.resolve()
  return new Promise((resolve, reject) => {
    const cleanup = () => { window.clearTimeout(timer); worker.removeEventListener('statechange', changed) }
    const changed = () => {
      if (worker.state === 'activated') { cleanup(); resolve() }
      else if (worker.state === 'redundant') { cleanup(); reject(new Error('redundant')) }
    }
    const timer = window.setTimeout(() => { cleanup(); reject(new Error('timeout')) }, 15_000)
    worker.addEventListener('statechange', changed)
    changed()
  })
}

async function activateUpdate(canActivate: () => boolean = () => false): Promise<boolean> {
  const worker = registration?.waiting
  if (!worker || state.updating || !canActivate()) return false
  publish({ updating: true, error: null })
  try {
    // No await lies between this guard and posting the explicit activation request.
    if (!canActivate()) return false
    const response = await requestActivation(worker)
    if (!response.ok) {
      publish({ error: response.code === 'OTHER_WINDOWS_OPEN'
        ? 'Cierra las otras ventanas o pestañas de KneeOAI antes de actualizar. Su trabajo no se ha modificado.'
        : 'No se pudo activar la actualización. Conserva tu trabajo y vuelve a intentarlo.' })
      return false
    }
    await waitForActivation(worker)
    if (!canActivate()) {
      publish({ updateAvailable: false, error: 'La actualización está preparada. Conserva tu trabajo y recarga esta ventana cuando termines.' })
      return false
    }
    // This is the sole reload site. It is reachable only through this tab's explicit action.
    window.location.reload()
    return true
  } catch {
    publish({ error: 'No se pudo completar la actualización. Conserva tu trabajo antes de volver a intentarlo.' })
    return false
  } finally { publish({ updating: false }) }
}

export const pwaController = {
  initialize,
  getState: (): Readonly<PwaState> => Object.freeze({ ...state }),
  subscribe(listener: Listener): () => void {
    listeners.add(listener)
    listener(Object.freeze({ ...state }))
    return () => { listeners.delete(listener) }
  },
  requestInstall,
  checkForUpdate,
  activateUpdate,
}
