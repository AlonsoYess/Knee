/** Actual browser installation in a NEW test profile; never use a personal browser profile. */
import { chromium, expect } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'

const args = process.argv.slice(2)
const option = (name, fallback) => { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1] }
const base = new URL(option('--url','http://127.0.0.1:8022'))
if (base.protocol !== 'http:' || base.hostname !== '127.0.0.1' || Number(base.port) < 8020 || Number(base.port) > 8099 || base.pathname !== '/') {
  throw new Error('Use a loopback test server on a dedicated port between 8020 and 8099')
}
const frontend = fileURLToPath(new URL('../../',import.meta.url))
const profile = resolve(frontend,'../backend/runtime',`pwa-install-profile-${randomUUID()}`)
const output = resolve(option('--output',resolve(frontend,'test-results/pwa-install')))
await mkdir(output,{recursive:true})
const headed = args.includes('--headed')
const options = { channel: 'chromium', headless: !headed, viewport: {width:1280,height:900}, args:['--disable-session-crashed-bubble'] }
if (headed && process.platform === 'win32') options.channel = 'msedge'
const summary = { profileIsIsolated:true,syntheticOnly:true,browser: options.channel ?? 'chromium',headless:!headed,passed:false }
let context, cdp, installed = false, manifestId

async function launchApp(context,cdp) {
  const result = await cdp.send('PWA.launch',{manifestId})
  // PWA.launch returns its "tab" container target, while Playwright exposes the
  // child "page" target. Match the real browser window, never their unequal IDs.
  const launchedWindow = await cdp.send('Browser.getWindowForTarget',{targetId:result.targetId})
  let page
  await expect.poll(async () => {
    const matches=[]
    for (const candidate of context.pages()) {
      const session = await context.newCDPSession(candidate)
      const info = await session.send('Target.getTargetInfo')
      await session.detach()
      const candidateWindow=await cdp.send('Browser.getWindowForTarget',{targetId:info.targetInfo.targetId})
      if (candidateWindow.windowId===launchedWindow.windowId) matches.push(candidate)
    }
    if (matches.length===1) { page=matches[0]; return true }
    return false
  }).toBe(true)
  await page.waitForLoadState('domcontentloaded')
  await expect(page.getByRole('heading',{name:'Mis evaluaciones',exact:true})).toBeVisible()
  const standalone = await page.evaluate(() => matchMedia('(display-mode: standalone)').matches)
  if (!standalone) throw new Error('The installed window did not open in standalone mode')
  return page
}

try {
  context = await chromium.launchPersistentContext(profile,options)
  const anchor = context.pages()[0] ?? await context.newPage()
  await anchor.goto(`${base.origin}/acceso`)
  await anchor.evaluate(async () => { await navigator.serviceWorker.ready })
  await anchor.locator('#login-email').fill('browser-a@kneeoai.invalid')
  await anchor.locator('#login-password').fill('BrowserTestsOnly-2026!')
  await anchor.getByRole('button',{name:'Iniciar sesión',exact:true}).click()
  await expect(anchor.getByRole('heading',{name:'Mis evaluaciones',exact:true})).toBeVisible()
  await anchor.getByRole('button',{name:'Nueva evaluación',exact:true}).click()
  await anchor.locator('#draft-age').fill('62')
  const saved = anchor.waitForResponse(response => response.request().method()==='POST' && new URL(response.url()).pathname==='/api/v1/evaluations')
  await anchor.getByRole('button',{name:'Guardar y salir',exact:true}).click()
  const response = await saved
  if (response.status()!==201) throw new Error('The synthetic case was not saved')
  const evaluation = await response.json()
  const manifest = await (await anchor.request.get(`${base.origin}/manifest.webmanifest`)).json()
  manifestId = new URL(manifest.id,base.origin).href
  cdp = context.browser() ? await context.browser().newBrowserCDPSession() : await context.newCDPSession(anchor)
  await cdp.send('PWA.install',{manifestId,installUrlOrBundleUrl:`${base.origin}/acceso`})
  installed=true; summary.installation=true
  // This sets only the isolated test profile's "open as window" user preference.
  // The production manifest's standalone display is checked separately in pwa.spec.ts.
  await cdp.send('PWA.changeAppUserSettings',{manifestId,displayMode:'standalone'})
  summary.testProfileOpenAsWindow=true
  const app = await launchApp(context,cdp)
  summary.standalone=true
  await expect(app.locator('#pwa-installed')).toBeVisible()
  await expect(app.getByRole('link',{name:`Ver resumen del caso ${evaluation.code}`,exact:true})).toBeVisible()
  await app.screenshot({path:resolve(output,'installed.png'),fullPage:true})
  await app.close()
  await context.close(); context=null

  context = await chromium.launchPersistentContext(profile,options)
  const secondAnchor = context.pages()[0] ?? await context.newPage()
  cdp = context.browser() ? await context.browser().newBrowserCDPSession() : await context.newCDPSession(secondAnchor)
  const reopened = await launchApp(context,cdp)
  summary.reopenedAfterBrowserRestart=true
  await expect(reopened.getByRole('link',{name:`Ver resumen del caso ${evaluation.code}`,exact:true})).toBeVisible()
  summary.savedCaseAndSessionRecovered=true
  await expect(reopened.getByRole('button',{name:'Calcular predicción',exact:true})).toHaveCount(0)
  await reopened.screenshot({path:resolve(output,'reopened.png'),fullPage:true})
  await cdp.send('PWA.uninstall',{manifestId}); installed=false
  summary.testInstallationRemoved=true
  summary.passed=true
} catch(error) {
  summary.error=String(error)
} finally {
  if (installed && cdp) {
    try { await cdp.send('PWA.uninstall',{manifestId}); summary.testInstallationRemoved=true }
    catch { summary.testInstallationRemoved=false }
  }
  if (context) await context.close()
  await writeFile(resolve(output,'verification.json'),`${JSON.stringify(summary,null,2)}\n`)
  console.log(JSON.stringify(summary))
}
if (!summary.passed) process.exitCode=1
