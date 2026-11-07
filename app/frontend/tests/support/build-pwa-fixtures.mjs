import { build } from 'vite'
import { fileURLToPath } from 'node:url'
import { resolve, relative } from 'node:path'
import { lstat, mkdir } from 'node:fs/promises'

export const frontend = fileURLToPath(new URL('../../', import.meta.url))
export const builds = resolve(frontend, 'test-results/pwa-builds')

export default async function buildPwaFixtures() {
  await mkdir(builds, { recursive: true })
  if ((await lstat(builds)).isSymbolicLink()) throw new Error('Fixture builds must remain in their test directory')
  for (const version of ['a', 'b']) {
    const destination = resolve(builds, version)
    if (relative(builds, destination) !== version) throw new Error('Unsafe test build destination')
    try { if ((await lstat(destination)).isSymbolicLink()) throw new Error('Fixture build must not use a link') }
    catch (error) { if (error.code !== 'ENOENT') throw error }
    process.env.KNEEOAI_BUILD_ID = `pwa-test-${version}`
    await build({ root: frontend, configFile: resolve(frontend, 'vite.config.ts'), logLevel: 'error',
      build: { outDir: destination, emptyOutDir: true } })
  }
  delete process.env.KNEEOAI_BUILD_ID
}
