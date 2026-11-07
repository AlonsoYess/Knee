import { spawn } from 'node:child_process'
import { join } from 'node:path'
import buildPwaFixtures, { frontend } from './build-pwa-fixtures.mjs'

await buildPwaFixtures()
const python = process.env.KNEEOAI_TEST_PYTHON ?? join(frontend, '../backend/.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
const child = spawn(python, ['tests/support/pwa_server.py'], { cwd: frontend, stdio: 'inherit', windowsHide: true })
child.on('error', error => { console.error(error); process.exit(1) })
child.on('exit', code => process.exit(code ?? 1))
