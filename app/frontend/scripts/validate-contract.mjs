import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import SwaggerParser from '@apidevtools/swagger-parser'
import Ajv2020 from 'ajv/dist/2020.js'
import addFormats from 'ajv-formats'

const source = JSON.parse(await readFile(new URL('../../contracts/openapi.json', import.meta.url), 'utf8'))

// Keep validation self-contained: a contract must never load private files or URLs.
function checkReferences(value) {
  if (!value || typeof value !== 'object') return
  if (typeof value.$ref === 'string') assert.ok(value.$ref.startsWith('#/'), 'Only internal references are allowed')
  for (const child of Object.values(value)) checkReferences(child)
}
checkReferences(source)
const api = await SwaggerParser.validate(source, { resolve: { external: false } })
const ajv = new Ajv2020({ allErrors: true, strict: false })
addFormats(ajv)
let examplesChecked = 0

function checkExample(schema, value, label) {
  const validate = ajv.compile(schema)
  assert.ok(validate(value), `${label}: ${ajv.errorsText(validate.errors)}`)
  examplesChecked++
}

function checkMedia(content, label) {
  for (const [media, item] of Object.entries(content ?? {})) {
    if (!item.schema) continue
    if (Object.hasOwn(item, 'example')) checkExample(item.schema, item.example, `${label} ${media}`)
    for (const [name, example] of Object.entries(item.examples ?? {})) {
      if (Object.hasOwn(example, 'value')) checkExample(item.schema, example.value, `${label} ${media} ${name}`)
    }
  }
}

for (const [name, schema] of Object.entries(api.components?.schemas ?? {})) {
  assert.ok(ajv.validateSchema(schema), `${name}: ${ajv.errorsText(ajv.errors)}`)
  ajv.compile(schema)
  if (Object.hasOwn(schema, 'example')) checkExample(schema, schema.example, name)
  for (const example of schema.examples ?? []) checkExample(schema, example, name)
}
let operations = 0
for (const [path, item] of Object.entries(api.paths)) {
  for (const method of ['get', 'post', 'put', 'patch', 'delete', 'head', 'options']) {
    const operation = item[method]
    if (!operation) continue
    operations++
    checkMedia(operation.requestBody?.content, `${method} ${path} request`)
    for (const [status, response] of Object.entries(operation.responses)) {
      checkMedia(response.content, `${method} ${path} ${status}`)
    }
  }
}
assert.ok(examplesChecked > 0, 'The contract needs validated fictional examples')
console.log(`OpenAPI ${api.openapi}: ${operations} operations and ${examplesChecked} fictional examples validated.`)
