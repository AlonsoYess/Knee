<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue'
const dialog = ref<HTMLDialogElement | null>(null)
let resolveAnswer: ((answer: boolean) => void) | null = null
function answer(value: boolean) { const resolve = resolveAnswer; resolveAnswer = null; dialog.value?.close(); resolve?.(value) }
function ask(): Promise<boolean> {
  if (resolveAnswer) return Promise.resolve(false)
  return new Promise(resolve => { resolveAnswer = resolve; dialog.value?.showModal() })
}
defineExpose({ ask })
onBeforeUnmount(() => { resolveAnswer?.(false); resolveAnswer = null })
</script>
<template>
  <dialog ref="dialog" class="app-dialog" aria-labelledby="unsaved-changes-title" @cancel.prevent="answer(false)" @close="answer(false)">
    <h2 id="unsaved-changes-title">Tienes cambios sin guardar</h2>
    <p>La selección o las observaciones de esta pantalla todavía no se han confirmado en el servidor.</p>
    <div class="workflow-actions"><button type="button" class="button button-secondary" autofocus @click="answer(false)">Seguir editando</button><button type="button" class="button button-primary" @click="answer(true)">Descartar cambios</button></div>
  </dialog>
</template>
