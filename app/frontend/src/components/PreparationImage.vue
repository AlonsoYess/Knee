<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import type { Point, PreparationDto, PreparationView, Shape } from '../api/preparationContracts'
import { loadPreparationImage } from '../state/preparations'
import { apiMessage, sessionGeneration } from '../state/session'

const props = defineProps<{ preparation: PreparationDto; view: PreparationView; label: string; nativeShape?: Shape | null; coordinateBox?: [number, number, number, number] | null; points?: Point[]; interactive?: boolean }>()
const emit = defineEmits<{ ready: [boolean]; point: [Point] }>()
const objectUrl = ref(''), error = ref(''), loaded = ref(false), nativeZoom = ref(false)
const naturalWidth = ref(0)
let controller: AbortController | null = null
let generation = 0
const box = computed<[number, number, number, number] | null>(() => props.view.startsWith('crop_') ? props.coordinateBox ?? null : props.nativeShape ? [0, 0, props.nativeShape[1], props.nativeShape[0]] : null)
const displayWidth = computed(() => nativeZoom.value ? `${props.nativeShape?.[1] ?? naturalWidth.value}px` : '100%')
function release() { controller?.abort(); controller = null; if (objectUrl.value) URL.revokeObjectURL(objectUrl.value); objectUrl.value = ''; loaded.value = false; emit('ready', false) }
watch(() => JSON.stringify([props.preparation.id, props.preparation.preparationRevisionId, props.view, sessionGeneration.value]), async () => {
  const request = ++generation; release(); error.value = ''; nativeZoom.value = false
  controller = new AbortController()
  try {
    const blob = await loadPreparationImage(props.preparation, props.view, controller.signal)
    if (request !== generation) return
    objectUrl.value = URL.createObjectURL(blob)
  } catch (failure) { if (request === generation && !controller?.signal.aborted) error.value = apiMessage(failure) }
}, { immediate: true })
function imageReady(event: Event) {
  const image = event.currentTarget as HTMLImageElement
  if (!props.nativeShape || image.naturalWidth !== props.nativeShape[1] || image.naturalHeight !== props.nativeShape[0]) {
    loaded.value = false; emit('ready', false); error.value = 'Las dimensiones de la vista no coinciden con las declaradas por el servidor. La inspección queda bloqueada.'; return
  }
  naturalWidth.value = image.naturalWidth; loaded.value = true; emit('ready', true)
}
function imageError() { loaded.value = false; emit('ready', false); error.value = 'No pudimos mostrar esta vista. La revisión queda bloqueada hasta recuperarla.' }
function capture(event: MouseEvent) {
  if (!props.interactive || !loaded.value || !box.value) return
  const rect = (event.currentTarget as HTMLImageElement).getBoundingClientRect(), coordinate = box.value
  const x = coordinate[0] + (event.clientX - rect.left) / rect.width * (coordinate[2] - coordinate[0])
  const y = coordinate[1] + (event.clientY - rect.top) / rect.height * (coordinate[3] - coordinate[1])
  emit('point', [Math.round(x * 100) / 100, Math.round(y * 100) / 100])
}
function marker(point: Point) { const b = box.value!; return { left: `${(point[0] - b[0]) / (b[2] - b[0]) * 100}%`, top: `${(point[1] - b[1]) / (b[3] - b[1]) * 100}%` } }
onBeforeUnmount(() => { generation += 1; release() })
</script>

<template>
  <figure class="preparation-image">
    <figcaption><strong>{{ label }}</strong><button type="button" class="text-button" :disabled="!loaded" @click="nativeZoom = !nativeZoom">{{ nativeZoom ? 'Ajustar al ancho' : 'Ver a tamaño nativo' }}</button></figcaption>
    <div class="preparation-image-scroll" :aria-busy="!loaded && !error">
      <div v-if="objectUrl" class="preparation-image-canvas" :style="{ width: displayWidth, maxWidth: nativeZoom ? 'none' : '100%' }">
        <img :src="objectUrl" :alt="label" :class="{ 'point-capture': interactive }" @load="imageReady" @error="imageError" @click="capture" />
        <template v-if="loaded && box"><span v-for="(point, index) in points ?? []" :key="index" class="coordinate-marker" :style="marker(point)" aria-hidden="true">{{ index + 1 }}</span></template>
      </div>
      <p v-if="!loaded && !error" role="status">Preparando vista autenticada…</p>
    </div>
    <p v-if="error" class="field-error" role="alert">{{ error }}</p>
    <p v-if="interactive" class="preparation-image-hint">Puedes marcar sobre la imagen o usar los campos de coordenadas. Las marcas no modifican el recorte.</p>
  </figure>
</template>

<style scoped>
.preparation-image { margin: 0; min-width: 0; }
figcaption { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; font-size: 13px; }
.preparation-image-scroll { overflow: auto; max-height: 650px; background: #152631; border-radius: 9px; min-height: 140px; padding: 9px; }
.preparation-image-scroll > p { color: #fff; padding: 20px; }
.preparation-image-canvas { position: relative; display: block; line-height: 0; }
img { width: 100%; height: auto; display: block; }
.point-capture { cursor: crosshair; }
.coordinate-marker { position: absolute; border: 2px solid #ffdb63; width: 24px; height: 24px; border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none; color: #ffdb63; font: 700 12px/20px sans-serif; text-align: center; background: #0007; }
.preparation-image-hint { color: var(--muted); font-size: 12px; line-height: 1.7; margin-top: 9px; }
</style>
