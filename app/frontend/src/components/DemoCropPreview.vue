<script setup lang="ts">
import { computed, useId } from 'vue'
import type { DemoCropFixture } from '../demo/demoCrops'

const props = defineProps<{ fixture: DemoCropFixture; view: 'original' | 'crop' }>()
const emit = defineEmits<{ load: []; error: [] }>()
const clipId = `demo-crop-${useId()}`
const viewBox = computed(() => props.view === 'original' ? '0 0 960 720' : `${props.fixture.box.x} ${props.fixture.box.y} ${props.fixture.box.width} ${props.fixture.box.height}`)
</script>

<template>
  <svg class="demo-crop-viewport" :viewBox="viewBox" role="img" :aria-label="`${view === 'original' ? 'Imagen sintética con marco de ejemplo' : 'Recorte sintético'} · ${fixture.label}`" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
    <defs v-if="view === 'crop'"><clipPath :id="clipId" clipPathUnits="userSpaceOnUse"><rect :x="fixture.box.x" :y="fixture.box.y" :width="fixture.box.width" :height="fixture.box.height" /></clipPath></defs>
    <image :key="fixture.sourceSrc" :href="fixture.sourceSrc" :clip-path="view === 'crop' ? `url(#${clipId})` : undefined" x="0" y="0" width="960" height="720" @load="emit('load')" @error="emit('error')" />
    <template v-if="view === 'original'">
      <rect :x="fixture.box.x" :y="fixture.box.y" :width="fixture.box.width" :height="fixture.box.height" fill="#5bcbce" fill-opacity=".08" stroke="#8de5e5" stroke-width="4" />
      <rect :x="fixture.box.x" :y="fixture.box.y - 34" width="190" height="30" rx="4" fill="#183a43" />
      <text :x="fixture.box.x + 9" :y="fixture.box.y - 13" fill="#dcffff" font-size="17" font-family="Segoe UI,Arial,sans-serif">MARCO DE EJEMPLO</text>
    </template>
  </svg>
</template>

<style scoped>
.demo-crop-viewport { display: block; width: 100%; height: 100%; min-width: 0; background: #12212e; border-radius: 8px; }
</style>
