export type DemoImageId = 'bilateral-a' | 'bilateral-b'
export interface DemoImageState {
  sampleId: '' | DemoImageId
  knee: '' | 'right' | 'left'
  sideConfirmed: boolean
}

// Native vector fixtures authored only to explore this UI. No clinical pixels or metadata.
export const demoImages: readonly {
  id: DemoImageId; label: string; fileName: string; src: string; description: string; width: number; height: number
}[] = [
  { id: 'bilateral-a', label: 'Imagen A', fileName: 'radiografia-demo-a.svg', src: '/demo/radiografia-demo-a.svg', description: 'Esquema bilateral sintético A. No es una radiografía clínica.', width: 960, height: 720 },
  { id: 'bilateral-b', label: 'Imagen B', fileName: 'radiografia-demo-b.svg', src: '/demo/radiografia-demo-b.svg', description: 'Esquema bilateral sintético B. No es una radiografía clínica.', width: 960, height: 720 },
]
