import { nextTick } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import LoginView from './views/LoginView.vue'
import EvaluationsView from './views/EvaluationsView.vue'
import NewEvaluationView from './views/NewEvaluationView.vue'
import RadiographView from './views/RadiographView.vue'
import CropReviewView from './views/CropReviewView.vue'
import CaseSummaryView from './views/CaseSummaryView.vue'
import { demoSessionActive } from './demoSession'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/acceso' },
    { path: '/acceso', name: 'login', component: LoginView, meta: { title: 'Acceso' } },
    { path: '/evaluaciones', name: 'evaluations', component: EvaluationsView, meta: { title: 'Mis evaluaciones', demoRequired: true } },
    { path: '/evaluaciones/nueva', name: 'new-evaluation', component: NewEvaluationView, meta: { title: 'Nueva evaluación', demoRequired: true } },
    { path: '/evaluaciones/:id/editar', name: 'edit-evaluation', component: NewEvaluationView, meta: { title: 'Editar borrador', demoRequired: true } },
    { path: '/evaluaciones/:id/radiografia', name: 'radiograph', component: RadiographView, meta: { title: 'Radiografía y lado', demoRequired: true } },
    { path: '/evaluaciones/:id/revision', name: 'crop-review', component: CropReviewView, meta: { title: 'Revisión del recorte', demoRequired: true } },
    { path: '/evaluaciones/:id/resumen', name: 'case-summary', component: CaseSummaryView, meta: { title: 'Resumen del caso', demoRequired: true } },
    { path: '/:pathMatch(.*)*', redirect: '/acceso' },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

router.beforeEach(to => {
  if (to.meta.demoRequired && !demoSessionActive.value) return { name: 'login', replace: true }
  if (to.name === 'login' && demoSessionActive.value) return { name: 'evaluations', replace: true }
})

router.afterEach(async to => {
  document.title = `KneeOAI · ${String(to.meta.title ?? 'Prototipo')}`
  await nextTick()
  document.querySelector<HTMLElement>('h1')?.focus({ preventScroll: true })
})

export default router
