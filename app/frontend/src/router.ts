import { nextTick } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import LoginView from './views/LoginView.vue'
import EvaluationsView from './views/EvaluationsView.vue'
import NewEvaluationView from './views/NewEvaluationView.vue'
import RadiographView from './views/RadiographView.vue'
import CropReviewView from './views/CropReviewView.vue'
import CaseSummaryView from './views/CaseSummaryView.vue'
import { refreshSession } from './state/session'
import { loadEvaluation, loadEvaluations } from './state/evaluations'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/acceso' },
    { path: '/acceso', name: 'login', component: LoginView, meta: { title: 'Acceso' } },
    { path: '/evaluaciones', name: 'evaluations', component: EvaluationsView, meta: { title: 'Mis evaluaciones', authRequired: true } },
    { path: '/evaluaciones/nueva', name: 'new-evaluation', component: NewEvaluationView, meta: { title: 'Nueva evaluación', authRequired: true } },
    { path: '/evaluaciones/:id/editar', name: 'edit-evaluation', component: NewEvaluationView, meta: { title: 'Editar borrador', authRequired: true } },
    { path: '/evaluaciones/:id/radiografia', name: 'radiograph', component: RadiographView, meta: { title: 'Radiografía y lado', authRequired: true } },
    { path: '/evaluaciones/:id/revision', name: 'crop-review', component: CropReviewView, meta: { title: 'Revisión del recorte', authRequired: true } },
    { path: '/evaluaciones/:id/resumen', name: 'case-summary', component: CaseSummaryView, meta: { title: 'Resumen del caso', authRequired: true } },
    { path: '/:pathMatch(.*)*', redirect: '/acceso' },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

router.beforeEach(async to => {
  let authenticated = false
  try { authenticated = (await refreshSession()).authenticated } catch {
    if (to.meta.authRequired) return { name: 'login', replace: true }
    return
  }
  if (to.meta.authRequired && !authenticated) return { name: 'login', replace: true }
  if (to.name === 'login' && authenticated) return { name: 'evaluations', replace: true }
  if (!authenticated) return
  try {
    if (to.name === 'evaluations') await loadEvaluations()
    else if (typeof to.params.id === 'string') await loadEvaluation(to.params.id)
  } catch { /* The destination shows the server error; no fictional fallback is loaded. */ }
})

router.afterEach(async to => {
  document.title = `KneeOAI · ${String(to.meta.title ?? 'Prototipo')}`
  await nextTick()
  document.querySelector<HTMLElement>('h1')?.focus({ preventScroll: true })
})

export default router
