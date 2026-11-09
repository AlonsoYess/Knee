# Fase 6, paso 1 — conexión técnica local

Fecha: 9 de noviembre de 2026. Alcance: FastAPI y un proceso científico separado, con modelos clínicos de desarrollo ya guardados. Esta entrega no habilita predicciones para el médico ni cambia el procedimiento científico.

## Qué funciona

- El backend puede verificar, mediante otro intérprete Python, los cinco pliegues guardados del modelo clínico principal `clinical_logistic`.
- El preparador `knee_app.predictor_bundle` copia exclusivamente los cinco archivos de pesos JSON y sus cinco estados clínicos a un directorio privado nuevo. Verifica índice, recibos de entrenamiento de desarrollo, hashes y doce fuentes científicas; no abre cohortes ni selecciona por rendimiento.
- `predictor_worker.py` comprueba el manifiesto fijado por el servidor y los archivos antes y después de cada operación. Reutiliza `sensitivity_real_inference._transform_baseline` y `_logistic_logits` sin copiar sus cálculos.
- `preflight` carga y verifica los cinco modelos sin procesar casos. `diagnostic` usa solo tres entradas ficticias fijadas en el código. No recibe datos del médico, imágenes, identificadores de participantes ni desenlaces.
- FastAPI conserva `GET /api/v1/engine-status` autenticado. Con el preflight correcto informa que la conexión local está comprobada, con `status=not_configured` y `package=null`: todavía falta el paquete predictivo final.
- Una configuración parcial, un archivo cambiado, un enlace, una dependencia incompatible, salida ambigua o un fallo del proceso producen indisponibilidad explícita. Las respuestas HTTP no revelan rutas, pesos, logits ni errores internos.

La separación es: **FastAPI → proceso técnico → funciones científicas existentes → estado de comprobación**. El intérprete de la web no importa `knee`, NumPy ni PyTorch para esta conexión.

## Evidencia real

| Comprobación | Resultado |
| --- | --- |
| Backend completo, incluidos conexión, worker y preparador reproducible | 330 pruebas correctas, sin omisiones. |
| Carga real local | Cinco modelos clínicos guardados y cinco estados verificados. |
| Diagnóstico fijo | Tres entradas ficticias por pliegue: quince logits técnicos. |
| Paridad con las funciones originales | Quince comparaciones coincidentes; diferencia absoluta máxima 0.0. |
| FastAPI real con base ficticia nueva | Sesión requerida; preflight autenticado correcto; sin importaciones científicas en el proceso API. |
| Predicción clínica | POST de ejecución sigue respondiendo 503 y no cambia la evaluación. |
| Integridad | Pesos, estados y doce fuentes científicas conservaron sus hashes. |

La paridad comprueba el transporte y la reutilización del cálculo existente. No mide capacidad predictiva, calibración, utilidad clínica ni rendimiento en una población nueva. Los valores KL y otros campos auxiliares de las tres fixtures son ficticios y solo completan el contrato interno del helper; no se añaden a las entradas del producto.

## Uso reproducible

Desde la raíz del repositorio, con rutas privadas explícitas y el entorno científico existente:

```powershell
$env:PYTHONPATH=(Resolve-Path app/backend).Path
app/backend/.venv/Scripts/python.exe -m knee_app.predictor_bundle --artifact-index INDICE_PRIVADO_ABSOLUTO --artifact-index-sha256 SHA256_VERIFICADO --checkout CHECKOUT_CIENTIFICO_ABSOLUTO --python PYTHON_CIENTIFICO_ABSOLUTO --output DIRECTORIO_PRIVADO_NUEVO
app/backend/.venv/Scripts/python.exe -m knee_app.predictor_check --config CONFIG_PRIVADA_ABSOLUTA --report INFORME_PRIVADO_NUEVO
app/backend/.venv/Scripts/python.exe -m knee_app.predictor_check --config CONFIG_PRIVADA_ABSOLUTA --report OTRO_INFORME_PRIVADO_NUEVO --diagnostic
```

El preparador escribe `bundle/`, `config_private.json` y trazabilidad en su salida privada. El índice debe ser el registro seleccionado de artefactos de desarrollo, autenticado por su SHA externo; no acepta un archivo de participantes como sustituto. La configuración conserva solo cuatro referencias: intérprete, checkout científico, bundle y SHA del manifiesto.

Para activar el preflight del backend local se configuran explícitamente `KNEEOAI_PREDICTOR_PYTHON`, `KNEEOAI_PREDICTOR_CHECKOUT`, `KNEEOAI_PREDICTOR_BUNDLE` y `KNEEOAI_PREDICTOR_MANIFEST_SHA256`. `KNEEOAI_PREDICTOR_TIMEOUT_SECONDS` limita el tiempo, con valor predeterminado de 30 y máximo de 120 segundos. Sin configuración, se mantiene el comportamiento anterior.

Se necesitan Python 3.12.14 y NumPy 2.5.3 en el entorno de esta comprobación. No se instala nada ni se adapta silenciosamente una versión. El proceso bloquea red, escrituras, procesos adicionales y lecturas Python ajenas a los archivos fijados o al intérprete. Esta protección evita I/O accidental en las funciones autenticadas; no se presenta como aislamiento frente a código nativo malicioso.

Los informes detallados son privados y nuevos: no sobrescriben configuración, modelos, checkout o distribución pública. El resumen de consola omite rutas y logits. Los recibos completos y las identidades de entrenamiento no se copian al bundle ni se publican.

## Qué falta antes de calcular para el médico

Los archivos auditados corresponden a pliegues de desarrollo. El plan vigente aún conserva sin autorizar la ejecución del ajuste final y de la calibración/umbral finales; los resultados de calibración exploratoria no se convierten en un paquete liberado por esta integración. No se ejecutó ninguna de esas operaciones.

No se adopta un pliegue como modelo final ni se promedian pliegues para sustituir ese paquete. Tampoco se convierte un logit en probabilidad clínica o se fija un umbral arbitrario. El contrato de resultado sigue exigiendo paquete formalizado, probabilidad calibrada a 48 meses y versiones trazables de transformación, calibrador y regla de clasificación.

La conexión del radiográfico y del multimodal todavía no se ha probado en este paso. Después del paquete final deberán conectarse ejecución y resultados persistentes, revisión QC aceptada y vigente, integridad del recorte, manejo de cambios del caso y presentación en Vue. El botón de cálculo permanece deshabilitado; no se crean tablas ni resultados clínicos provisionales.

La imagen de despliegue de fase 5 permanece como distribución de aplicación. Su entrada estricta rechaza `KNEEOAI_PREDICTOR_*`, y esta entrega no configura ni prueba un despliegue científico Docker. Tampoco publica la web, contrata alojamiento ni compra dominio.

No se reentrenó, ajustó calibración, eligió umbral, modificó metodología ni abrió la prueba reservada. La base del usuario y los activos originales no fueron migrados ni modificados.
