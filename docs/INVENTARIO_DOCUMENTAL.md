# Inventario documental público

## Control

| Campo | Valor |
| --- | --- |
| Estado | Activo |
| Versión | 1.19 pública |
| Fecha de corte | 1 de octubre de 2026 |
| Ámbito | Artefactos rectores y avance controlado hasta la Fase 1, paso 4 |
| Inventario exacto | Privado en Google Drive autorizado |

Este archivo es el resumen publicable del control documental. El inventario privado conserva nombres de archivo completos, enlaces, identificadores de Drive, tamaños, huellas SHA-256 y ubicación exacta de cada versión. Esa evidencia no se replica en GitHub.

## Versiones vigentes

| Documento | Versión/estado | Ubicación en Git | Respaldo privado |
| --- | --- | --- | --- |
| Contrato de alcance y trazabilidad | 1.3, MCR-2026-004 aprobada | `docs/00_ALCANCE_Y_TRAZABILIDAD.md` | Nueva versión local pendiente de sincronización |
| Reglas metodológicas invariables | 1.4, MCR-2026-004 aprobada | `docs/01_REGLAS_INVARIABLES.md` | Nueva versión local pendiente de sincronización |
| Contrato ejecutable del alcance | 1.5, MCR-2026-004 aprobada y no aplicada a radiografías; condición previa satisfecha | `configs/governance/scope_contract.json` | Nueva versión local pendiente de sincronización |
| Estructura e inventario público de fuentes | 1.1 pública, ejecutado | `docs/02_ESTRUCTURA_DRIVE_Y_FUENTES.md` | Inventario exacto preservado |
| Registro público resumido de fuentes | 1.1 pública, verificado | `configs/governance/source_registry.json` | Registro exacto preservado |
| Control de cambios metodológicos | 1.0, aprobado | `docs/03_CONTROL_CAMBIOS_METODOLOGICOS.md` | Verificado |
| Plantilla de solicitud MCR | 1.0, aprobada | `docs/templates/SOLICITUD_CAMBIO_METODOLOGICO.md` | Verificado |
| Registro de cambios MCR | 1.6 local; MCR-2026-003 retirada y MCR-2026-004 aprobada, no aplicada | `configs/governance/methodology_change_log.json` | Nueva versión pendiente de sincronización; respaldos anteriores preservados |
| Arquitectura reproducible | 1.1, aprobada | `docs/04_ARQUITECTURA_REPRODUCIBLE.md` | Verificado |
| Notebook de cierre técnico en Colab | 1.1, aprobado para ejecución; ruta privada mediante secreto | `notebooks/00_arranque_colab.ipynb` | Verificado |
| Acta pública de cierre técnico | 1.0, cerrada | `docs/05_CIERRE_FASE_0.md` | Evidencias privadas verificadas |
| Estrategia de modelos | 1.0, aprobada | `docs/06_ESTRATEGIA_MODELOS.md` | Verificado |
| Protocolo y cierre del piloto DICOM | Cerrados | `docs/07_FASE_1_PASO_1.md`, `docs/08_CIERRE_FASE_1_PASO_1.md` | Evidencias privadas verificadas |
| Protocolo y avance del inventario radiográfico | Cerrados | `docs/09_FASE_1_PASO_2.md`, `docs/10_AVANCE_FASE_1_PASO_2.md` | Evidencias privadas verificadas |
| Acta pública de cierre de la Fase 1, paso 2 | 1.0, cerrada | `docs/11_CIERRE_FASE_1_PASO_2.md` | Inventario, lotes y registros privados verificados |
| Notebooks de inventario y descarga selectiva | Ejecutados | `notebooks/02_inventario_adquisiciones.ipynb`, `notebooks/03_descarga_selectiva_lote.ipynb`, `notebooks/04_descarga_lotes_pendientes.ipynb` | Copias aprobadas verificadas |
| Protocolo y notebooks de separación bilateral | Cerrados; `bilateral_split_v0.2_pilot` congelado | `docs/12_FASE_1_PASO_3.md`, `notebooks/05_validacion_separacion_bilateral.ipynb`, `notebooks/06_cierre_revision_separacion_bilateral.ipynb` | Diez vistas aceptadas, cero exclusiones y parámetros privados verificados |
| Acta pública de cierre de la Fase 1, paso 3 | 1.0, cerrada | `docs/13_CIERRE_FASE_1_PASO_3.md` | Registro, resumen y congelamiento privados verificados |
| Protocolo, módulo y notebooks de localización tibiofemoral | `v0.1`, `v0.2`, `v0.3` y `v0.4` rechazadas | `docs/14_FASE_1_PASO_4.md`, `src/knee/joint_localization.py`, `src/knee/joint_review.py`, `notebooks/07_validacion_localizacion_tibiofemoral.ipynb`, `notebooks/08_repeticion_localizacion_tibiofemoral_v02.ipynb`, `notebooks/09_repeticion_localizacion_tibiofemoral_v03.ipynb`, `notebooks/10_cierre_revision_localizacion_tibiofemoral_v03.ipynb`, `notebooks/11_repeticion_localizacion_tibiofemoral_v04.ipynb`, `notebooks/12_cierre_revision_localizacion_tibiofemoral_v04.ipynb` | Cuatro revisiones documentadas; cierres de `v0.3` y `v0.4` verificados |
| Acta pública de revisión de `v0.1` | 1.0, rechazada después de revisión ciega | `docs/15_REVISION_FASE_1_PASO_4_V01.md` | Decisiones individuales preservadas en privado |
| Acta pública de revisión de `v0.2` | 1.0, rechazada después de revisión ciega | `docs/16_REVISION_FASE_1_PASO_4_V02.md` | Decisiones individuales preservadas en privado |
| Propuesta técnica de `v0.3` | 0.1, antecedente aprobado | `docs/17_PROPUESTA_FASE_1_PASO_4_V03.md` | Dio origen a la implementación determinista autorizada |
| Implementación de `v0.3` | 0.1, ejecutada y rechazada | `docs/18_IMPLEMENTACION_FASE_1_PASO_4_V03.md`, `configs/joint_localization.v0.3.example.json`, `notebooks/09_repeticion_localizacion_tibiofemoral_v03.ipynb` | Se ejecutó sobre las mismas veinte rodillas desde una revisión Git identificable |
| Acta pública de revisión de `v0.3` | 1.0, rechazada y cerrada reproduciblemente | `docs/19_REVISION_FASE_1_PASO_4_V03.md`, `notebooks/10_cierre_revision_localizacion_tibiofemoral_v03.ipynb` | Decisiones individuales y cierre verificados en privado |
| Propuesta técnica de `v0.4` | 0.1, antecedente aprobado | `docs/20_PROPUESTA_FASE_1_PASO_4_V04.md` | Dio origen a la implementación determinista autorizada |
| Implementación de `v0.4` | 0.1, ejecutada y rechazada | `docs/21_IMPLEMENTACION_FASE_1_PASO_4_V04.md`, `configs/joint_localization.v0.4.example.json`, `notebooks/11_repeticion_localizacion_tibiofemoral_v04.ipynb` | Se ejecutó sobre las mismas veinte rodillas desde una revisión Git identificable |
| Acta pública de revisión de `v0.4` | 1.0 con seguimiento; rechazada y cerrada reproduciblemente | `docs/22_REVISION_FASE_1_PASO_4_V04.md`, `notebooks/12_cierre_revision_localizacion_tibiofemoral_v04.ipynb` | CSV completado por nueva revisión asistida; ambos JSON contrastados tras ejecución del investigador en Colab |
| Solicitud de localización anatómica, clase M2 | MCR-2026-003, documental 1.2; retirada por sustitución, historial preservado | `docs/23_MCR_2026_003_LOCALIZACION_ANATOMICA.md` | Preparada localmente; pendiente de respaldo privado |
| Acta de aprobación M2 y comprobaciones previas | 1.0; G0 incompleta, G1/G2 no acreditadas | `docs/24_APROBACION_M2_Y_COMPROBACIONES_PREVIAS.md` | Preparada localmente; pendiente de respaldo privado |
| Revisión M2 de ROI y evaluación técnica | MCR-2026-004, documental 1.1; contenido técnico 1.0 aprobado, no aplicado | `docs/25_MCR_2026_004_RECORTE_ROI_Y_REVISION_TECNICA.md` | Preparada localmente; no publicada ni respaldada en Drive |

| Aprobación MCR-2026-004 y seguimiento del cierre | 1.0 con seguimiento; condición 8.2 satisfecha, piloto nuevo pendiente | `docs/26_APROBACION_MCR_2026_004_Y_CIERRE_PENDIENTE.md` | Ambos JSON contrastados en Drive; histórico inicial conservado |
| Candidato MCR-2026-004 y regresión histórica | Código preparado, 200 pruebas satisfactorias en entorno aislado; no aplicado a radiografías | `src/knee/roi_mcr004.py`, `src/knee/roi_mcr004_pilot.py`, `src/knee/third_party/emory_hiti/`, `configs/roi_mcr004_historical.example.json`, `requirements-mcr004.txt`, `notebooks/13_regresion_historica_roi_mcr004.ipynb` | Publicación autorizada; copia privada de libreta 13 verificada en `03_notebooks`; regresión y revisión técnica pendientes |
| Acta de implementación MCR-2026-004 | 1.1, preparación, verificación sintética y publicación autorizada | `docs/27_IMPLEMENTACION_LOCAL_MCR004.md` | Sin evaluación radiográfica ni cambio de umbrales |

## Historial preservado

Drive conserva, sin sobrescritura, las revisiones previas del contrato de alcance, reglas invariables, contrato ejecutable, control de cambios, plantilla MCR, registro MCR, README, plan de implementación, guía de Fase 0 y registro de decisiones. La relación exacta de versiones, ubicaciones y huellas se mantiene en el inventario privado.

## Instantáneas del repositorio

Se verificaron copias privadas del README principal, README del prototipo, plan de implementación, guía de Fase 0, decisiones, arquitectura y notebook. Las revisiones superadas permanecen en el historial de Drive.

## Evidencia de verificación

- los artefactos vigentes y las versiones históricas están en sus áreas privadas esperadas;
- los metadatos de Drive indican que los archivos verificados no están compartidos públicamente;
- los doce archivos fuente únicos fueron controlados mediante el inventario privado;
- las pruebas locales incluyen contratos de gobernanza y del notebook de Colab;
- los pasos 1 a 6 están cerrados y versionados;
- la ejecución técnica en Colab/Drive terminó correctamente y sus dos evidencias privadas fueron contrastadas;
- las 1,916 adquisiciones basales están disponibles y legibles, la cola está vacía y los veinte lotes conservan evidencia verificable;
- el paso 3 quedó cerrado con una entrada ciega al desenlace, diez vistas revisadas y aceptadas, una guarda de límites verificada y los parámetros de la versión 0.2 congelados antes del procesamiento masivo;
- el paso 4 conserva las ejecuciones y revisiones ciegas de `v0.1`, `v0.2`, `v0.3` y `v0.4`: 9/20, 8/20, 0/20 y 8/20 recortes íntegramente aceptables, respectivamente, sin exclusiones técnicas ni congelamiento de parámetros;
- el cierre reproducible de la revisión de `v0.3` fue ejecutado y verificado sin habilitar ninguna operación posterior;
- la corrección `v0.4` fue ejecutada y rechazada; la libreta de cierre se ejecutó en Colab y los dos JSON se contrastaron, sin publicar decisiones individuales;
- el candidato MCR-2026-004 está preparado, probado sintéticamente y autorizado para publicación; la libreta 13 tiene copia privada verificada, pero no se aplicó a radiografías y la regresión histórica y la revisión técnica permanecen pendientes;
- no se ejecutó entrenamiento ni se abrió el conjunto de prueba reservado.

El inventario público se ampliará solo con información necesaria para reproducibilidad y gobierno. Credenciales, identificadores individuales, rutas privadas, huellas de fuentes restringidas y enlaces privados nunca se publicarán.
