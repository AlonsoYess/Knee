# Control de cambios metodológicos

## Control del documento

| Campo | Valor |
| --- | --- |
| Documento | Mecanismo de registro, evaluación y autorización de cambios metodológicos |
| Estado | Aprobado por el investigador; vigente |
| Versión | 1.0 |
| Fecha de aprobación | 27 de septiembre de 2026 |
| Paso | Fase 0, paso 4 |
| Contrato rector | `docs/00_ALCANCE_Y_TRAZABILIDAD.md`, versión 1.1 |
| Reglas vigentes | `docs/01_REGLAS_INVARIABLES.md`, versión 1.0 |
| Registro ejecutable | `configs/governance/methodology_change_log.json` |
| Plantilla | `docs/templates/SOLICITUD_CAMBIO_METODOLOGICO.md` |
| Destino de respaldo | Drive privado autorizado; ubicación exacta en el inventario privado |
| Estado del respaldo | Documento, plantilla y registro 1.0 sincronizados, verificados y privados en Google Drive |
| Entrenamiento autorizado | No |

## 1. Objetivo

Este mecanismo impide que una mejora, dificultad técnica o resultado experimental modifique silenciosamente la metodología. Toda propuesta metodológica deberá existir como registro independiente **antes** de cambiar código, datos, particiones, configuración experimental, entrenamiento, evaluación o prototipo.

El mecanismo no convierte una propuesta en autorización. La aprobación corresponde al investigador y debe quedar registrada expresamente. Una propuesta incompleta, no aprobada, rechazada o retirada permanece bloqueada.

## 2. Alcance del mecanismo

### 2.1 Cambios que obligatoriamente requieren una solicitud MCR

Entre otros:

- modificar población, criterios de inclusión, unidad de análisis, horizonte o desenlace;
- agregar, retirar o transformar predictores del análisis principal;
- sustituir modelos comprometidos o incorporar arquitecturas, ensambles o aprendizaje adicional;
- cambiar resolución, región anatómica, estrategia de recorte o preprocesamiento con efecto experimental;
- modificar pérdida, balance, particiones, pliegues, semillas, calibración, umbral o métrica principal;
- alterar comparadores, contrastes, multiplicidad, bootstrap o regla de superioridad;
- usar información de prueba para tomar decisiones;
- ampliar las salidas o el uso declarado del prototipo;
- reducir el protocolo por falta de datos, tiempo o capacidad de cómputo.

### 2.2 Acciones que no son cambios metodológicos

Correcciones ortográficas, reorganización interna sin cambio de comportamiento, pruebas equivalentes, documentación de hechos ya ejecutados y optimizaciones puramente técnicas pueden registrarse en `docs/DECISIONES.md` y en el historial de Git. Si existe duda razonable sobre su impacto experimental, se tratarán como MCR.

## 3. Clasificación de impacto

| Clase | Definición | Ejemplos | Requisito |
| --- | --- | --- | --- |
| `M1` | Ajuste operativo dentro del protocolo, con posible efecto en reproducibilidad o resultados. | Parámetros de recorte, reducción del espacio de búsqueda, capas descongeladas. | Solicitud, análisis y aprobación previa. |
| `M2` | Cambio metodológico que modifica una decisión escrita del capítulo III o una regla invariable. | Pérdida, arquitectura, entradas, calibración, partición. | Aprobación previa y actualización trazable de metodología y contratos. |
| `M3` | Cambio de alcance o inferencia que afecta objetivos, hipótesis, población, desenlace o prueba reservada. | Nuevo desenlace, cohorte diferente, reapertura de prueba, salida clínica adicional. | No aplicar sin revisión académica integral y aprobación expresa. |
| `M4` | Propuesta posterior a la apertura de prueba motivada por sus resultados. | Reajustar el modelo tras observar PR-AUC de prueba. | Prohibida para el análisis confirmatorio; solo podría definirse como estudio nuevo y separado. |

La clase no se reduce para facilitar la aprobación. Ante dos clases posibles se aplica la de mayor impacto.

## 4. Ciclo de vida obligatorio

```text
borrador
  -> en_analisis
  -> esperando_aprobacion
      -> aprobada_no_aplicada
          -> implementada
              -> implementada_y_verificada
      -> rechazada
      -> retirada
```

Reglas del ciclo:

1. `borrador`, `en_analisis` y `esperando_aprobacion` no autorizan cambios.
2. Solo `aprobada_no_aplicada` permite comenzar la implementación descrita, sin ampliar lo aprobado.
3. `rechazada` y `retirada` bloquean la aplicación y conservan el razonamiento histórico.
4. `implementada` exige verificación antes de usarse en una ejecución experimental válida.
5. `implementada_y_verificada` debe señalar exactamente los documentos, configuraciones, pruebas y versiones modificados.
6. Una propuesta no cambia de contenido después de ser aprobada. Si cambia la solución o su impacto, se crea una nueva versión o solicitud.

## 5. Puertas de control

### Puerta 1. Registro previo

La solicitud recibe un identificador `MCR-AAAA-NNN`, fecha, autor, problema observado y regla afectada. No debe existir implementación previa, salvo un registro histórico marcado expresamente como tal.

### Puerta 2. Evidencia y alternativas

Debe incluir evidencia verificable, alternativa propuesta, al menos una alternativa evaluada y explicación de por qué el protocolo vigente no sería suficiente.

### Puerta 3. Evaluación de impacto

Se examinan por separado:

- capítulos I, II y III;
- problemas, objetivos e hipótesis;
- cohorte, variables, etiqueta, imágenes y exclusiones;
- comparabilidad de escenarios y validez estadística;
- riesgo de fuga, sobreajuste, sesgo y optimismo;
- calibración, interpretabilidad y prototipo;
- tiempo, cómputo, almacenamiento y reproducibilidad;
- documentos, código, configuración y pruebas que cambiarían.

### Puerta 4. Protección de la prueba

La solicitud debe confirmar que el bloque de prueba no fue consultado para formular, elegir o justificar la propuesta. Si fue consultado, la propuesta no puede modificar el análisis confirmatorio vigente.

### Puerta 5. Decisión explícita

El investigador registra `aprobada`, `rechazada` o `retirada`, fecha, alcance exacto y condiciones. Silencio, continuidad de trabajo o una aprobación de otro paso no equivalen a aprobar una MCR.

### Puerta 6. Actualización antes de aplicar

Cuando corresponda, primero se actualizan tesis, contrato de alcance, reglas, configuraciones y plan. Solo después puede modificarse la implementación.

### Puerta 7. Verificación posterior

Se ejecutan pruebas, se comprueba que el cambio aplicado coincide con el autorizado y se registra su efecto. La prueba reservada sigue cerrada salvo que el protocolo ya congelado autorice su apertura.

## 6. Responsabilidades

| Rol | Responsabilidad |
| --- | --- |
| Investigador | Decide aprobar, rechazar o retirar; define condiciones y asume la actualización académica cuando corresponda. |
| Codex | Detecta posibles cambios, prepara el análisis, registra evidencia y no implementa una MCR sin aprobación explícita. |
| Registro JSON | Conserva estados y decisiones en formato verificable automáticamente. |
| Pruebas automáticas | Bloquean estados incoherentes, aprobaciones incompletas y registros que admitan consulta previa de prueba. |

## 7. Regla de bloqueo

Una propuesta está autorizada para comenzar solo cuando:

- el registro completo es válido;
- su estado es `aprobada_no_aplicada`;
- la decisión es `aprobada` por el investigador;
- existe fecha de decisión y alcance aprobado;
- el bloque de prueba no fue consultado;
- se identificaron documentos y artefactos que deben actualizarse.

`implementada` e `implementada_y_verificada` son estados posteriores y no autorizaciones reutilizables para cambios adicionales.

La herramienta `knee-change-check` validará el registro y, cuando se indique una solicitud, responderá `AUTHORIZED` o `BLOCKED`. Un resultado `AUTHORIZED` no autoriza entrenamiento general: solo el cambio exacto descrito.

## 8. Registro histórico inicial

`MCR-2026-001` documenta de forma retrospectiva y transparente la decisión `RES-KL-001`: retirar la salida de KL estimado. Se marca como registro histórico porque la decisión ocurrió antes de crear este mecanismo. No se modificó un modelo, no se ejecutó entrenamiento y el Word será corregido posteriormente por el investigador.

Este antecedente no permite registrar retroactivamente futuros cambios como práctica ordinaria.

## 8.1 Primera propuesta prospectiva aplicada

`MCR-2026-002` registra la decisión del investigador del 28 de septiembre de 2026 de ampliar, antes del entrenamiento, los candidatos radiográficos y de fusión multimodal. La propuesta conserva las líneas base del Capítulo III, agrega un registro cerrado de modelos modernos, mantiene intactos población, predictores, desenlace, particiones, métricas y prueba reservada, y exige revisión de licencias y pesos. Esta decisión no autoriza entrenamiento: únicamente autoriza actualizar contrato, reglas, estrategia, configuración y pruebas que preparan las fases posteriores.

## 9. Condición de cierre de la Fase 0, paso 4

El paso se cerrará cuando:

1. el investigador apruebe el mecanismo y sus clases;
2. existan plantilla y registro JSON coherentes;
3. el validador bloquee solicitudes incompletas o no aprobadas;
4. las pruebas automáticas sean satisfactorias;
5. las copias versionadas estén verificadas y privadas en Drive.

Las cinco condiciones se cumplieron el 27 de septiembre de 2026. La aprobación expresa del investigador cierra la **Fase 0, paso 4 en la versión 1.0**. Esta aprobación adopta el mecanismo de control, pero no autoriza por sí sola ninguna propuesta metodológica nueva ni entrenamiento.
