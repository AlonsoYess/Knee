# Aprobación de MCR-2026-004 y cierre operativo pendiente de v0.4

## 1. Decisión y alcance

- Fecha de registro: 1 de octubre de 2026.
- Aprobación expresa del investigador: «OK DALE HAGAMOSLO», en respuesta al plan de aprobar MCR-2026-004, sustituir MCR-2026-003 preservando historia, verificar cierre de v0.4 y avanzar bajo esas condiciones.
- MCR-2026-004: `aprobada_no_aplicada`. Contenido técnico 1.0 sin cambios; documento 25 revisado a 1.1 para registrar la decisión.
- MCR-2026-003: `retirada` por sustitución, sin implementación. Se preservan su aprobación del 30 de septiembre, requisitos y evaluación histórica.
- Registro MCR 1.5; contrato rector 1.3; reglas 1.4; contrato ejecutable 1.4.

El alcance autorizado es la adaptación determinista en revisión fija, ROI proporcional y revisión técnica de las secciones 4.1–4.4 de la [MCR vigente](25_MCR_2026_004_RECORTE_ROI_Y_REVISION_TECNICA.md). No requiere lector anatómico adicional ni pesos. Conserva umbrales, reserva de desarrollo y la secuencia: cierre v0.4, implementación, regresión histórica y confirmación nueva condicionada.

Objetivos, desenlace a 48 meses, predictores, escenarios, MCR-2026-001 y MCR-2026-002 no cambian. Los compromisos profesionales posteriores del prototipo/ficha no se confunden con anotación anatómica ni se eliminan.

## 2. Comprobación actual de Drive, de solo lectura

Se inspeccionó la carpeta exacta `v0_4_piloto`, comprobando su pertenencia a `paso_4_localizacion_tibiofemoral`. Su listado devolvió siete entradas directas: cinco archivos y dos carpetas. También se buscaron por nombre exacto los dos registros esperados, dentro de esa carpeta.

| Evidencia | Resultado observado el 1 de octubre de 2026 |
| --- | --- |
| `registro_preparacion.json` | Estado `ready_for_blinded_review`, revisión pendiente, veinte rodillas procesadas; ejecución del 30 de septiembre |
| `revision_visual_ciega.csv` | Veinte filas; las seis columnas SI/NO de revisión están vacías en las veinte filas |
| `registro_cierre_revision.json` | No encontrado en la carpeta comprobada |
| `resumen_revision_visual_publico.json` | No encontrado en la carpeta comprobada |
| Escrituras en Drive | Ninguna |

No se publican identificadores, enlaces privados, coordenadas ni decisiones por rodilla. La ausencia descrita corresponde a esta carpeta y este momento, no a una afirmación sobre todos los archivos del investigador.

El [acta 22](22_REVISION_FASE_1_PASO_4_V04.md) documenta 8/20 aceptables y 12/20 rechazados, nueve fallos de centrado, seis de artefactos y cero exclusiones. Ese rechazo se conserva. El CSV vacío no lo confirma ni lo contradice: indica que las decisiones individuales aún no están consolidadas allí. No se inventan respuestas ni se asignan casos para forzar el agregado.

## 3. Resultado de las condiciones previas

La aprobación y actualización de contratos están registradas. La enmienda académica está documentada en la sección 6 de la MCR; su incorporación al Word queda pendiente, sin sobrescribir los originales.

El cierre operativo de v0.4 **no está acreditado**. Por la secuencia aprobada en 8.1–8.2, `integration_ready=false`: no se implementa aún el candidato ni se instala su entorno.

El verificador genérico de MCR puede devolver `AUTHORIZED` para MCR-2026-004 porque comprueba aprobación formal. No verifica sus requisitos operativos ni equivale a autorización para saltarse esta condición. MCR-2026-003 debe devolver `BLOCKED` por su retirada.

## 4. Acción concreta para continuar

1. Recuperar las decisiones individuales de la revisión v0.4 ya realizada y contrastarlas con las veinte filas y sus imágenes. Si ese registro no existe, realizar y documentar una revisión explícita con los criterios de v0.4, sin presentarla como transcripción histórica ni usar los criterios nuevos. No declarar completado el cegamiento sin comprobarlo.
2. Completar el CSV privado con respuestas y observaciones acreditadas. No basta escribir «8/20» ni distribuir ocho SI arbitrarios.
3. Abrir en Colab la [libreta 12 ya preparada](../notebooks/12_cierre_revision_localizacion_tibiofemoral_v04.ipynb), una vez completo el CSV, y ejecutar sus validaciones y cierre. La libreta es local; no se afirma que esté publicada o respaldada en Drive.
4. Contrastar los dos JSON generados: veinte revisadas, ocho aceptables, doce rechazadas, cero exclusiones, nueve fallos de centrado, seis de artefactos y parámetros no congelados. Si difieren, detenerse y resolver documentalmente la discrepancia, sin modificar respuestas para cuadrar resultados.
5. Tras verificar esa evidencia, continuar con la implementación acotada ya aprobada. No se necesita volver a aprobar el mismo contenido de MCR-2026-004; un cambio de método, geometría, muestra o umbrales sí requiere revisión.

No se debe repetir la libreta 11 para resolver este pendiente: volver a producir los recortes no completa la revisión.

## 5. Límites de esta entrega

Solo se actualizó gobernanza y documentación local, con pruebas de coherencia. No se modificaron el algoritmo, recortes históricos, Word ni datos privados. No se seleccionó muestra, descargaron pesos, procesaron imágenes, entrenó, crearon particiones o abrió la prueba reservada. No se hizo commit, publicación Git ni respaldo en Drive.

La Fase 1, paso 4 permanece abierta. Aprobar esta enmienda no acredita calidad del candidato ni cierra el paso.

## 6. Verificación local de esta actualización

- Las 186 pruebas del repositorio finalizaron satisfactoriamente. Son pruebas de código existente y gobernanza, no evaluación del candidato nuevo.
- El registro es válido: MCR-2026-004 devuelve `AUTHORIZED` en el control formal y MCR-2026-003 devuelve `BLOCKED`. La condición operativa de cierre pendiente permanece explícita.
- Se comparó con la instantánea previa a esta edición: MCR-2026-001 y MCR-2026-002 intactas; aprobación histórica de MCR-2026-003 conservada; candidato, validación y secciones 2–8 de MCR-2026-004 sin cambios.
- Se comprobaron sin cambios los demás campos del contrato ejecutable y las secciones de interpretabilidad, prototipo, exclusiones y gestión documental del contrato rector.
- La comprobación de diferencias no detectó errores de espacios. Se preservaron las modificaciones previas del árbol de trabajo.

## 7. Seguimiento del 1 de octubre: CSV completado y verificación local

Después de la instantánea de solo lectura de la sección 2, el investigador pidió resolver el CSV antes de iniciar la prueba nueva. No se halló una versión anterior con decisiones por rodilla. Por ello se efectuó una **nueva revisión técnica asistida por Codex** de las veinte vistas facilitadas por el investigador. Las observaciones de cada fila indican expresamente que no son transcripción de la revisión histórica y que el agregado histórico 8/20 ya era conocido. No se consultaron desenlaces individuales ni se aplicaron los criterios nuevos de MCR-2026-004 a v0.4.

El CSV privado original de Drive se completó en sus veinte filas, conservando las columnas y metadatos automáticos. El validador `knee.joint_review` se ejecutó localmente sobre el CSV y copias exactas de los JSON de parámetros y preparación de v0.4. Pasó con 8 recortes aceptables, 12 rechazados, nueve fallos de centrado, seis artefactos periféricos, tres defectos superpuestos, cero exclusiones técnicas y parámetros no congelados. Se verificó la lectura posterior del CSV en Drive y se depositaron allí el resumen de validación y el registro de ejecución, sin publicar decisiones individuales en Git.

**Límite de la evidencia:** la libreta 12 no se ejecutó en Colab. El registro de cierre identifica la ejecución local equivalente y su procedencia; no debe presentarse como salida de aquella libreta ni como recuperación de un acta original. El paso 4 permanece abierto, sin procesamiento masivo, particiones, entrenamiento ni apertura de la prueba reservada. Antes de iniciar la regresión del candidato nuevo se debe satisfacer o modificar expresamente la condición de ejecución en Colab de 8.2; el CSV pendiente ya no es el impedimento.

El investigador eligió **mantener la ejecución en Colab**. Se depositó una copia de la libreta 12 en la carpeta privada aprobada `03_notebooks`, con el mismo contenido que la fuente local y procedencia explícita en el futuro registro de Colab. Esta entrega no ejecutó esa copia; la integración y la nueva regresión siguen bloqueadas hasta recibir y verificar su salida.

Se intentó abrir la copia en Colab desde el navegador disponible, que redirigió a la pantalla de inicio de sesión de Google. No se automatizó la autenticación ni se ejecutó ninguna celda. El investigador debe iniciar sesión y ejecutar la libreta 12; después se contrastarán sus dos JSON y la salida final antes de implementar el candidato nuevo.

## 8. Seguimiento del 1 de octubre: cierre ejecutado en Colab y contrastado

El investigador ejecutó la libreta 12 en Google Colab y facilitó su salida final. Se leyeron de nuevo en la carpeta privada `v0_4_piloto` los archivos `resumen_revision_visual_publico.json` y `registro_cierre_revision.json` generados a las 14:40 UTC. Ambos coinciden en versión `tibiofemoral_crop_v0.4_pilot`, 20 rodillas revisadas, 8 recortes aceptables, 12 rechazados, 0 exclusiones técnicas, 9 fallos de centrado, 6 con artefactos periféricos y parámetros no congelados. El registro identifica expresamente `Google Colab, libreta 12`. Corrección de trazabilidad: solo el resumen conserva la huella SHA-256 del CSV revisado `09835d83c88084eed47fee850710cb2a4f199f3c8940f384763add9000f1b8ce`; la libreta 12 no incluye ese campo en el registro. La huella se contrastó después con el CSV privado real en Drive, sin alterar ninguno de los tres archivos.

Por tanto, la condición operativa 8.2 de MCR-2026-004 queda **satisfecha** y v0.4 está formalmente **rechazada**, no promovida. La revisión por rodilla fue una nueva revisión técnica asistida por Codex, con el agregado histórico conocido; no se presenta como transcripción de la revisión histórica ni como evaluación radiológica independiente. El paso 4 sigue abierto. Quedan autorizadas únicamente la implementación acotada y, tras sus pruebas, la regresión sobre el piloto histórico conforme a la MCR aprobada. No se han iniciado la muestra nueva, el procesamiento masivo, las particiones, el entrenamiento ni la prueba reservada.
