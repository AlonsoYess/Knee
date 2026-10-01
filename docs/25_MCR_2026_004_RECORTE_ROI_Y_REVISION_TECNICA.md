# MCR-2026-004 — Recorte de ROI y revisión técnica ajustados al alcance

## 1. Identificación

| Campo | Valor |
| --- | --- |
| ID | `MCR-2026-004` |
| Clase | `M2` |
| Versión | 1.1 documental; contenido técnico 1.0 aprobado sin cambios |
| Fecha | 30 de septiembre de 2026 |
| Proponente | Codex, a solicitud del investigador |
| Estado | `aprobada_no_aplicada` |
| Paso | Fase 1, paso 4: localización tibiofemoral y recorte |
| Relación | Sustituye MCR-2026-003 por aprobación expresa del 1 de octubre de 2026 |
| Registro | `configs/governance/methodology_change_log.json`, versión 1.5 |
| Autorización actual | Contenido 1.0 aprobado; implementar solo tras verificar 8.1–8.2 |

M2 designa la clase de cambio, no una nueva fase de la tesis. Se abre una solicitud distinta porque [MCR-2026-003](23_MCR_2026_003_LOCALIZACION_ANATOMICA.md) ya tiene contenido aprobado que no debe reescribirse. Su aprobación, condiciones incumplidas y [acta 24](24_APROBACION_M2_Y_COMPROBACIONES_PREVIAS.md) se conservan. El «OK SIGUIENTE PASO» autorizó preparar la versión 1.0. Posteriormente, el «OK DALE HAGAMOSLO» del 1 de octubre aprobó esta solicitud y la sustitución, en respuesta al plan explícito. Las secciones 2–8 conservan el contenido técnico sometido a decisión; sus formulaciones prospectivas no acreditan ejecución. La comprobación actual se registra en el acta 26.

## 2. Problema observado

La tesis busca comparar predicción clínica, radiográfica y multimodal de progresión estructural; no incluye como objetivo desarrollar un localizador de puntos anatómicos. La preparación de imágenes sirve al OE1 y a la comparabilidad de los modelos, no constituye una investigación clínica independiente.

La revisión de v0.4 registró **8/20 recortes aceptables**, nueve errores de centrado y seis campos con puntos de la regla, con tres defectos coincidentes, sin pérdidas anatómicas ni exclusiones técnicas. [Acta 22](22_REVISION_FASE_1_PASO_4_V04.md). Su rechazo permanece intacto. El cierre operativo mediante la libreta 12 continúa **no verificado según el acta 24**; esta propuesta no consultó de nuevo Drive ni ejecutó el cierre.

MCR-2026-003 añadió lectores anatómicos, adjudicación experta, errores en milímetros, una muestra nueva de 60 participantes y aceptación de 120/120 rodillas. El investigador ha indicado que no dispone de ese lector. Son recursos y objetivos de evaluación adicionales, no requisitos impuestos por el capítulo III para preparar recortes. La ausencia de esos recursos no justifica aceptar un recorte incorrecto; sí obliga a corregir la propuesta antes de implementarla.

## 3. Evidencia

### 3.1 Correspondencia con los documentos originales

| Fuente de autoridad | Compromiso | Consecuencia para esta propuesta |
| --- | --- | --- |
| `SRC-THESIS-001`, 1.3 y 1.5 | Preparar datos y comparar modelos de progresión; prototipo experimental | No añadir un objetivo de precisión anatómica o diagnóstico |
| `SRC-THESIS-002`, 3.2.1 | Perfiles de intensidad/gradiente, ventana proporcional, conservación articular y revisión sin desenlace | El objetivo técnico es una ROI utilizable; el Word no impone 140 mm ni anotación de 16 puntos |
| `SRC-THESIS-002`, 3.2.1 | Excluir elementos periféricos en lo posible; rechazos documentados; parámetros fijados en desarrollo | Separar defectos críticos y artefactos periféricos; no eliminar fallos del denominador |
| `SRC-THESIS-002`, 3.1.5 | Etiquetas KL de OAI, sin nueva graduación por los investigadores | No pedir al investigador que diagnostique, gradúe KL o cree etiquetas clínicas |
| `SRC-THESIS-002`, 3.1.5 y 3.2.3 | Revisión del contenido de la ficha por dos expertos y valoración posterior del prototipo por traumatólogo | Mantener esos compromisos separados; no convertirlos en lectores del recorte ni afirmar que su disponibilidad actual esté confirmada |
| MCR-2026-001 y MCR-2026-002 | Retirada de salida KL estimada y registro aprobado de modelos | Conservar ambas decisiones; no regresar silenciosamente a versiones antiguas del Word |

La exclusión absoluta de artefactos y el campo fijo de 140 mm ya figuraban en el protocolo operativo 14: no fueron introducidos por primera vez en MCR-2026-003. Cambiarlos ahora también requiere aprobación. Las nuevas reglas no se aplicarán retroactivamente para recalcular los 8/20 de v0.4.

### 3.2 Candidato inspeccionado y límites de la evidencia

**Candidato único propuesto:** adaptación trazable del recorte determinista de [Emory-HITI/knee-crop](https://github.com/Emory-HITI/knee-crop), revisión `c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93`, paquete declarado 0.1.0. El código fue leído, no instalado ni ejecutado.

| Componente consultado | Hallazgo verificable | Tratamiento previsto, sujeto a aprobación |
| --- | --- | --- |
| [LICENSE](https://github.com/Emory-HITI/knee-crop/blob/c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93/LICENSE) | MIT; atribución HITI-LAB 2025. `LICENSE.save` no contiene una licencia alternativa sustantiva | Conservar aviso y licencia en cualquier código reutilizado |
| [pyproject.toml](https://github.com/Emory-HITI/knee-crop/blob/c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93/pyproject.toml) | Paquete alfa, Python >=3.9; NumPy >=1.26,<2 y dependencias OpenCV/SciPy | Crear entorno de comprobación separado y fijar dependencias; no instalar encima del entorno actual con NumPy 2.5.3 |
| [config.py](https://github.com/Emory-HITI/knee-crop/blob/c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93/src/knee_cropping/config.py) | Parámetros explícitos de escalado, contraste, bordes y suavizado | Primera evaluación con valores de esa revisión, registrados íntegramente; sin búsqueda de hiperparámetros |
| [pipeline.py](https://github.com/Emory-HITI/knee-crop/blob/c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93/src/knee_cropping/pipeline.py) | Entrada PNG; modo unilateral; perfiles, CLAHE y derivadas; ancho variable y centro vertical candidato | Reutilizar la lógica unilateral con adaptador DICOM y geometría rastreable; no introducir otro separador bilateral |

El [artículo de Chavoshi et al., 2026](https://link.springer.com/article/10.1007/s10278-026-01961-9) enlaza ese repositorio y describe evaluación en OAI/MRKR. Su sección pública de disponibilidad de datos, sin embargo, habla de simulaciones sin participantes. Solo se consultaron secciones públicas; no el texto completo de pago. La inconsistencia no está resuelta: no se usan sus cifras para asegurar éxito, superioridad ni exactitud en esta tesis. El respaldo operativo es código inspeccionable; su desempeño local sigue desconocido.

Esta familia determinista ya estaba citada como alternativa, pero `src/knee/joint_localization.py` implementa las variantes propias v0.1–v0.4, no una reproducción de este paquete. La diferencia propuesta es evaluar una implementación externa identificada y su ROI proporcional, no afirmar que los perfiles constituyen una técnica nueva.

### 3.3 Hallazgos de la inspección de código que impiden usarlo directamente

- El lector PNG en escala de grises no conserva por sí mismo el contrato DICOM nativo. Las funciones devuelven imágenes recortadas, no una transformación completa a coordenadas originales.
- El camino bilateral puede conservar estado `success` aunque falle una rodilla; las escrituras no verifican siempre el valor devuelto por `cv2.imwrite`. No se utilizará ese indicador como aceptación técnica.
- Hay continuaciones tras errores de detección de bordes; el adaptador deberá convertirlas en rechazo explícito, no en confianza implícita.
- Se calcula una ventana Savitzky–Golay ajustada, pero se llama al filtro con la configurada. Las entradas demasiado pequeñas, perfiles vacíos o constantes y regiones sin intervalo válido deberán rechazarse antes del cálculo; no se improvisará una candidata.
- El recorte vertical se limita a los bordes de la imagen. Se registrará la caja solicitada antes de limitarla y se rechazará cualquier truncamiento, sin relleno artificial ni desplazamiento corrector.

Estos son hallazgos estáticos, no resultados de pruebas ejecutadas. Las correcciones de contrato/errores y la devolución de coordenadas se documentarán; cualquier cambio en la señal matemática de localización requerirá otra revisión de configuración antes de evaluar.

## 4. Cambio propuesto

### 4.1 Sustitución delimitada

| MCR-2026-003 / protocolo anterior | Propuesta MCR-2026-004 |
| --- | --- |
| Localizador aprendido y puntos anatómicos | Candidato determinista externo, sin pesos, entrenamiento ni ajuste fino |
| Campo físico fijo de 140 × 140 mm | ROI proporcional derivada del ancho estimado por el algoritmo; dimensiones físicas registradas, sin afirmar escala física constante |
| Centro anatómico de referencia y errores en mm | Comprobación técnica de cobertura, encuadre, lateralidad y defectos observables; sin referencia clínica de precisión |
| Dos lectores y adjudicación experta del recorte | Revisión técnica por el investigador con guía fijada y reconocimiento explícito de la limitación de un solo revisor |
| 60 participantes nuevos / 120 rodillas, todos correctos | Confirmación técnica acotada en 20 participantes nuevos / 40 rodillas previstas; umbrales operativos definidos en 4.4 |
| Ausencia absoluta de cualquier marca periférica | Rechazo de contaminación crítica; categoría separada y limitada de advertencia periférica |

### 4.2 Procedimiento que se implementaría

1. Mantener lectura DICOM, identificación y separación/lateralidad `bilateral_split_v0.2_pilot` congeladas. Alimentar una mitad y su lado verificado al candidato unilateral; no invocar el separador externo ni reflejar automáticamente la imagen.
2. Conservar el arreglo nativo. Crear una copia de trabajo normalizada por imagen con p1/p99 y conversión a 8 bits; aplicar en ella el escalado, tratamiento de bordes y señales del candidato. CLAHE se usa para localizar, no se impone a la entrada predictiva.
3. Registrar cada recorte y escala intermedios. Obtener `left_min`, `right_min` y `notch` según la lógica de la revisión fijada: ancho `right_min-left_min` y extensión vertical `notch ± ancho//2`. Retornar la caja **solicitada** y su transformación inversa, no solo una imagen.
4. Llevar la caja al DICOM original conservando los offsets de separación, recortes intermedios y escalas reales por eje. Utilizar límites semiabiertos y redondeo exterior (inicio hacia abajo, fin hacia arriba); rechazar límites inválidos o fuera de la mitad original. Registrar espaciado y dimensiones físicas, incluido espaciado anisótropo. El centro de caja es un indicador geométrico, no una medición del espacio articular.
5. Extraer los píxeles del arreglo nativo sin dibujar marcas, enmascarar, rotar, rellenar ni deformar el recorte. El redimensionamiento posterior se rige por el registro de modelos vigente y será idéntico entre escenarios comparables; esta MCR no modifica resoluciones predictivas.
6. Emitir caja, procedencia, parámetros y estado por rodilla: `candidate` o `abstain`. Una candidata no es un recorte aprobado. Las comprobaciones automáticas de dimensiones, integridad, coordenadas y señal no se presentarán como una garantía de conservación anatómica.

La adaptación no se denominará reproducción exacta del resultado publicado: cambian la entrada, la conservación de píxeles, la instrumentación y la política de errores. Se distinguirá el núcleo reutilizado del adaptador propio mediante atribución, revisión, pruebas y diferencias documentadas.

### 4.3 Guía de revisión técnica propuesta

Revisar cada candidata frente a su imagen unilateral original, sin desenlace, variables clínicas, predicciones, versión nominal ni puntuación de confianza. El revisor registrará decisiones y motivos antes de leer el resumen automático. El cegamiento a la versión tiene alcance limitado: el desarrollador puede reconocer la apariencia del método. No se denominará revisión independiente ni validación radiológica experta.

| Criterio | Aceptable técnicamente | Rechazo o duda |
| --- | --- | --- |
| Identidad/lado | Coincide con la mitad y lateralidad verificadas | Mezcla, inversión no explicada o lado irresoluble |
| Cobertura | Se reconocen completos ambos compartimentos articulares, contornos distales de ambos cóndilos y platillos tibiales con hueso adyacente, contrastando el original | Contorno articular cortado, compartimento ausente o imposibilidad de decidir |
| Encuadre | La región de ambos espacios articulares queda dentro de la mitad central vertical del recorte, entre 25 % y 75 % de su altura, con margen visible alrededor de los contornos laterales | Región pegada al borde, fuera de esa banda o truncada. No se exige que una línea recta siga una interlínea curva |
| Contaminación crítica | Sin texto legible, rectángulos de anonimización ni dispositivos superpuestos al hueso o región articular; sin relleno artificial | Cualquiera de esos defectos, aunque la caja parezca centrada |
| Advertencia periférica | Solo puntos aislados de regla o borde del dispositivo, fuera de los contornos óseos y confinados al cuarto lateral externo de la ROI | Regla atravesando hueso/espacio articular, elemento fuera de esa franja o naturaleza dudosa |
| Visualización | Detalle suficiente para decidir al cotejar el original con ventana de visualización ajustable | Saturación o contraste que impiden decidir incluso tras revisar la visualización |

La banda central es una regla operativa de encuadre propuesta, no una tolerancia clínica validada. No se dibujarán puntos de verdad anatómica ni se calcularán IoU, error milimétrico o concordancia entre expertos sin referencias apropiadas. Ajustar la ventana del visor no cambia píxeles ni coordenadas del recorte.

Decisiones posibles: `aceptable`, `aceptable_con_advertencia_periferica`, `rechazado`, `no_evaluable`. La última no equivale a normalidad ni a aprobación. No se fuerza al investigador a resolver anatomía dudosa: ese caso cuenta como no satisfactorio. La valoración de Codex puede ayudar a organizar observaciones, pero no sustituye una referencia experta ni se contabiliza como segundo lector.

Permitir puntos periféricos no demuestra ausencia de aprendizaje espurio. Se informará su frecuencia y permanecerán como limitación; la auditoría de interpretabilidad posterior ya prevista no garantiza eliminarlos. Texto legible y anonimización siguen siendo defectos críticos en cualquier posición.

### 4.4 Muestras, métricas y decisión previa a resultados

**Regresión exploratoria:** las mismas 10 adquisiciones / 20 rodillas. Se revisan todas; no se reinterpretan las actas antiguas. Permiten comprobar integración y defectos conocidos, no demostrar desempeño poblacional. Una única configuración inicial con valores upstream, sin barrido de parámetros. Si falla, documentar el fallo y mantener el paso abierto; no iniciar automáticamente otra iteración.

**Confirmación técnica nueva:** 20 participantes distintos del piloto, 40 rodillas previstas, solo V00. Selección aleatoria sin reemplazo, semilla 2026, sobre lista privada ordenada de participantes con adquisición basal disponible, sin consultar V06, progresión, KL ni calidad del recorte. Si existen duplicados, resolver previamente con las reglas del inventario; una adquisición por participante. No se selecciona ningún sujeto hasta la aprobación y la superación de la regresión. La selección y sus identificadores serán privados. Todos los sujetos utilizados quedan reservados para desarrollo, también los que fallen. Si ya existiese partición congelada, solo son elegibles sujetos de desarrollo. Si no hay suficientes, detenerse; no reducir el tamaño ni reemplazar fallos.

El tamaño 20 es una decisión operativa para observar 40 salidas en una muestra distinta con una carga de revisión acotada. **No deriva de potencia estadística ni permite certificar una tasa poblacional del 95 %.** Deben informarse condiciones técnicas observadas y no cubiertas (modalidad, tamaño, espaciado); no se promete representatividad de subgrupos. Si se ajusta el método a esos resultados, la muestra pierde su papel de confirmación no usada para ajuste y cualquier repetición exige nueva propuesta.

Definir por conjunto: `N` rodillas previstas, `A` candidatas emitidas automáticamente, `Q` candidatas técnicamente aceptables (con o sin advertencia), `W` aceptables con advertencia y `F=A-Q` candidatas rechazadas o no evaluables. Las abstenciones, fallos de lectura y salidas faltantes permanecen en `N`. Reportar `A/N`, `Q/N`, `Q/A` (no definida cuando A=0), `W/N`, `F`, motivos y participantes con ambas rodillas aceptables. No tratar las dos rodillas como observaciones estadísticamente independientes.

| Puerta operativa propuesta | Piloto histórico | Confirmación nueva |
| --- | --- | --- |
| Cobertura útil `Q/N` | >= 95 %: al menos 19/20 | >= 95 %: al menos 38/40 |
| Candidatas incorrectas o no evaluables `F` | 0 | 0 |
| Advertencias periféricas `W/N` | <= 10 %: máximo 2/20 | <= 10 %: máximo 4/40 |
| Trazabilidad de todas las entradas y revisión de todas las candidatas | Completa | Completa |
| Pruebas de integridad, coordenadas y reproducibilidad | Todas satisfactorias | Misma versión congelada |

El 95 % fija un presupuesto operativo máximo de salidas no utilizables del 5 %; el 10 % limita la contaminación periférica tolerada. Ambos son **umbrales propuestos por este protocolo**, no estándares publicados, cifras clínicas ni resultados observados. Se justifican como límites explícitos de pérdida y contaminación para decidir un ensayo acotado; no eliminan sesgo de selección. No se bajarán después de ver el resultado. `F=0` significa cero errores detectados entre las candidatas del conjunto revisado, no cero errores garantizados en toda OAI.

Una candidata errónea detectada por el revisor hace fallar la puerta: no se convierte retrospectivamente en abstención del algoritmo para alcanzar el umbral. El rechazo automático sí puede abstenerse previamente, dentro del presupuesto. Un sistema que rechace todo no aprueba. Los fallos del algoritmo no se reclasifican como exclusiones de la cohorte.

La promoción exige todas las puertas en ambos conjuntos y un acta del investigador. Se mantiene control de calidad posterior en el procesamiento general; aprobar aquí no prueba precisión clínica ni autoriza por sí solo el procesamiento masivo, la creación de particiones o el entrenamiento. Las exclusiones posteriores y la cohorte efectiva común deberán seguir el capítulo III y conservar el denominador y los motivos.

## 5. Alternativas evaluadas

| Alternativa | Ventaja | Límite / decisión |
| --- | --- | --- |
| Mantener MCR-2026-003 | Referencias anatómicas precisas si existen lectores y pesos adecuados | No viable con recursos confirmados; condiciones aún incumplidas. Se propone sustituirla, no declarar esas condiciones satisfechas |
| Continuar afinando v0.4 | Código conocido | Cuatro versiones y reutilización del mismo piloto; no seleccionada |
| Adaptar knee-crop en una revisión fija | Código accesible, sin pesos, ROI proporcional y señales explícitas | Seleccionada para evaluación, no declarada superior; exige adaptación y pruebas; comparte riesgos de intensidad y artefactos |
| Detector aprendido o entrenamiento propio | Podría localizar cajas o puntos | Pesos, datos, procedencia y/o anotaciones adicionales sin resolver; no seleccionado |
| Recortes manuales de producción | Control visual directo | No reproduce inferencia automática del prototipo; no seleccionado |

## 6. Impacto académico y metodológico

| Dimensión | Impacto de aprobar esta solicitud |
| --- | --- |
| Capítulos I / objetivos / hipótesis | Sin cambios. Se conserva el objetivo multimodal longitudinal y sus comparaciones |
| Capítulo II | Referenciar el método realmente utilizado y sus límites, sin atribuir resultados propios al artículo |
| Capítulo III, 3.2.1 | Precisar ROI proporcional, revisión técnica no experta, muestra nueva, umbrales y política de rechazo |
| Cohorte / predictores / desenlace | Sin cambiar criterios de elegibilidad, edad/sexo/IMC, V00/V06, KL basal o progresión a 48 meses; registrar pérdidas por QC y conservar comparabilidad |
| Partición / inferencia | Reserva técnica en desarrollo sin crear bloque 0 ni pliegues; AP, calibración, contrastes, bootstrap y Holm intactos |
| Modelos | MCR-2026-002 y `configs/model_registry.json` intactos; no se autoriza entrenamiento |
| Prototipo / interpretabilidad | Mismo procesamiento automático y rechazo; sin nuevas salidas clínicas. Reconocer riesgo residual de artefactos y limitación del revisor |
| Compromisos profesionales | Valoración final y revisión de ficha permanecen; no son requisitos de anotación anatómica para este paso |

### Enmienda propuesta de 3.2.1 — no incorporada al Word

> La región tibiofemoral se delimitará mediante una adaptación versionada de un procedimiento determinista de intensidad, contraste local y derivadas, aplicado a cada mitad con lateralidad previamente verificada. La ventana será proporcional al ancho estimado por el procedimiento, sin imponer un campo físico constante. La localización utilizará una copia de trabajo; el recorte conservará los píxeles originales y una transformación verificable de coordenadas. Se aplicará el mismo procedimiento en los modelos radiográfico y multimodal y en el prototipo, sin correcciones manuales privilegiadas de producción.
>
> La evaluación será técnica, no una validación radiológica de puntos anatómicos. El investigador revisará cobertura, encuadre y contaminación mediante una guía fijada, sin desenlace ni predicciones. Los casos dudosos no se declararán correctos. Se distinguirán errores críticos, abstenciones y advertencias periféricas, informando denominadores completos y limitaciones de un solo revisor. Tras la regresión sobre el piloto histórico, una configuración congelada se comprobará en veinte participantes adicionales seleccionados sin consultar el desenlace; todos permanecerán en desarrollo. La promoción seguirá los umbrales preespecificados de MCR-2026-004 y no sustituirá el control de calidad posterior ni la evaluación predictiva. La prueba reservada permanecerá cerrada.

## 7. Riesgos y recursos

- **Fracaso del candidato:** código publicado no garantiza recortes correctos ni evita que se repitan fallos de perfiles. Si falla, se informa; no se promueve ni se adapta silenciosamente para aprobar.
- **Subjetividad:** revisión única sin referencia experta. Limita lo que puede afirmarse, aunque exista guía y cegamiento al desenlace. No se reportará precisión anatómica ni fiabilidad entre lectores.
- **Escala:** la ROI proporcional cambia el campo físico respecto de 140 mm. Registrar dimensiones y revisar conservación de contexto. No declarar equivalencia con recortes físicos de otros estudios.
- **Artefactos:** las marcas periféricas toleradas pueden seguir siendo señales espurias. Documentar advertencias y su límite; no prometer que el localizador las elimina.
- **Selección:** toda revisión local para ajuste queda en desarrollo. No elegir recortador por AP, KL o resultados de prueba. La ausencia de pesos elimina la auditoría de checkpoint para este candidato, pero no permite afirmar que los autores nunca utilizaron imágenes OAI al diseñar sus reglas.
- **Compatibilidad:** el entorno local actual usa NumPy 2.5.3; upstream exige <2. Comprobar en entorno separado, fijar versiones y documentar traslado a Colab sin alterar el entorno de trabajo existente. CPU prevista; tiempo y memoria no medidos.
- **Recursos humanos:** investigador para 60 revisiones iniciales previstas (20 históricas + 40 nuevas), sin exigir otro lector anatómico. No afirmar que ya revisó ni que pueda resolver todos los casos.
- **Privacidad:** ninguna radiografía, identificación, coordenada individual o manifiesto se envía a GitHub/servicios públicos. No se necesitan pesos, cuentas nuevas ni envío de imágenes al proveedor.

## 8. Plan de implementación y verificación

1. **Decisión y coherencia:** aprobar esta solicitud exacta y resolver expresamente la sustitución de MCR-2026-003, conservando su historial. Actualizar contrato rector, reglas, alcance ejecutable, protocolo 14 y plan antes de implementar. Gestionar incorporación académica sin sobrescribir los Word originales ni inferir autorización para editarlos.
2. **Cierre v0.4:** ejecutar y contrastar la libreta 12 en privado; 20 revisiones, 8 aceptables, 12 rechazadas y parámetros no congelados. Si el CSV no coincide, resolver la discrepancia sin forzar el resumen.
3. **Implementación acotada:** núcleo adaptado y atribuido en `src/knee`, configuración fijada, esquema de revisión separado de las actas antiguas, notebook como orquestador, entorno de prueba aislado. Sin modificar ni sustituir v0.1–v0.4.
4. **Pruebas:** equivalencia del núcleo con upstream sobre matrices sintéticas válidas; ambas lateralidades sin inversión; ida/vuelta de coordenadas y offsets; espaciado anisótropo; conservación de píxeles nativos; imágenes uniformes, pequeñas, intervalos vacíos y bordes; truncamiento; errores por rodilla y escritura fallida; repetibilidad exacta de cajas en el entorno fijado. Las pruebas sintéticas no prueban éxito anatómico.
5. **Regresión:** ejecutar solo sobre el piloto histórico, registrar resultados y revisión de todas las candidatas. Detenerse si no supera 4.4.
6. **Congelamiento y confirmación:** fijar código, dependencias, configuración, guía y selección antes de generar resultados en los veinte participantes nuevos. Sin reconfigurar tras observarlos.
7. **Acta:** informar métricas completas, condiciones no cubiertas, advertencias y limitaciones; aprobar o rechazar el paso 4. El procesamiento general requiere su propio paso posterior, no queda habilitado por preparar esta solicitud.

Artefactos públicos: esta propuesta, registro, atribución/licencia futura, código y pruebas futuros e informes agregados. Artefactos privados futuros: selección, reserva de desarrollo, coordenadas, revisiones, imágenes y registros de ejecución. La propuesta actual no ha creado estos últimos ni se ha respaldado en Drive.

**Reversión:** conservar resultados, declarar candidato rechazado y mantener el paso 4 abierto. No reactivar v0.4 como aceptada, aumentar el presupuesto de fallos, cambiar la muestra, añadir máscaras o sustituir algoritmo sin revisión previa.

## 9. Decisión del investigador

| Campo | Valor |
| --- | --- |
| Decisión | `aprobada`; solicitud `aprobada_no_aplicada` |
| Fecha de aprobación | 1 de octubre de 2026 |
| Evidencia | «OK DALE HAGAMOSLO», en respuesta al plan explícito de aprobar esta MCR, sustituir la anterior y verificar cierre de v0.4 antes de implementar |
| Alcance aprobado | Sustituir MCR-2026-003 por candidato determinista fijado, ROI proporcional y evaluación técnica de 4.1–4.4; implementar y evaluar únicamente después de cumplir 8.1–8.2 |
| No incluido | Entrenamiento, ajuste fino, procesamiento masivo, particiones definitivas, apertura de prueba, publicación Git/Drive o edición de Word |
| Situación de la solicitud anterior | Retirada sin aplicación; aprobación y contenido técnico históricos conservados |
| Requisito pendiente observado | CSV de v0.4 sin respuestas y registros de cierre ausentes; no iniciar integración |

## 10. Cierre posterior

### Instantánea histórica: preparación del 30 de septiembre de 2026

Esta entrega prepara la revisión; **no implementa el cambio metodológico**. No se integró el candidato, instalaron dependencias, procesaron radiografías, seleccionaron participantes ni modificaron Word, contratos aprobados, modelos o recortes históricos. No se descargaron pesos, entrenó, creó particiones ni abrió la prueba reservada. Los controles documentales locales se reportan aparte: no acreditan rendimiento del localizador.

El cierre de implementación y del paso 4 quedan pendientes de aprobación y evidencia futura. La aprobación histórica de MCR-2026-003 no autoriza MCR-2026-004.

### Comprobación documental de esta entrega

- Las 186 pruebas locales del repositorio finalizaron satisfactoriamente, incluidas siete nuevas comprobaciones documentales de esta propuesta; no incluyen ejecución del candidato externo.
- El registro resulta válido y la consulta específica de MCR-2026-004 devuelve `BLOCKED`, como corresponde a una propuesta pendiente.
- La comparación de los tres registros MCR anteriores con la instantánea anterior a esta edición confirma que se conservaron sin cambios.
- La revisión de diferencias no detectó errores de espacios; las modificaciones previas del árbol de trabajo se preservaron. No se realizó commit, publicación remota ni respaldo privado.

### Actualización del 1 de octubre de 2026

Aprobación y sustitución registradas; contratos y documentos rectores actualizados. El contenido técnico 1.0 permanece sin cambios. La implementación continúa sin aplicar: la lectura actual de Drive confirmó veinte filas sin respuestas en el CSV de v0.4 y ausencia de ambos registros de cierre. El [acta 26](26_APROBACION_MCR_2026_004_Y_CIERRE_PENDIENTE.md) separa estos hechos de la decisión 8/20 ya documentada. No se ejecutó el candidato ni se habilitaron etapas posteriores.

### Seguimiento posterior del 1 de octubre: condición previa satisfecha y código preparado

El investigador ejecutó la libreta 12 en Colab. Se contrastaron en Drive los dos JSON de cierre y la huella del CSV: v0.4 quedó rechazada con 8/20 aceptables, 12/20 rechazadas, nueve fallos de centrado y seis artefactos; parámetros no congelados. La procedencia de la nueva revisión asistida se conserva en el [acta 26](26_APROBACION_MCR_2026_004_Y_CIERRE_PENDIENTE.md). Esto satisface 8.2, sin cerrar el paso 4.

Se prepararon localmente el núcleo fijado de Emory-HITI con atribución MIT, el adaptador DICOM de ROI proporcional, el corredor de regresión limitada a las diez adquisiciones históricas, las dependencias aisladas y la libreta 13. Once pruebas propias pasaron sobre entradas sintéticas o controladas, incluidas equivalencia de salida del núcleo instrumentado, transformación de coordenadas, conservación de píxeles, ambas lateralidades, abstenciones, rechazo de cambios de parámetros y fallo de escritura. Estas pruebas **no** acreditan calidad de los recortes en OAI. No se aplicó el candidato a radiografías, seleccionó la muestra nueva ni publicó código o libreta. MCR-2026-004 continúa `aprobada_no_aplicada` hasta que se ejecute y documente la evaluación; los umbrales, la geometría y el contenido técnico 1.0 no cambian.
