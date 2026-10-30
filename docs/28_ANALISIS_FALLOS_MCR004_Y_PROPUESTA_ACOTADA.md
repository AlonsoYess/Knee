# Análisis del rechazo MCR-2026-004 y propuesta acotada

| Campo | Valor |
| --- | --- |
| Fecha | 1 de octubre de 2026, hora de Perú |
| Estado | Análisis terminado; propuesta técnica pendiente de decisión, sin implementación |
| Ubicación | Fase 1, paso 4: localización tibiofemoral y recorte |
| Autorización de este trabajo | Analizar causas, contrastar alternativas y presentar una propuesta dentro del alcance aprobado |
| Código inspeccionado | `9bca503441a4cbeaedd7d63afa95ce6101ef2aa3` |
| Evidencia de ejecución | Cierre de la libreta 14 en Colab, 2026-10-01 23:42:47 UTC, contrastado con los registros privados en Drive |
| Resultado vigente | `rejected_after_technical_review`; paso 4 abierto |
| Publicación | Documento local para revisión; no publicado ni sincronizado con Drive en esta entrega |

## 1. Conclusión y recomendación

El problema observado actualmente se concentra en la contaminación incluida en la caja y en la incertidumbre de dos evaluaciones. El CSV cerrado registra lateralidad, cobertura, encuadre y visualización suficientes en las veinte candidatas. Esto respalda investigar el control final de la ROI antes de sustituir el localizador; no constituye una validación anatómica independiente ni demuestra que el método funcionará en toda la cohorte.

Se recomienda conservar el núcleo MCR004 como referencia y preparar una **auditoría diagnóstica de factibilidad del recorte y de sus controles finales**, delimitada en la sección 6. Su pregunta es concreta: ¿existe una caja que conserve la anatomía exigida y excluya el elemento problemático, y puede obtenerse con una regla automática general? La máscara y las tres advertencias periféricas parecen susceptibles de ese análisis geométrico. Las dos marcas de naturaleza incierta son el límite decisivo: están próximas o superpuestas a regiones que deben conservarse y su origen no quedó resuelto por la ventana ajustable.

No hay evidencia suficiente para prescribir ahora un porcentaje de reducción, seleccionar una red sustituta o prometer 19/20. La propuesta de corrección que se estudiaría es un **refinamiento geométrico con comprobación final de contaminación y abstención prospectiva**, condicionado a demostrar su factibilidad. Esta entrega no define ese componente como listo para implementar.

## 2. Alcance que gobierna el análisis

Se consultaron el contrato rector, las reglas invariables, el contrato ejecutable, el registro de modelos y las MCR previas. La contribución principal continúa siendo comparar predicción clínica, radiográfica y multimodal de progresión estructural a 48 meses en OAI. Se conservan V00 como fuente de predictores, unidad rodilla, agrupación por participante, población basal KL 2/3, edad/sexo/IMC, desenlace, cohorte común, evaluación y prototipo académico.

El recortador sirve a la preparación radiográfica del OE1. No se añade una investigación de precisión de puntos anatómicos, un estimador KL, diagnóstico de las marcas, anotación clínica experta o entrenamiento de un detector propio. Se conserva también la ampliación de candidatos predictivos ya aprobada por MCR-2026-002: este análisis no vuelve al listado antiguo de solo dos arquitecturas ni agrega otras.

Una sustitución del recortador puede seguir siendo preprocesamiento dentro del objetivo general, pero requiere su propia justificación y control de cambios. No corresponde reactivar MCR-2026-003, retirada, ni imponer sus lectores y recursos como condición nueva.

## 3. Qué acredita el cierre y qué no

Se cotejaron el resumen y el registro de cierre de Drive, el CSV cerrado y seis vistas correspondientes a las tres incidencias no satisfactorias y las tres aceptadas con advertencia. En las dos dudas se usaron las vistas de ventana ajustada conservadas. No se generaron nuevas decisiones ni se modificó el CSV. Las notas históricas de algunas filas conservan la frase de propuesta pendiente; la confirmación efectiva está en el registro de cierre y en sus respuestas asociadas.

| Indicador | Resultado | Criterio vigente |
| --- | ---: | ---: |
| Rodillas previstas y revisadas | 20 | 20, denominador completo |
| Candidatas emitidas | 20 | Reportar todas |
| Aceptables sin advertencia | 14 | Subgrupo de Q |
| Aceptables con advertencia, W | 3 | Máximo 2/20 |
| Aceptables totales, Q | 17/20, 85 % | Mínimo 19/20 |
| Rechazadas | 1 | Cuenta en F |
| No evaluables | 2 | Cuentan en F |
| Candidatas incorrectas/no evaluables, F | 3 | 0 |
| Abstenciones del algoritmo | 0 | Permanecen en el denominador |
| Integridad nativa comprobada en Colab | 20 | Satisfactoria en todas las candidatas |

Las veinte filas tienen `SI` en lateralidad, cobertura, encuadre y visualización. La integridad acredita que los recortes guardados corresponden a sus píxeles y coordenadas de origen; no acredita ausencia de artefactos. La revisión fue técnica asistida por Codex, confirmada por el investigador, con versión conocida y sin desenlaces; no fue un segundo lector ni una validación clínica.

**Corrección de interpretación:** 8/20 de v0.4 y 17/20 de MCR004 son resultados bajo protocolos distintos. Cambiaron el campo de recorte, la regla de encuadre y la tolerancia de advertencias. Se puede informar la secuencia histórica, pero no atribuir una mejora causal de 45 puntos porcentuales al algoritmo. Tampoco debe calcularse una tasa poblacional fiable con veinte rodillas de diez participantes reutilizados en desarrollo.

## 4. Causas confirmadas e incertidumbres

### 4.1 Máscara negra en una candidata rechazada

La vista conservada muestra parte de un rectángulo negro en una esquina inferior de la ROI; el original también lo muestra. El criterio aprobado considera la anonimización crítica en cualquier posición. La cobertura y el encuadre fueron registrados como correctos.

En el núcleo sí existe detección de cajas negras. `get_black_box_rows` busca contornos oscuros de geometría aproximadamente rectangular. Sin embargo, sus filas solo ajustan el intervalo de búsqueda vertical del punto `notch`; no se comprueba la intersección entre la máscara y la caja final. Además, la extensión vertical final depende de la mitad del ancho y puede alcanzar una máscara aun cuando el centro se haya buscado fuera de ella. Véase [pipeline.py](../src/knee/third_party/emory_hiti/pipeline.py), funciones `get_black_box_rows` y `process_knee_side`.

Esto confirma una insuficiencia del control final. No demuestra cuál de las dos rutas ocurrió en esa imagen: máscara no detectada, o máscara detectada pero incluida al extender la ROI. La traza guardada no registra esa distinción. La posición periférica sugiere una posible corrección geométrica; todavía falta demostrar que una regla automática la resolvería preservando márgenes suficientes.

### 4.2 Tres advertencias periféricas

Las vistas muestran puntos de regla o borde de dispositivo en la periferia, fuera de los contornos óseos según la revisión cerrada. El algoritmo obtiene límites horizontales a partir de sumas de intensidad por columna y mínimos alrededor de un pico; no delimita una envolvente del hueso con un margen de seguridad. El ancho también determina la altura de la caja.

La búsqueda de líneas exteriores se restringe a bandas de los bordes y no equivale a detectar cualquier regla o punto aislado. Por ello una caja válida puede conservar esos elementos. Es una explicación estructural de por qué el método los permite; que los puntos hayan desplazado los mínimos en cada caso no está demostrado. Véanse `get_mins`, `preprocess_and_crop` y `process_knee_side` en el mismo archivo.

Reducir una franja externa podría ayudar en esas vistas, pero recortar un porcentaje fijo a todas las imágenes podría truncar otras rodillas. La investigación debe preservar la cobertura en las veinte entradas, no mejorar solo las imágenes problemáticas.

### 4.3 Dos marcas de naturaleza incierta

Una vista conserva un trazo oblicuo fino que alcanza la región tibial; otra conserva marcas lineales y un punto brillante superpuesto a la región tibial. Ambas imágenes son visibles y la incertidumbre se refiere a contaminación. No se confirma que los trazos sean dispositivos, estructuras anatómicas, lesiones o errores digitales.

La ventana ajustable ya se utilizó. Pedir al investigador que vuelva a moverla sin una pregunta técnica nueva no resuelve esta limitación. Las funciones de vista usan una transformación lineal y redimensionamiento para mostrar el original y la ROI; no clasifican las marcas. Las vistas reducidas permiten localizar la duda, pero no determinar su origen físico.

Si un elemento problemático forma parte de una región que obligatoriamente debe conservarse, ninguna caja rectangular puede excluir ese mismo elemento y conservar íntegra esa región. Cambiar a un localizador aprendido no elimina esa incompatibilidad. Borrar, rellenar o enmascarar píxeles alteraría el contrato vigente y no es la propuesta.

### 4.4 Significado del resultado `candidate`

El [adaptador](../src/knee/roi_mcr004.py) comprueba entrada, parámetros fijados, geometría, truncamiento, equivalencia con el núcleo e integridad de píxeles. La abstención actual cubre fallos de esos contratos; no implementa un control general de artefactos ni determina que una marca desconocida sea segura. Las pruebas sintéticas verifican esas propiedades y no sustituyen la revisión de imágenes. El resultado de Colab es consistente con esa separación de funciones.

## 5. Alternativas contrastadas con fuentes primarias

| Alternativa | Utilidad posible | Límite y decisión para este paso |
| --- | --- | --- |
| Núcleo actual con refinamiento y control final | Atiende la inclusión de periferia innecesaria y la falta de veto final | Primera línea a estudiar; condicionada a factibilidad y una regla general, todavía sin rendimiento demostrado |
| KneeLocalizer, propuestas anatómicas más HOG/SVM | Proporciona otra forma de localizar la articulación | No es un detector de contaminación; requiere auditar modelo, geometría e integración. No seleccionado ahora |
| KNEEL preentrenado | Puede proporcionar referencias anatómicas para definir márgenes | Requiere acceso, licencia y procedencia de pesos, además de revisar solapamiento con OAI. No resuelve por sí solo una marca sobre la anatomía |
| Entrenar un detector propio | Permitiría diseñar una tarea de detección específica | Agrega anotaciones y validación propias sin justificación proporcional al fallo observado; no propuesto |
| Recortes manuales de producción | Resolvería algunos límites individualmente | No satisface la inferencia automática reproducible comprometida |
| Máscaras, relleno o exclusión retroactiva de fallos | Podrían cambiar la apariencia o los recuentos | Alteran datos o criterios y no resuelven la evaluación vigente; no propuestos |

[Tiulpin et al. (2017)](https://arxiv.org/abs/1701.08991) describen propuestas anatómicas evaluadas con HOG y SVM; el [repositorio oficial KneeLocalizer](https://github.com/imedslab/KneeLocalizer) ofrece cajas de 120 mm. Son antecedentes de localización, no una garantía de aceptación bajo nuestra guía ni una razón para imponer ese tamaño.

[KNEEL](https://github.com/imedslab/KNEEL) es una implementación de localización anatómica. Su [ficha oficial de pesos](https://huggingface.co/imeds/kneel) declara modelos actualizados y licencia CC-BY-NC-4.0. Si se retomase esta alternativa, habría que verificar el checkpoint exacto, sus datos y su compatibilidad; no se descargaron pesos ni se adoptó el modelo.

El [artículo asociado a knee-crop](https://link.springer.com/article/10.1007/s10278-026-01961-9) presenta un método determinista sin anotaciones ni entrenamiento. La página pública continúa mostrando una discrepancia entre el resumen, que menciona OAI/MRKR, y la disponibilidad de datos, que habla de simulaciones. Solo se accedió al contenido público: sus cifras no se usan para prometer desempeño local. El fundamento técnico inspeccionable es la revisión de código fijada y nuestra evaluación.

Un [estudio primario con radiografías OAI sobre atajos predictivos](https://www.nature.com/articles/s41598-024-79838-6) muestra que señales de centro y otros factores pueden estar distribuidas por la imagen. Por tanto, retirar marcas visibles no demuestra que se haya eliminado todo aprendizaje espurio. Esto justifica un control prudente y mantener las limitaciones; no añade a la tesis un experimento nuevo sobre sesgos.

## 6. Propuesta concreta del próximo trabajo

### 6.1 Auditoría diagnóstica delimitada

Preparar una libreta de diagnóstico, con lógica compartida en `src/knee`, para que el investigador la ejecute en Colab sobre las mismas diez adquisiciones. El producto será un informe de factibilidad; no generará recortes de producción nuevos ni cambiará el cierre.

1. Verificar el cierre rechazado, las huellas y las veinte correspondencias originales. Utilizar únicamente los datos basales del piloto ya autorizado.
2. Reproducir el candidato con los parámetros fijados y exigir igualdad exacta de cajas y píxeles con la ejecución cerrada. Si difiere, detener el diagnóstico y explicar la discrepancia.
3. Instrumentar, sin cambiar el cálculo, límites horizontales, banda vertical antes/después, filas de máscara detectadas, `notch` y caja final. Esto distinguirá las rutas de la máscara que la evidencia actual no permite separar.
4. Mostrar las regiones cuestionadas sin reducción espacial, junto con su contexto, y reutilizar las ventanas conservadas. Comprobar si una diferencia visible proviene de la representación. La presencia en píxeles originales no identifica por sí sola la naturaleza de una marca: esa limitación deberá quedar escrita.
5. Registrar para las seis incidencias la relación entre elemento observado, caja y anatomía exigida: potencialmente separable por un borde, inseparable de la región necesaria, o indeterminada. Cualquier marcación manual se usará exclusivamente como evidencia de diagnóstico, nunca como corrección privilegiada de producción ni verdad anatómica experta.
6. Entregar un mapa de causas y una conclusión de factibilidad. No seleccionar umbrales por el número de aprobaciones ni crear una familia de configuraciones para buscar la que pase.

Se prevé que CPU y dependencias ya fijadas sean suficientes para esta instrumentación; el tiempo real se medirá. No requiere pesos, imágenes externas ni un nuevo lector. Codex preparará la evidencia y explicará las dudas; no se pedirá al estudiante diagnosticar trazos. Si el origen sigue sin poder determinarse, se registrará ese límite.

### 6.2 Regla de decisión tras el diagnóstico

| Hallazgo | Consecuencia |
| --- | --- |
| Defecto separable sin perder anatomía y señal automática verificable | Redactar una MCR delimitada para refinamiento de caja y control final; fijar reglas y pruebas antes de ejecutarla |
| Una salida no recuperable pero detectable automáticamente | Puede estudiarse abstención prospectiva; sigue contando en N y consume el único fallo de cobertura permitido en veinte |
| Dos o más salidas no recuperables bajo las reglas actuales | No prometer aprobar el piloto con refinamiento; mantener rechazo y presentar el límite para decisión metodológica, sin cambiar muestra ni denominador |
| La localización o el límite anatómico resulta realmente insuficiente | Reconsiderar un localizador externo como preprocesamiento, con auditoría de licencia, pesos y procedencia y una nueva propuesta |
| Duda que exige información no disponible | Mantener `no_evaluable`; una nueva opinión de Codex no equivale a evidencia independiente |

No se abre todavía una solicitud MCR completa para un algoritmo cuya regla final no está especificada. Si se demuestra factibilidad, la corrección propuesta afectaría geometría y política de aceptación automática de MCR004 y se trataría como M2, conservando las puertas 19/20, F=0 y W<=2. El diagnóstico de comportamiento equivalente, por sí mismo, no modifica el método aprobado; la implementación de una corrección requerirá la decisión prevista en el [control de cambios](03_CONTROL_CAMBIOS_METODOLOGICOS.md).

### 6.3 Condición numérica que impide prometer éxito

Con denominador veinte, convertir prospectivamente las tres actuales incidencias en abstenciones dejaría Q=17/20 y W=3/20. Aunque F pasase a cero, las otras puertas seguirían fallando. Corregir únicamente la máscara mientras persisten ambas dudas dejaría como máximo Q=18/20.

Para superar la puerta con un futuro candidato habría que obtener al menos dos salidas adicionales genuinamente aceptables, ninguna candidata incorrecta/no evaluable y reducir al menos una de las tres advertencias actuales. Si la evidencia no permite eso, ninguna mejora de presentación ni cambio de nombre resuelve la restricción. El cierre actual permanece inalterado.

## 7. Impacto, riesgos y límites

La recomendación mantiene objetivos, hipótesis, predictores, desenlace, modelos aprobados y evaluación. Un refinamiento de ROI sí afectaría la sección de preprocesamiento del capítulo III y deberá documentarse antes de aplicarlo, con el mismo procedimiento en los escenarios radiográfico/multimodal y en inferencia. No se modifican los Word originales en esta entrega.

El piloto histórico es desarrollo reutilizado: incluso un futuro resultado 20/20 sería una regresión exploratoria, no confirmación independiente. La muestra nueva de veinte participantes permanece sin seleccionar hasta superar las puertas y fijar el candidato. Las exclusiones de calidad futuras deben ser trazables y mantener la cohorte común; no se eliminan ahora rodillas para elevar la fracción de aceptación.

La revisión cerrada de veinte rodillas, la reinspección de seis incidencias, las pruebas sintéticas y la revisión de código no permiten garantizar calidad en toda OAI. La ausencia de referencia clínica limita las afirmaciones; tampoco se conoce si cada marca sería aprovechada por los modelos predictivos. Esta entrega no entrenó modelos, consultó desenlaces, abrió prueba ni procesó nuevas radiografías.

## 8. Deuda documental identificada y entrega realizada

Algunos textos y estados ejecutables aún describen MCR004 como no aplicada o su cierre como pendiente. El resultado actual contrastado es ejecución realizada y candidato rechazado. Se debe actualizar ese seguimiento conservando la historia, sin repetir el cierre. También existen resúmenes antiguos del listado de modelos; las ampliaciones aprobadas de MCR002 permanecen vigentes.

Esta entrega añade únicamente este análisis y un anexo local privado de evidencias por caso, excluido de Git. No cambia contratos, registros MCR, código, configuración, CSV, notebooks ni archivos de Drive; no realiza commit ni publicación. El próximo producto recomendado es la auditoría diagnóstica especificada en 6.1. La decisión sobre una corrección vendrá determinada por ese resultado, no por una promesa previa de aprobación.

### Seguimiento posterior de la propuesta

Los estados anteriores describen la entrega de análisis, conservada como instantánea histórica. Posteriormente, el investigador autorizó preparar y colocar en Drive la libreta diagnóstica descrita en 6.1. La implementación equivalente y su verificación se documentan en [29_LIBRETA_DIAGNOSTICA_MCR004.md](29_LIBRETA_DIAGNOSTICA_MCR004.md). Esa autorización no adopta un algoritmo corregido ni modifica el cierre rechazado. La interpretación de la ejecución real en Colab permanece pendiente.
