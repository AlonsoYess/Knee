# Propuesta técnica de `tibiofemoral_crop_v0.4_pilot`

## Estado y límite de autorización

**Antecedente aprobado e implementado; ejecución privada pendiente.** Este documento respondió al rechazo visual de `v0.3` y definió una corrección reproducible sobre las mismas veinte rodillas del piloto. El investigador autorizó posteriormente implementar `v0.4` y publicar los cambios en la rama actual. La implementación preparada se documenta en [`21_IMPLEMENTACION_FASE_1_PASO_4_V04.md`](21_IMPLEMENTACION_FASE_1_PASO_4_V04.md); todavía no se ha ejecutado en Colab ni revisado visualmente.

La propuesta conserva:

- el enfoque determinista y multiseñal previsto en el capítulo III;
- las mismas diez adquisiciones y veinte rodillas de desarrollo;
- la separación bilateral y lateralidad ya congeladas;
- el campo candidato de 140 × 140 mm calculado desde el espaciado DICOM;
- la revisión visual ciega al desenlace;
- la prohibición de ajustes manuales o coordenadas particulares por rodilla.

No cambia población, predictores, desenlace, horizonte, unidad de análisis, particiones, métricas ni modelos. Procesamiento masivo, particiones, entrenamiento y prueba reservada permanecen bloqueados.

## Evidencia que motiva la corrección

Las tres versiones deterministas produjeron 9/20, 8/20 y 0/20 recortes íntegramente aceptables. En `v0.2` la anatomía quedó completa en las veinte vistas, pero persistieron ocho errores verticales y cinco campos con regla o puntos. En `v0.3` persistieron ocho errores verticales y aparecieron once campos con anatomía incompleta y catorce con borde, fondo negro o puntos de la regla.

El cruce entre las métricas automáticas privadas y la revisión visual de `v0.3` muestra:

- 19/20 resultados automáticos `LOW` y solo uno `HIGH`;
- 15/20 cajas con desplazamiento de límite superior al máximo configurado;
- 12/20 desacuerdos verticales automáticos y 8/20 líneas visualmente no centradas;
- 19/20 casos superaron la puerta de fuerza de bordes, por lo que esa puerta no discriminó adecuadamente el centrado real;
- la única candidata `HIGH` fue rechazada porque conservó puntos de la regla;
- no hubo fallos ni exclusiones técnicas.

El código vigente explica la regresión horizontal. La extensión detectada de componentes brillantes aumenta el margen interno y ese margen se entrega al mismo ajuste que posiciona la caja. Cuando el margen crece, la caja puede ser empujada hacia el límite exterior del campo unilateral, aunque eso corte un compartimento o introduzca fondo negro. La detección de artefactos quedó, por tanto, acoplada indebidamente a la conservación anatómica.

El fallo vertical también es estructural. Los pares de bordes se buscan en una banda amplia y se ordenan con una priorización vertical suave. Si no existe una combinación concordante entre compartimentos, la implementación actual utiliza de todas formas los mejores pares independientes para producir una línea. La concordancia entre dos señales de una misma familia no demostró suficiente especificidad para distinguir la interlínea de otros bordes oscuros.

## Diseño propuesto

### 1. Caja horizontal gobernada primero por anatomía

`v0.4` dejará de convertir la extensión de la regla en un desplazamiento obligatorio. Primero estimará una envolvente anatómica horizontal reproducible alrededor de la región articular y enumerará las posiciones posibles de una caja de 140 mm.

Una posición solo será anatómicamente elegible si:

- contiene el centro anatómico y las bandas medial y lateral usadas para localizar la interlínea;
- conserva la envolvente ósea de ambos compartimentos con un resguardo físico configurado;
- no excede el límite permitido de fondo ni el desplazamiento máximo respecto del centro anatómico.

Entre las posiciones elegibles se escogerá la que maximice la conservación anatómica y minimice, de manera subordinada, artefactos periféricos y fondo. La regla, los puntos y los bordes actuarán como **puertas de rechazo**, no como fuerzas capaces de sacar la caja de la rodilla.

Si ninguna posición de 140 mm conserva simultáneamente la anatomía y excluye los artefactos, el resultado será `REVIEW_REQUIRED_NO_FEASIBLE_CROP`. Se conservará para revisión la mejor caja anatómica, pero nunca recibirá confianza `HIGH`, no se reducirá el campo y no se aplicará una corrección manual.

### 2. Consenso vertical entre familias de señal

La nueva línea candidata deberá estar respaldada por dos estimadores diferentes:

1. el máximo multiseñal por compartimento de `v0.2`, que combina oscuridad, contraste óseo y gradiente;
2. el par de bordes dirigidos de `v0.3`, que exige el orden fémur–espacio oscuro–tibia.

La aceptación automática requerirá concordancia física entre ambos estimadores en cada compartimento y concordancia medial–lateral. También se comprobará que exista soporte óseo continuo por encima y por debajo del espacio candidato en varias subbandas horizontales, para evitar que una espina, un borde condilar interno o una estructura inferior sea tomada como interlínea.

Si no existe consenso, no se declarará una línea confiable a partir del promedio de candidatos incompatibles. El resultado quedará explícitamente como `REVIEW_REQUIRED_VERTICAL_DISAGREEMENT`.

### 3. Detector periférico como validador independiente

La detección de regla, puntos, borde de placa y fondo negro se evaluará sobre la caja anatómicamente elegible ya construida. Se registrarán por separado:

- componentes compactos brillantes compatibles con puntos de regla;
- estructuras lineales repetidas compatibles con regla;
- contacto con borde o fondo negro;
- distancia física entre esos componentes y la anatomía.

Los clips quirúrgicos internos no se clasificarán automáticamente como artefacto periférico solo por ser brillantes: la decisión exigirá localización periférica y patrón geométrico compatible. Ninguna puntuación vertical favorable podrá compensar una puerta periférica fallida.

### 4. Confianza conservadora y trazable

`HIGH` requerirá simultáneamente:

- consenso vertical entre las dos familias de señal;
- concordancia medial–lateral;
- envolvente anatómica completa dentro de la caja;
- ausencia de regla, puntos, borde y fondo inadmisibles;
- desplazamiento y saturación dentro de límites;
- campo físico exacto de 140 × 140 mm.

El valor continuo de confianza se conservará solo como diagnóstico. Cada puerta y cada motivo de revisión quedarán en la salida privada; el resumen público contendrá únicamente recuentos agregados y la versión del algoritmo.

## Verificación previa a una nueva revisión visual

La eventual implementación deberá incorporar pruebas sintéticas y de regresión que demuestren:

1. que aumentar la extensión detectada de la regla no puede mover el centro articular fuera de la caja;
2. que las bandas de ambos compartimentos y la envolvente anatómica permanecen dentro del recorte;
3. que una regla o sus puntos producen rechazo y no desplazamiento anatómico;
4. que el fondo negro y el contacto con el borde invalidan la confianza alta;
5. que un par de bordes por encima o por debajo de la interlínea es rechazado cuando no concuerda con el estimador multiseñal y el soporte óseo;
6. que no se promedian candidatos incompatibles para fabricar una línea confiable;
7. que el escenario equivalente a la falsa candidata `HIGH` de `v0.3` ya no supera todas las puertas;
8. que los comportamientos históricos de `v0.2` y `v0.3` siguen disponibles por despacho explícito y no son sobrescritos;
9. que código y configuración no contienen identificadores, rutas privadas, desenlaces ni ajustes por caso.

Después de superar las pruebas, una libreta nueva deberá verificar el cierre rechazado de `v0.3` y ejecutar `v0.4` únicamente sobre las mismas veinte rodillas. Las nuevas vistas y el CSV se someterán a otra revisión visual ciega completa.

## Criterio de aceptación y límite contra sobreajuste

El paso 4 solo podrá cerrarse si las veinte rodillas:

- se procesan sin fallos de integridad;
- tienen decisión visual completa;
- muestran la articulación centrada y la anatomía relevante completa;
- excluyen texto, bordes, rectángulos de anonimización y la regla;
- resuelven explícitamente toda confianza baja;
- son reproducibles sin correcciones manuales privilegiadas.

Se propone que `v0.4` sea la última iteración determinista ajustada con estas mismas veinte rodillas. Una cuarta versión rechazada indicaría que continuar afinando reglas sobre el mismo piloto aumenta el riesgo de sobreajuste sin evidencia de generalización. En ese caso no se preparará automáticamente una `v0.5`: se documentará el límite del enfoque y se presentará una propuesta metodológica separada para un localizador de puntos anatómicos con anotaciones, validación y particiones propias. Esa alternativa no queda autorizada por este documento.

## Autorización registrada

La aprobación recibida autorizó únicamente:

- implementar `tibiofemoral_crop_v0.4_pilot` con pruebas públicas;
- crear una configuración pública nueva;
- preparar una libreta nueva para las mismas veinte rodillas;
- colocar la copia aprobada de esa libreta en la carpeta privada de notebooks;
- realizar una nueva revisión visual ciega.

No autorizó ejecutar Colab en nombre del investigador, procesar la cohorte completa, crear particiones, entrenar modelos ni abrir la prueba reservada.
