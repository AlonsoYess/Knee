# Propuesta técnica de `tibiofemoral_crop_v0.3_pilot`

## Estado y límite de autorización

**Propuesta documentada; no implementada ni autorizada para ejecución.** Este documento responde al rechazo visual de `v0.2` y define una corrección reproducible dentro de las mismas veinte rodillas del piloto. No modifica todavía código, configuraciones ni notebooks.

> Actualización posterior del 30 de septiembre de 2026: la implementación fue autorizada y quedó preparada localmente. Su estado vigente se documenta en [`18_IMPLEMENTACION_FASE_1_PASO_4_V03.md`](18_IMPLEMENTACION_FASE_1_PASO_4_V03.md). El piloto privado todavía no se ha ejecutado.

La propuesta conserva:

- el enfoque determinista y multiseñal previsto en el capítulo III;
- las mismas diez adquisiciones y veinte rodillas de desarrollo;
- la separación bilateral y lateralidad ya congeladas;
- el campo candidato de 140 × 140 mm calculado desde el espaciado DICOM;
- la revisión visual ciega al desenlace;
- la prohibición de ajustes manuales por rodilla.

No cambia población, predictores, desenlace, horizonte, unidad de análisis, particiones, métricas ni modelos. Si la corrección necesitara una red entrenada u otra arquitectura de localización, se detendrá el paso y se tramitará un cambio metodológico antes de implementarla.

## Evidencia que motiva la corrección

La revisión de `v0.2` encontró ocho localizaciones verticales incorrectas: siete desplazadas hacia la tibia y una hacia el fémur. Cinco campos conservaron puntos de la regla. La anatomía quedó completa en las veinte vistas y no hubo exclusiones técnicas.

El código vigente explica por qué la concordancia no basta:

1. el perfil vertical convierte el gradiente en valor absoluto, por lo que pierde la dirección de los bordes;
2. una banda oscura superior o inferior puede competir con el espacio articular si produce oscuridad, contraste y gradiente altos;
3. la concordancia entre compartimentos puede ser alta cuando ambos seleccionan el mismo máximo equivocado;
4. la confianza combina señales en una suma ponderada y no impone que cada puerta anatómica sea válida;
5. la protección de la regla utiliza un margen interno fijo de 12 mm, pero algunos puntos penetran más allá de esa franja.

## Diseño propuesto

### 1. Candidato vertical basado en dos bordes dirigidos

Para cada compartimento se conservará el perfil mediano suavizado, pero se calculará el gradiente **con signo**. Un candidato de espacio articular deberá estar delimitado, en orden vertical, por:

1. una transición compatible con la salida del cóndilo femoral brillante hacia una banda más oscura;
2. una transición opuesta compatible con la entrada a la meseta tibial brillante.

El centro candidato será el punto medio del par de bordes, no el máximo aislado de oscuridad. Los pares se puntuarán mediante la fuerza de ambos bordes, la oscuridad interna, el brillo óseo inmediatamente externo y una priorización vertical suave. La separación admisible entre bordes se expresará en milímetros usando el espaciado DICOM y se fijará en configuración pública antes de ejecutar el piloto; no existirán coordenadas particulares por caso.

### 2. Consenso anatómico entre compartimentos

Cada compartimento conservará sus mejores pares de bordes. La selección conjunta exigirá proximidad entre los centros medial y lateral y compatibilidad de sus anchos físicos. Si no existe un par concordante, el resultado será `REVIEW_REQUIRED_VERTICAL_DISAGREEMENT`; no se promediarán candidatos incompatibles para producir una falsa confianza alta.

### 3. Banda periférica prohibida para la regla

La correspondencia congelada de lateralidad continuará determinando cuál es el borde interno. Sobre ese borde se estimará una banda prohibida a partir de componentes brillantes o de alto contraste repetidos dentro de la región vertical de trabajo. El límite efectivo será el mayor entre:

- el margen físico mínimo configurado;
- la extensión detectada del artefacto más un resguardo físico reproducible.

La caja de 140 mm deberá caber completamente fuera de esa banda. Si no cabe, el resultado será `REVIEW_REQUIRED_ARTIFACT_CLEARANCE`; no se reducirá el campo, no se incluirá la regla y no se excluirá silenciosamente el DICOM.

### 4. Puertas de confianza separadas

La decisión automática dejará de depender exclusivamente de una suma ponderada. `HIGH` requerirá simultáneamente:

- par de bordes verticales válido en ambos compartimentos;
- concordancia de centros y anchos;
- caja completa sin desplazamiento inadmisible por límites;
- distancia horizontal suficiente respecto de la banda prohibida;
- fracciones de fondo y saturación dentro de los límites configurados.

El valor continuo de confianza podrá conservarse como diagnóstico, pero ninguna señal favorable compensará una puerta obligatoria fallida.

### 5. Trazabilidad adicional

La salida privada registrará, como mínimo:

- filas de los cuatro bordes dirigidos seleccionados;
- centros y anchos físicos por compartimento;
- concordancia vertical y de ancho;
- extensión detectada de la banda periférica;
- holgura final respecto de la regla;
- estado individual de cada puerta de confianza;
- motivo explícito de revisión cuando corresponda.

Las salidas públicas conservarán solo recuentos agregados y la versión del algoritmo.

## Verificación previa a una nueva revisión visual

La eventual implementación deberá incorporar pruebas que demuestren:

1. selección correcta de un espacio oscuro delimitado por gradientes opuestos en perfiles sintéticos;
2. rechazo de máximos oscuros falsos situados por encima o por debajo de la interlínea;
3. desacuerdo explícito cuando los compartimentos proponen centros incompatibles;
4. exclusión de artefactos que se extienden más allá del margen fijo anterior;
5. imposibilidad de declarar `HIGH` cuando falle una puerta obligatoria;
6. ausencia de identificadores, rutas privadas, desenlaces y ajustes por caso en código o configuración.

Después de superar las pruebas, una libreta nueva deberá reproducir primero el rechazo de `v0.2` y ejecutar `v0.3` únicamente sobre las mismas veinte rodillas. Sus vistas y CSV se someterán a una nueva revisión ciega independiente.

## Criterio de aceptación

El paso 4 solo podrá cerrarse si las veinte rodillas:

- se procesan sin fallos de integridad;
- tienen decisión visual completa;
- muestran la articulación centrada y la anatomía relevante completa;
- excluyen texto, bordes, rectángulos de anonimización y la regla;
- resuelven explícitamente toda confianza baja;
- son reproducibles sin correcciones manuales privilegiadas.

Solo después se congelarán parámetros y podrá autorizarse el paso 5. Procesamiento masivo, particiones, entrenamiento y prueba reservada continúan bloqueados.
