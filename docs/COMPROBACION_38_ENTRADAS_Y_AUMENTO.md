# Comprobación 38: variables clínicas complementarias y aumento

Esta etapa prepara y comprueba entradas del plan 1.2. No ajusta estimadores, crea optimizadores, calcula gradientes, selecciona nuevos estudios ni abre el archivo reservado de prueba. La ejecución real de Colab corresponde al investigador; una comprobación local se identifica expresamente como tal.

## Datos privados de entrada

Se utiliza el mismo paquete de desarrollo ya verificado y la referencia de la libreta 32. La proyección adicional contiene únicamente sus participantes y lados, con edad, sexo, IMC, KL inicial, cirugía previa y WOMAC total de V00. Los tres predictores principales deben coincidir con el paquete existente.

`scripts/project_development_baseline.py` genera esa proyección desde el CSV auditado con su hash externo. Resuelve identidad y lado antes de extraer los campos clínicos; descarta los registros ajenos a desarrollo. El análisis sintáctico del CSV recorre sus bytes, pero no convierte ni guarda columnas de seguimiento o desenlaces. No lee un archivo de prueba. Se comprueban la cobertura exacta, los dominios de las variables y los hashes de las fuentes antes y después.

La carpeta privada de desarrollo recibe `development_baseline_projection_v1.json` y `development_baseline_descriptor_v1.json`. El descriptor vincula el hash y tamaño de la proyección, la fuente auditada, la visita basal y el paquete de desarrollo. Ambos archivos permanecen fuera de GitHub. No sustituyen ni alteran el paquete original.

## Ejecución en Colab

Abrir la libreta 38, seleccionar **CPU** y ejecutar las celdas en orden. La libreta solicita las mismas dos carpetas habituales: registros privados y desarrollo con `transport.json`. Reconstruye solo el transporte de desarrollo y lee únicamente los dos archivos adicionales previstos.

La instalación utiliza un intérprete temporal aislado con las versiones ya fijadas. Se ejecutan las pruebas sintéticas y después la comprobación de los datos reales. La lógica permanece en `src/knee`; la libreta llama a sus comandos.

La preparación genera tres escenarios clínicos por cinco pliegues: 15 estados de imputación y escalado, 30 matrices y 45 solicitudes descriptivas de futuros trabajos. Cada estadística se calcula solo en el entrenamiento del pliegue correspondiente. Las solicitudes no contienen autorización para entrenar.

El aumento se comprueba en todas las posiciones de entrenamiento y validación de los cinco pliegues, en la primera época. Se verifican la reproducción del mismo sorteo, el relleno, la geometría, la lateralidad, la ausencia de cambios en los originales y la normalización de los seis codificadores accesibles. No se construyen redes con pesos preentrenados.

Las tres vistas privadas se seleccionan de manera determinista entre las rodillas de entrenamiento del pliegue 0: primera variación gamma menor que uno, primera mayor que uno y primer control sin aumento, con casos distintos de respaldo si falta una categoría. La selección no consulta etiquetas, variables clínicas ni apariencia de la imagen. Son copias de visualización; no acreditan validación anatómica ni mejora predictiva.

## Integridad y portabilidad

Los hashes históricos de ocho archivos de texto se obtuvieron en Windows con finales CRLF, mientras que Git puede entregarlos con LF. El cargador nuevo acepta exclusivamente esas representaciones del mismo texto registrado cuando el hash completo de una de ellas coincide con el hash original. No cambia el hash esperado, no normaliza contenido adicional ni modifica los archivos históricos. Además, reproduce y compara todos los estados y matrices clínicos de la referencia anterior. Los hashes de `handoff.json` y del paquete privado se comprueban sobre sus bytes exactos.

Las comprobaciones originales de las libretas anteriores conservan sus controles. Esta compatibilidad se implementa únicamente en el cargador de la versión 1.2.

## Resultado

Se guarda una carpeta privada nueva junto a los registros anteriores y se verifica cada archivo mediante lectura de vuelta. El JSON final indica si se ejecutó realmente en Colab, sus recuentos, los controles aprobados y que no hubo entrenamiento ni acceso predictivo a prueba. El investigador comparte únicamente ese JSON. La revisión de las tres vistas precede a cualquier autorización posterior de entrenamiento.
