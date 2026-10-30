# Código público de preparación

`configs/public_preprocessing_sources_v1.json` identifica las fuentes de preparación incorporadas al repositorio y sus huellas del texto con saltos de línea LF. Los archivos científicos enumerados se conservan sin cambiar su lógica. El inventario permite verificar la copia pública y no reemplaza las huellas de los artefactos privados originales.

La publicación cubre la consolidación de la cohorte tras el control radiográfico, las verificaciones de los recortes aceptados, la adaptación física a 224 × 224, la auditoría de encabezados DICOM, las particiones por participante y la preparación y comprobación del transporte para Colab. También incluye las funciones de geometría basadas en coordenadas externas y sus pruebas sintéticas.

Los comandos de `scripts/` reciben las configuraciones privadas correspondientes. Esas configuraciones describen los archivos de entrada y sus huellas; no se sustituyen por valores ficticios para reproducir una ejecución. Publicar estos comandos no autoriza volver a seleccionar participantes, crear otras particiones o procesar imágenes.

Las libretas 31 se conservan como fuentes históricas limpias, sin resultados ni datos de participantes. Las copias operativas anteriores, sus cápsulas de restauración, las observaciones por caso y las configuraciones que contienen referencias a evidencia privada permanecen en Drive. No se modifican para hacerlas coincidir con una copia pública nueva. El entrenamiento actual utiliza el paquete de desarrollo sellado y los ejecutores metodológicos versionados posteriores.

Las pruebas `test_public_preprocessing_sources` comprueban las huellas de las fuentes publicadas, la presencia de las dependencias de las tres entradas principales de preparación y la ausencia de resultados en las libretas históricas incluidas. Estas comprobaciones utilizan código y casos sintéticos; no reconstruyen la cohorte ni abren la prueba reservada.
