# Fase 1, paso 1: reconciliación y auditoría del piloto DICOM

## Propósito

Este paso verifica que el piloto corresponda a diez adquisiciones radiográficas basales únicas antes de diseñar recortes o procesar la cohorte. No entrena modelos, no crea particiones y no utiliza el desenlace de progresión para aceptar o rechazar imágenes.

## Hallazgo que obliga a rehacer el control

La ejecución histórica informó diez paquetes DICOM legibles, pero ese número no equivalía a diez adquisiciones únicas. La reconstrucción reproducible confirmó que la carpeta usada entonces contenía nueve adquisiciones esperadas: una estaba representada por dos empaquetados del mismo contenido y otra adquisición del manifiesto no había sido incorporada. Por tanto, ese informe se conserva como antecedente, pero no puede cerrar el piloto.

La fuente local completa contiene las diez adquisiciones esperadas y copias redundantes de varios paquetes. Estas copias sirven para comprobar la reconciliación, pero no deben trasladarse como si fueran estudios adicionales.

## Controles implementados

El comando `knee-dicom-audit`:

1. lee únicamente las rutas esperadas del manifiesto privado;
2. reconoce paquetes `.tar`, `.tar.gz` y `.tgz` aunque la ruta esté codificada;
3. calcula huellas SHA-256 separadas para el paquete, el objeto DICOM y los píxeles decodificados;
4. identifica copias de empaquetado, omisiones, conflictos de contenido y la misma imagen asociada a dos claves;
5. exige lectura DICOM, matriz de píxeles no vacía y un solo objeto radiográfico por paquete;
6. registra cuadros, profundidad, fotometría, dimensiones, modalidad, proyección, lateralidad, espaciado y sintaxis de transferencia;
7. selecciona un único paquete canónico por adquisición;
8. produce un resumen público agregado y un detalle privado con rutas y huellas.

La huella de píxeles se utiliza para contar adquisiciones únicas; la huella del archivo DICOM y la del paquete se conservan además para trazabilidad e integridad. Ningún identificador individual ni ruta privada se publica en GitHub.

## Criterio de cierre

El paso queda cerrado solo cuando la ejecución estricta desde Drive demuestre simultáneamente:

- diez filas y diez claves únicas en el manifiesto piloto;
- diez paquetes canónicos, sin duplicados ni inesperados;
- diez DICOM legibles y diez huellas de píxeles distintas;
- un cuadro por adquisición y perfil técnico registrado;
- cero omisiones, conflictos de contenido o errores de lectura;
- evidencia privada guardada en `outputs/auditorias/fase_1/paso_1_piloto`;
- revisión identificable del repositorio y pruebas automáticas aprobadas.

## Ejecución controlada

La fuente editable de la libreta es `notebooks/01_auditoria_piloto_dicoms.ipynb`. En Colab monta Drive, obtiene `KNEE_DATA_ROOT`, instala una revisión identificable del repositorio, ejecuta las pruebas y llama al comando con `configs/pilot_audit.example.json`. La lógica de auditoría permanece en `src/knee/dicom_audit.py`.

Los resultados individuales, los paquetes y las huellas permanecen en Drive. El repositorio conserva solo el código, la configuración relativa, las pruebas y este protocolo público.

## Estado

**Cerrado el 29 de septiembre de 2026.** La libreta fue ejecutada en Colab sobre el commit `f540a60f7399f7f97b4f196169c9f5109d017381`. Las evidencias privadas de Drive confirman diez entradas esperadas, diez paquetes canónicos, diez DICOM legibles, diez huellas únicas de paquete, DICOM y píxeles, cero duplicados y cero controles fallidos.

El cierre público y sus límites están documentados en [`08_CIERRE_FASE_1_PASO_1.md`](08_CIERRE_FASE_1_PASO_1.md). Este resultado autoriza continuar con el inventario radiográfico del paso 2, pero no autoriza entrenamiento, procesamiento masivo sin control ni apertura de la prueba reservada.
