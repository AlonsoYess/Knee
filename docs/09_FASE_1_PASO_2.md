# Fase 1, paso 2: inventario y descarga selectiva

## Propósito

Consolidar las adquisiciones radiográficas basales esperadas por la cohorte y controlar su descarga selectiva, sin transferir imágenes ajenas al estudio. Este paso trabaja a nivel de adquisición bilateral y no utiliza KL de seguimiento, progresión ni ninguna otra etiqueta de desenlace.

## Base verificada

El manifiesto privado contiene 1,916 adquisiciones basales únicas, una por participante, enlazadas exactamente con 2,778 rodillas elegibles. De esas adquisiciones, 1,054 aportan una rodilla y 862 aportan ambas. Las 1,916 rutas de imagen y las 1,916 claves de adquisición son únicas; cada ruta termina en el identificador de ocho dígitos correspondiente al código radiográfico basal.

Estos recuentos validan el contrato de entrada. No representan todavía la cohorte radiográfica utilizable, que se cerrará después de descargar, leer y controlar los DICOM.

## Procedimiento reproducible

1. Leer únicamente las columnas de identidad radiográfica necesarias de las hojas privadas de adquisiciones únicas y rodillas.
2. Comprobar el número esperado de adquisiciones, la unicidad de claves, la correspondencia entre ruta y código basal, y la relación exacta entre cada adquisición y sus rodillas elegibles.
3. Recorrer solo las carpetas privadas configuradas para el piloto aprobado y la cohorte V00.
4. Abrir cada paquete TAR sin extraerlo, localizar un único objeto DICOM de imagen, decodificar sus píxeles y calcular huellas SHA-256 de paquete, objeto DICOM y matriz de píxeles.
5. Clasificar cada adquisición sin usar etiquetas:

   - `DISPONIBLE`: existe un único paquete legible;
   - `PENDIENTE_DESCARGA`: aún no existe un paquete local; no equivale a ausencia en la fuente;
   - `ILEGIBLE`: el paquete existe, pero no contiene una imagen DICOM decodificable única;
   - `DUPLICADO_EQUIVALENTE`: existen varias copias con los mismos píxeles;
   - `DUPLICADO_CONFLICTIVO`: una misma clave contiene píxeles distintos;
   - `DUPLICADO_PARCIALMENTE_ILEGIBLE`: coexisten una copia válida y otra inválida.

6. Generar una cola privada solamente para las adquisiciones pendientes o que requieran reemplazo.
7. Dividir la cola, en orden determinista, en lotes operativos de 100 elementos. El tamaño de lote es ajustable, solo organiza la transferencia y no modifica la muestra ni la metodología.
8. Congelar la asignación adquisición–lote antes de la primera descarga. Las adquisiciones completadas salen de la cola, pero las restantes conservan su número original y nunca se renumeran.
9. Volver a ejecutar la misma auditoría después de cada lote. Una adquisición deja automáticamente la cola cuando su paquete legible aparece en una de las carpetas controladas.

## Descarga autorizada

La descarga se realizará con la cuenta autorizada del investigador en NDA. El mecanismo principal será el cliente oficial `nda-tools` versión 0.7.0: `downloadcmd` acepta mediante `-t` un archivo de texto con las rutas S3 exactas que deben descargarse. El código construirá ese archivo directamente desde el lote elegido de la cola privada; así no dependerá de selección manual ni incorporará imágenes ajenas a la cohorte.

Cada lote se descargará primero al almacenamiento temporal de Colab. Antes de copiar un archivo a Drive, el proceso exigirá que el lote esté completo, que cada paquete contenga una imagen DICOM decodificable y que no existan rutas inesperadas, duplicados ni conflictos de contenido. Solo entonces se guardará bajo `dicom/originales/cohorte_v00/lote_NNN/`, se volverán a verificar las huellas y se actualizará el inventario. Los diez archivos ya aprobados del piloto no se descargarán de nuevo.

El usuario y el identificador de paquete se leerán de secretos de Colab. La contraseña se solicitará de forma oculta, permanecerá únicamente en un almacén efímero del entorno y será eliminada al terminar. No se pegarán ni persistirán credenciales en GitHub, Drive, notebooks, documentos o registros.

La celda de autenticación puede reintentarse de forma segura dentro del mismo entorno: la carpeta temporal de registros se crea de manera idempotente. La salida ordinaria del cliente permanece oculta para no exponer rutas o identificadores; si el cliente falla, el notebook presenta únicamente un diagnóstico saneado que sustituye usuario, paquete, rutas S3, ubicaciones temporales y nombres de paquetes. Un rechazo `401` se distingue expresamente de un problema de autorización sobre el paquete.

Referencias operativas oficiales:

- [NDA Download Manager User Guide](https://nda.nih.gov/static/docs/NDA_Download_Manager_User_Guide_v0.1.39.pdf)
- [NDA: acceso a archivos y S3 links](https://nda.nih.gov/s/guid/nda-guid.html)
- [Cliente oficial nda-tools](https://github.com/NDAR/nda-tools)
- [nda-tools 0.7.0 en PyPI](https://pypi.org/project/nda-tools/0.7.0/)

## Evidencias privadas en Drive

La ejecución crea, dentro de `outputs/auditorias/fase_1/paso_2_inventario/`:

- `resumen_inventario_publico.json`: solo recuentos agregados y estado del proceso;
- `inventario_adquisiciones_privado.csv`: una fila por adquisición, con estado, rutas y huellas;
- `cola_descarga_selectiva_privada.csv`: pendientes ordenados y agrupados por lote;
- `plan_descarga_congelado_privado.csv`: asignación inmutable de las 1,906 adquisiciones pendientes iniciales a sus lotes;
- `auditoria_archivos_privada.csv`: detalle técnico de cada paquete encontrado;
- `archivos_inesperados_privado.csv`: archivos ubicados en las carpetas controladas que no pertenecen al manifiesto;
- `metadatos_inventario_privado.json`: huella del manifiesto y raíces inspeccionadas;
- `registro_ejecucion_fase1_paso2.json`: fecha, commit y recuentos de la ejecución.

Ningún archivo con identificadores, rutas o huellas individuales se publicará en GitHub. La libreta `02_inventario_adquisiciones.ipynb` solo monta Drive, obtiene el código versionado, ejecuta las pruebas y llama al módulo `knee.acquisition_inventory`.

## Reglas de detención

La ejecución se detiene si cambia el contrato de 1,916 adquisiciones, si una ruta deja de corresponder con su código basal, si falla la relación con las 2,778 rodillas, si aparece contenido conflictivo bajo una misma clave o si existen paquetes inesperados en las carpetas controladas. Un archivo no descargado se mantiene como pendiente y nunca se declara ausente en origen por inferencia.

La promoción de un lote es atómica a nivel lógico: si cualquiera de sus adquisiciones falta o falla la auditoría, ninguna se incorpora como lote aprobado en Drive. Una nueva ejecución puede reutilizar archivos idénticos ya promovidos, pero se detiene si encuentra el mismo nombre con contenido diferente.

## Criterio de cierre

El paso 2 podrá cerrarse cuando:

1. las 1,916 adquisiciones conserven una clasificación verificable;
2. la cola de descarga no contenga pendientes que no hayan sido intentados;
3. los paquetes disponibles sean legibles y tengan huellas registradas;
4. cualquier ausencia real, ilegibilidad persistente o duplicado haya sido investigado y documentado;
5. no existan archivos ajenos a la cohorte en las carpetas auditadas;
6. la evidencia privada y el commit ejecutado estén guardados en Drive.

Este cierre no autoriza separación bilateral masiva, recortes, particiones, entrenamiento ni apertura de la prueba reservada.
