# Avance verificado de la Fase 1, paso 2

## Control

| Campo | Valor |
| --- | --- |
| Estado | En curso; inventario inicial verificado |
| Fecha | 29 de septiembre de 2026 |
| Revisión ejecutada en Colab | `ca52f0f9253b35a1318861cd3d89f5953cebcdad` |
| Siguiente operación | Descarga selectiva por lotes desde NDA |

## Resultado verificado

La ejecución de Colab validó el contrato de 1,916 adquisiciones basales únicas, pertenecientes a 1,916 participantes y enlazadas con 2,778 rodillas elegibles. Encontró diez paquetes canónicos ya aprobados en el piloto y clasificó las 1,906 adquisiciones restantes como pendientes de descarga. No declaró ninguna de ellas ausente en origen.

Las evidencias guardadas en Drive fueron contrastadas de forma independiente:

| Control | Resultado |
| --- | ---: |
| Filas del inventario | 1,916 |
| Claves de adquisición únicas | 1,916 |
| Participantes únicos | 1,916 |
| Adquisiciones disponibles | 10 |
| Adquisiciones pendientes de descarga | 1,906 |
| Filas de la cola selectiva | 1,906 |
| Claves únicas en la cola | 1,906 |
| Lotes operativos | 20 |
| Lotes de 100 adquisiciones | 19 |
| Adquisiciones del último lote | 6 |
| Paquetes auditados y legibles | 10 |
| Huellas únicas de paquete, DICOM y píxeles | 10 en cada nivel |
| Archivos inesperados | 0 |

Los diez registros disponibles contienen las tres huellas requeridas. Los 1,906 pendientes no tienen una huella local, como corresponde a archivos aún no descargados. La revisión ejecutada coincide con el commit registrado; no se entrenó ningún modelo ni se abrió la prueba reservada.

## Mecanismo de descarga seleccionado

Se utilizará el cliente oficial `nda-tools` versión 0.7.0. Su comando `downloadcmd` admite una lista de rutas S3 exactas mediante `-t`; por tanto, cada lote se construirá directamente desde la cola privada y no mediante selección manual. El procedimiento:

1. genera un archivo temporal con una ruta S3 por línea para el lote elegido;
2. descarga únicamente esas rutas a almacenamiento temporal de Colab;
3. exige que estén presentes todos los paquetes esperados;
4. abre cada paquete, decodifica el DICOM y verifica contenido e integridad;
5. rechaza el lote completo ante omisiones, archivos inesperados, duplicados, conflictos o ilegibilidad;
6. copia a Drive solo el lote íntegro y registra sus huellas;
7. actualiza el inventario general y elimina las credenciales y archivos temporales.

La libreta `03_descarga_selectiva_lote.ipynb` implementa esta operación. El usuario y el identificador del paquete se suministran como secretos de Colab; la contraseña se solicita de forma oculta y se guarda únicamente en un almacén efímero que se elimina al terminar. Ninguna credencial se incorpora al repositorio, al notebook ni a Drive.

La asignación inicial a los veinte lotes queda congelada en un archivo privado. Después de completar un lote, sus adquisiciones desaparecen de la cola pendiente, pero todos los lotes restantes conservan su número y composición originales. Esta regla evita omisiones o saltos ocasionados por una renumeración dinámica.

## Estado del paso

El inventario inicial está aprobado, pero el paso 2 sigue abierto hasta completar o documentar los 1,906 pendientes. La siguiente ejecución autorizable es el lote 1. Este avance no autoriza separación bilateral masiva, recortes, particiones, entrenamiento ni apertura de la prueba reservada.
