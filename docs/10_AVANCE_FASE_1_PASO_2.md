# Avance verificado de la Fase 1, paso 2

## Control

| Campo | Valor |
| --- | --- |
| Estado | En curso; inventario inicial y lote 1 verificados |
| Fecha | 29 de septiembre de 2026 |
| Revisión del inventario inicial | `ca52f0f9253b35a1318861cd3d89f5953cebcdad` |
| Revisión ejecutada para el lote 1 | `6c093a9db9b20cb1ed7762616539e801bc04f618` |
| Siguiente operación | Ejecutar el notebook corregido para el lote 2 |

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

## Resultado verificado del lote 1

El primer lote fue descargado con la cuenta autorizada del investigador, auditado íntegramente en el almacenamiento temporal de Colab y promovido a Drive solo después de superar todos los controles. La evidencia pública y privada, la carpeta canónica y el inventario actualizado fueron contrastados directamente en Drive.

| Control | Resultado |
| --- | ---: |
| Paquetes esperados y encontrados | 100 |
| Paquetes legibles | 100 |
| Adquisiciones únicas seleccionadas | 100 |
| Huellas únicas de paquete, DICOM y píxeles | 100 en cada nivel |
| Faltantes, ilegibles, duplicados, conflictos o inesperados | 0 |
| Paquetes promovidos a Drive | 100 |
| Adquisiciones disponibles acumuladas | 110 |
| Adquisiciones pendientes | 1,806 |
| Lotes restantes | 19, numerados del 2 al 20 |

La ejecución no entrenó modelos ni abrió la prueba reservada. El paso 2 permanece abierto hasta completar o documentar los lotes restantes.

Durante la primera autenticación se detectó un rechazo `401` porque el cliente de línea de comandos requiere la contraseña específica de NDA Tools, distinta del flujo web RAS/Login.gov con multifactor. Tras establecerla desde `Update Password`, la autenticación, consulta del paquete y descarga funcionaron correctamente. Esta incidencia no modificó datos ni metodología.

La libreta se corrigió para que la celda de autenticación sea reejecutable, elimine siempre la credencial efímera y entregue diagnósticos saneados sin revelar usuario, identificador de paquete, rutas S3, ubicaciones temporales ni nombres de paquetes. Esta es una corrección operativa y de seguridad, no un cambio metodológico.

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

El inventario inicial y el lote 1 están aprobados, pero el paso 2 sigue abierto hasta completar o documentar los 1,806 pendientes. La siguiente ejecución autorizable es el lote 2 con la libreta corregida y versionada. Este avance no autoriza separación bilateral masiva, recortes, particiones, entrenamiento ni apertura de la prueba reservada.
