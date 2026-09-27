# Fase 0  Base reproducible

## Estado y procedencia

El CSV auditado procede de una extracción autorizada Oracle/miNDAR de OAI. El identificador del paquete, el SQL maestro, el CSV y el libro de manifiestos se conservan en almacenamiento restringido; no se copian a este repositorio público. El capítulo III especifica el criterio de selección y las versiones de las tablas OAI. El código de esta fase verifica los archivos recibidos sin recalcular ni modificar la cohorte.

La ejecución local con los archivos entregados el 26 de septiembre de 2026 produjo: 2 778 rodillas, 1 916 participantes, 1 916 imágenes bilaterales vinculadas, 473 progresoras, 2 305 no progresoras y tres IMC ausentes. No se detectaron discrepancias en las verificaciones implementadas. Esto comprueba la integridad tabular y del manifiesto; la lectura y calidad de todos los DICOM siguen pendientes.

## Organización de Drive

Crear una carpeta privada `MyDrive/KneeOAI` con esta estructura:

```text
KneeOAI/
  data/
    cohorte_oai_48m_final.csv
    documentacion_cohorte_OAI_48m.xlsx
  dicom/             # archivos obtenidos selectivamente de OAI en la fase 1
  outputs/           # auditorías, particiones y bitácoras; se crea al ejecutar
```

El nombre local del CSV recibido incluye sufijos de descarga (`(1)(1)`); en Drive debe guardarse una copia con el nombre indicado arriba, sin cambiar su contenido. Si se elige otra organización, crear una copia local del JSON de configuración y cambiar solo sus rutas relativas. La variable de entorno `KNEE_DATA_ROOT` apunta a `KneeOAI`; no se edita el repositorio para incorporar una ruta privada.

## Ejecutar en Colab

Abrir [la libreta de arranque](../notebooks/00_arranque_colab.ipynb), conectar Drive y ejecutar las celdas en orden. La libreta clona este repositorio, instala el paquete, verifica el CSV contra la hoja `Manifiesto rodillas` y escribe en `outputs/` dos JSON agregados: auditoría y registro de ejecución. Una ejecución fallida se detiene antes de crear el registro. Volver a clonar o actualizar el repositorio cuando cambie el código; una libreta que ya encuentre `/content/Knee` avisa de esa situación para evitar ejecutar silenciosamente una revisión anterior.

También se puede ejecutar localmente:

```bash
python -m pip install -e .
export KNEE_DATA_ROOT=/ruta/privada/KneeOAI
knee-audit --config configs/paths.example.json
knee-run-record --config configs/paths.example.json
python -m unittest discover -s tests -v
```

En Windows PowerShell la variable se establece con `$env:KNEE_DATA_ROOT = 'C:\\ruta\\privada\\KneeOAI'`.

## Qué comprueba la auditoría

- Orden y presencia de las 21 columnas del CSV y campos esenciales del manifiesto.
- Clave única participante–rodilla, lateralidad 1/2 y un solo vínculo radiográfico basal por participante.
- KL basal 2/3, etiqueta coherente con incremento KL >= 1 y fecha V00 anterior a V06.
- Edad, sexo, IMC presente positivo, y conteos de IMC ausentes.
- Igualdad de las claves, barcode, ruta de imagen, KL, etiqueta y fechas entre CSV y manifiesto.
- SHA-256 de los dos archivos para identificar la versión exacta utilizada.

Los reportes contienen recuentos y huellas, no identificadores ni rutas de participantes. Deben permanecer en Drive hasta confirmar que sus condiciones de publicación permiten compartirlos. No se considera que una ruta S3 pruebe que un DICOM fue descargado o sea utilizable.

## Condición de cierre

Fase 0 cerrada cuando el repositorio ejecuta los controles localmente y en Colab con los archivos de Drive, sin diferencias de cohortes, y la bitácora registra la revisión del código y las huellas. La ejecución local ya está comprobada; la ejecución en la cuenta Colab/Drive del investigador está pendiente.
