# Fase 6, paso 2 — preparación local radiográfica y multimodal

Fecha: 9 de noviembre de 2026. **Estado: código, entorno y metadatos preparados; carga y ejecución de los pesos entrenados pendientes de descarga.** No se habilita el resultado del médico.

## Preparado y comprobado

- Entorno CPU independiente, fuera de FastAPI y del entorno clínico del paso 1. Python 3.12.14, NumPy 1.26.4, Torch 2.6.0+cpu, TorchVision 0.21.0+cpu, Transformers 4.50.2, safetensors 0.5.3 y timm 1.0.15. Dependencias en `backend/requirements-neural-probe.lock`; `pip check` e importaciones correctas. La combinación CPU sigue la receta existente y la distribución oficial de [PyTorch](https://pytorch.org/get-started/previous-versions/#v260).
- Cinco trabajos radiográficos SkelEx de etapa 2 y cinco multimodales de fusión, identificados por sus recibos originales. No se sustituyen por los pesos anteriores ni se eligen de nuevo por rendimiento.
- Manifiesto privado fijado por SHA con diez trabajos, diez estados clínicos, 140 fuentes científicas, 15 archivos públicos de receta/licencia y tres archivos externos de SkelEx. Solicitudes, recibos y estados verificados; ningún peso vacío se crea para ocupar su lugar.
- Los tres archivos externos oficiales fueron descargados y verificados por tamaño/SHA. Son soporte del constructor original, no sustituyen los diez pesos entrenados.
- Cliente y worker por proceso: un trabajo y un pliegue cada vez, CPU explícita, sin importar los frameworks en el backend web. Se reutiliza `_neural_logits`, incluido su constructor, carga estricta, normalización, modo de evaluación y comprobación de parámetros inmutables.
- **431 pruebas del backend correctas**, sin omisiones. Incluyen protocolo, selección de archivos, rechazo de promoción clínica, rutas, temporales, errores seguros y conservación del comportamiento previo.
- Chequeo instrumentado real del entorno y las guardas: imports correctos, receta original autenticada y dos entradas ficticias normalizadas a `[3,224,224]`, float32 CPU, finitas. Fuentes y recetas conservaron sus huellas. Este chequeo no construyó un modelo, cargó pesos entrenados ni ejecutó un forward neuronal.
- Comprobación real de archivo faltante: devuelve `ASSET_MISSING` y `weightsVerified=false`. No genera una salida alternativa.

El diagnóstico del worker usa únicamente dos imágenes grises ficticias fijadas en el código y dos basales ficticios. No acepta datos de paciente, radiografías, particiones o desenlaces. No aplica sigmoid, promedio de pliegues, calibración, umbral ni clasificación; conserva `clinicalReleaseReady=false`.

## Descarga privada y primer lote

El conector de Drive rechazó los archivos de aproximadamente 1,21 GB por su límite de 256 MiB por archivo. El control automatizado del navegador también falló. El investigador eligió preparar enlaces y descargar en D; no se modificaron permisos de Drive.

Se prepararon diez enlaces canónicos y carpetas separadas en un listado HTML privado. El primer lote contiene solo el radiográfico de pliegue 0 y el multimodal de pliegue 0: **aproximadamente 2,43 GB**. Cada archivo conserva el nombre `best_model.safetensors` en su propia carpeta. Después se verificará su tamaño y SHA antes de usarlo. Los tamaños exactos y las huellas permanecen en el inventario privado.

Los diez pesos suman aproximadamente 12,13 GB. Con el soporte externo son unos 13,45 GB, sin contar entorno y una copia temporal. No se descargan imágenes de participantes ni se publica ninguno de estos activos en GitHub o en un registro Docker.

La primera comprobación con pesos será carga estricta del primer radiográfico y del primer multimodal. Solo tras aprobarla se ejecutarán las dos entradas ficticias y se comparará con la misma llamada CPU original. Esa paridad no demuestra equivalencia bit a bit con GPU/AMP ni capacidad predictiva o utilidad clínica.

## Herramienta de comprobación

El archivo privado de configuración contiene exactamente `python`, `checkout`, `bundle`, `manifestSha256` y `scratchRoot`. Todas las rutas son explícitas y se rechazan enlaces. El directorio de temporales pertenece exclusivamente a esta comprobación; no se adoptan ni eliminan carpetas de otros procesos.

Desde la raíz del repositorio:

```powershell
$env:PYTHONPATH=(Resolve-Path app/backend).Path
app/backend/.venv/Scripts/python.exe -m knee_app.neural_probe_check --config CONFIG_PRIVADA_ABSOLUTA --slot radiographic --fold 0 --report INFORME_PRIVADO_NUEVO
app/backend/.venv/Scripts/python.exe -m knee_app.neural_probe_check --config CONFIG_PRIVADA_ABSOLUTA --slot multimodal --fold 0 --report OTRO_INFORME_PRIVADO_NUEVO
```

`--diagnostic` solicita las dos fixtures internas, después de comprobar los pesos. El tiempo es acotado por trabajo, predeterminado de 300 segundos y máximo de 600. Un timeout no provoca reintento, cambio de dispositivo o resultado simulado. Los informes son nuevos y privados; la consola omite logits y rutas.

La guardia bloquea conexiones, resolución de nombres, procesos adicionales y escrituras fuera del scratch privado. Permite consultar el nombre local del equipo y leer por introspección dos archivos propios del worker, fijados por SHA/tamaño antes y después. Las fuentes científicas y recetas solo pueden leerse si están fijadas en el manifiesto. La comprobación instrumentada permite además su propio archivo de harness privado; esa excepción no existe en el protocolo del worker. Las guardas Python evitan I/O accidental, sin presentarse como aislamiento de código nativo malicioso.

## Alcance pendiente

Todavía no se ha comprobado la carga de ningún peso entrenado radiográfico o multimodal en esta entrega. Tampoco se midieron su memoria máxima, latencia por caso, capacidad para un alojamiento ni paridad de sus logits. Las pruebas sintéticas y la importación de librerías no sustituyen esas comprobaciones.

Las operaciones pesadas permanecen en la CLI privada; no se cargan diez modelos en cada consulta HTTP de estado. La API y Vue conservan la predicción bloqueada y sus respuestas previas. El paquete final formalizado, calibrador y regla de clasificación continúan pendientes por su vía científica. No se cambió la metodología, reentrenó, ajustó nada ni abrió la prueba reservada. Alojamiento, dominio y despliegue científico Docker siguen fuera de esta preparación local.
