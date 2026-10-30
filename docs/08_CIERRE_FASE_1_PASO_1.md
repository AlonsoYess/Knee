# Cierre de la Fase 1, paso 1

## Control

| Campo | Valor |
| --- | --- |
| Estado | Cerrado |
| Fecha | 29 de septiembre de 2026 |
| Revisión ejecutada en Colab | `f540a60f7399f7f97b4f196169c9f5109d017381` |
| Alcance | Reconciliación y auditoría técnica del piloto DICOM |

## Resultado

El manifiesto piloto y la carpeta canónica de Drive quedaron reconciliados. La ejecución estricta encontró diez entradas esperadas, diez paquetes, diez DICOM legibles y diez adquisiciones únicas. Las huellas de los paquetes, de los objetos DICOM y de los píxeles decodificados son distintas entre las diez adquisiciones. No se registraron duplicados, paquetes inesperados, omisiones, conflictos de contenido ni errores de lectura.

El detalle individual, las rutas, las huellas y los paquetes originales permanecen exclusivamente en Drive privado. Este documento contiene solo resultados agregados publicables.

## Evidencias verificadas

| Control | Resultado |
| --- | ---: |
| Filas del manifiesto piloto | 10 |
| Claves únicas del manifiesto | 10 |
| Paquetes canónicos encontrados | 10 |
| Paquetes DICOM legibles | 10 |
| Adquisiciones únicas seleccionadas | 10 |
| Copias duplicadas en la carpeta canónica | 0 |
| Controles fallidos | 0 |
| Huellas únicas de paquete | 10 |
| Huellas únicas de objeto DICOM | 10 |
| Huellas únicas de píxeles | 10 |

El registro de ejecución coincide con la revisión indicada, informa estado correcto y confirma que no se entrenó ningún modelo ni se abrió la prueba reservada. El archivo de detalle contiene diez registros válidos y seleccionados, sin código de error.

## Perfil técnico agregado

- Todas las adquisiciones contienen un solo cuadro, 16 bits almacenados y fotometría `MONOCHROME2`.
- Las dimensiones nativas observadas presentan cuatro combinaciones diferentes, por lo que no deben usarse coordenadas absolutas idénticas.
- Las modalidades registradas son ocho `CR` y dos `RG`.
- La proyección `PA` aparece en ocho adquisiciones y está vacía en dos.
- La lateralidad DICOM está vacía en nueve adquisiciones y figura como `L` en una; por ello no puede determinar por sí sola la lateralidad de imágenes bilaterales.
- Se observaron cuatro valores de espaciado de píxel, que deberán conservarse en el control de preprocesamiento.

## Límites del cierre

Este cierre acredita el piloto canónico y el funcionamiento reproducible de la auditoría. No acredita todavía:

- disponibilidad y legibilidad de todas las adquisiciones de la cohorte;
- separación bilateral ni lateralidad inferida por anatomía o marcadores;
- localización de la articulación tibiofemoral;
- aceptación de recortes para entrenamiento;
- cohorte analítica final ni particiones;
- autorización para entrenar modelos o consultar la prueba reservada.

## Siguiente paso

La continuación es la Fase 1, paso 2: consolidar el inventario de adquisiciones basales de la cohorte, contrastarlo con las rutas únicas del manifiesto y preparar su descarga selectiva. Cada adquisición deberá quedar clasificada como disponible, ausente, duplicada o ilegible antes de cualquier procesamiento masivo.
