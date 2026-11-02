# Cierre de la Fase 1, paso 3: separación bilateral y lateralidad

## Conclusión

El paso 3 de la Fase 1 queda cerrado técnicamente. La revisión visual ciega respaldó la separación bilateral y la correspondencia de lateralidad en las diez adquisiciones del piloto. Los parámetros de `bilateral_split_v0.2_pilot` quedaron congelados antes de cualquier procesamiento masivo.

Este cierre fue ejecutado sobre el commit `2489a7ec6253fb4aaa12818416f249780d26ea24`. Las evidencias individuales, las huellas de integridad y las ubicaciones exactas permanecen únicamente en Google Drive privado; este documento publica solo resultados agregados y parámetros técnicos reproducibles.

## Resultado verificado

| Control | Resultado |
| --- | ---: |
| Adquisiciones únicas revisadas | 10 |
| Separaciones aceptadas | 10 |
| Lateralidades respaldadas | 10 |
| Exclusiones técnicas | 0 |
| Casos de confianza alta | 9 |
| Casos de confianza baja | 1 |
| Casos limítrofes revisados y aceptados | 1 |
| Revisión ciega al desenlace | Sí |
| Parámetros congelados | Sí |

El caso de confianza baja fue identificado por la guarda de límites, revisado explícitamente y aceptado sin cambiar manualmente la línea. La mitad izquierda de la matriz quedó confirmada como rodilla derecha del participante y la mitad derecha como rodilla izquierda. El campo DICOM de lateralidad no se utilizó como decisión automática.

## Parámetros congelados

| Parámetro | Valor |
| --- | --- |
| Versión del algoritmo | `bilateral_split_v0.2_pilot` |
| Banda horizontal de búsqueda | 0.40–0.60 del ancho |
| Guarda de límites | 0.01 del ancho |
| Banda vertical de análisis | 0.15–0.85 de la altura |
| Fracción de suavizado | 0.015 |
| Peso de intensidad | 0.70 |
| Peso de gradiente | 0.30 |
| Penalización por distancia al centro | 0.12 |
| Proporción mínima de ancho por mitad | 0.66 |
| Prominencia mínima del valle | 0.05 |
| Umbral de confianza alta | 0.75 |
| Mitad izquierda de la imagen | Rodilla derecha (`SIDE=1`) |
| Mitad derecha de la imagen | Rodilla izquierda (`SIDE=2`) |

Cualquier modificación posterior de estos valores deberá registrarse y aprobarse mediante el control de cambios metodológicos antes de aplicarse.

## Controles de seguridad preservados

- No se consultó el desenlace de progresión durante la revisión.
- No se ejecutó procesamiento masivo de las 1,916 adquisiciones.
- No se entrenó ningún modelo.
- No se generaron particiones analíticas.
- No se abrió ni creó el conjunto de prueba reservado.

## Autorización resultante

Este cierre autoriza únicamente iniciar la Fase 1, paso 4: diseñar y validar, sobre los diez casos del piloto, la localización tibiofemoral y el recorte reproducible de cada rodilla. El recorte deberá excluir texto, bordes y la regla central, conservar trazabilidad con el original, registrar confianza y permitir la misma operación durante la futura inferencia.

El procesamiento masivo, la formación de particiones, el entrenamiento y la apertura de la prueba reservada continúan bloqueados hasta sus cierres y autorizaciones correspondientes.
