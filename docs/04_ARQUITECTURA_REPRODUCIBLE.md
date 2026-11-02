# Arquitectura reproducible de ejecución y prototipo

## Control

| Campo | Valor |
| --- | --- |
| Versión | 1.1 |
| Estado | Aprobado |
| Fecha | 28 de septiembre de 2026 |
| Alcance | Frontera GitHub–Colab–Drive y continuidad hacia el prototipo |
| Aprobado por | Investigador |

## Propósito

Esta arquitectura mantiene una única implementación auditable desde la preparación de datos hasta la aplicación. Google Colab aporta capacidad de cómputo, pero no es la fuente del código. Google Drive conserva datos y artefactos privados. GitHub conserva el código, las configuraciones sin rutas privadas, las pruebas y la documentación pública autorizada.

## Responsabilidades

| Componente | Responsabilidad | Contenido excluido |
| --- | --- | --- |
| GitHub | Código versionado, pruebas, configuraciones de ejemplo, notebooks de orquestación y documentación pública. | Datos individuales, DICOM, credenciales, pesos y predicciones por caso. |
| Google Colab | Ejecutar una revisión identificable de GitHub, montar Drive y proporcionar CPU/GPU. | Lógica científica exclusiva o cambios manuales no versionados. |
| Google Drive | Fuentes privadas, DICOM, recortes, particiones, pesos, auditorías, bitácoras y entregables aprobados. | Código sin control de versión como única copia. |
| `src/knee` | Implementación compartida de auditoría, preparación, entrenamiento, evaluación e inferencia. | Interacción específica de una interfaz o una celda de notebook. |
| `notebooks` | Configurar el entorno y llamar a `src/knee` mediante comandos reproducibles. | Copias paralelas del pipeline o decisiones metodológicas ocultas. |
| `app` | API FastAPI e interfaz que consumen el paquete de inferencia congelado. | Preprocesamiento o inferencia reimplementados. |

La dirección de dependencia es `notebooks → src/knee` y `app → src/knee`. El paquete central no depende de Colab ni de la interfaz.

### Frontera de publicación

El inventario técnico exacto pertenece a Drive. GitHub solo conserva un resumen suficiente para gobernanza pública: identificadores abstractos, función, autoridad, sensibilidad y estado de verificación. Nombres exactos de fuentes privadas, rutas, tamaños, huellas, identificadores y enlaces de Drive no se publican. Esta separación no elimina evidencia: la conserva en el ámbito privado autorizado y evita convertir el repositorio público en un índice de datos restringidos.

## Cierre técnico de la Fase 0

La libreta `00_arranque_colab.ipynb` debe:

1. montar el Drive autorizado;
2. clonar en un entorno limpio la rama declarada del repositorio;
3. obtener y mostrar el commit exacto resuelto;
4. instalar el paquete desde ese clon;
5. verificar la presencia de las fuentes privadas sin imprimir identificadores ni rutas de participantes;
6. ejecutar todas las pruebas automáticas y los contratos de gobernanza;
7. ejecutar la auditoría y crear la bitácora mediante módulos de `src/knee`;
8. comprobar que auditoría y bitácora comparten las mismas huellas y que la bitácora registra el commit ejecutado.

Los resultados se guardan en el área privada de auditorías configurada en Drive. La ruta exacta no se publica. El cierre solo acredita integridad tabular, configuración reproducible y trazabilidad; no acredita disponibilidad DICOM, particiones, entrenamiento ni desempeño predictivo.

## Continuidad hacia el entrenamiento y la aplicación

Las fases posteriores ampliarán `src/knee` con módulos separados para control DICOM, recortes, particiones, entrenamiento, calibración e inferencia. Cada notebook seguirá siendo un punto de entrada delgado. La aplicación de la Fase 7 cargará el paquete de inferencia congelado y los artefactos versionados, y sus pruebas compararán su salida con la del pipeline experimental para los mismos casos.

La salida autorizada del prototipo es la probabilidad calibrada de progresión estructural a 48 meses y su clasificación mediante el umbral fijado en desarrollo. `RES-KL-001` excluye la estimación automática del grado Kellgren–Lawrence.

## Evidencia para revisión y sustentación

Cada ejecución aceptada deberá permitir identificar como mínimo:

- commit de Git ejecutado y estado limpio del clon;
- versiones de Python y dependencias críticas;
- huellas SHA-256 de las fuentes;
- configuración empleada sin rutas privadas;
- conteos agregados y estado de cada control;
- ubicación y nombre de los artefactos generados;
- resultado de las pruebas automáticas.

Ninguna celda ejecutada manualmente fuera del flujo versionado constituye evidencia suficiente de reproducibilidad.
