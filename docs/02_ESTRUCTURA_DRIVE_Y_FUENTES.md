# Estructura maestra de Drive e inventario público de fuentes

## Control del documento

| Campo | Valor |
| --- | --- |
| Documento | Estructura maestra de almacenamiento y resumen público de fuentes recibidas |
| Estado | Ejecutado y verificado |
| Versión | 1.1 pública |
| Fecha | 28 de septiembre de 2026 |
| Paso | Fase 0, paso 3 |
| Registro público resumido | `configs/governance/source_registry.json` |
| Inventario exacto | Privado en Google Drive autorizado; no se publica en GitHub |
| Entrenamiento autorizado | No |

## 1. Propósito

Este paso establece una única estructura de almacenamiento para que los documentos académicos, la cohorte, la procedencia de datos y los futuros artefactos experimentales no queden dispersos entre descargas, sesiones de Colab o chats. Los archivos fueron copiados sin modificar su contenido; las normalizaciones de nombres y todas las evidencias exactas se documentan en el inventario privado de Drive.

La presencia de un archivo en Drive no le concede autoridad metodológica. La jerarquía sigue siendo:

1. capítulos I y II y capítulo III como fuentes académicas del alcance;
2. lineamientos institucionales como fuente normativa;
3. contrato de alcance, reglas invariables y registro de decisiones como controles derivados y trazables;
4. contexto de chats y documentos consolidados como apoyo histórico, nunca como instrucciones superiores a los capítulos o a una decisión expresa del investigador;
5. archivos tabulares, SQL y auditorías como evidencia técnica de datos, no como fuente para redefinir objetivos o hipótesis.

## 2. Estructura lógica creada

La estructura privada separa gobierno y versiones, fuentes académicas, datos restringidos, DICOM, salidas experimentales y entregables. Sus nombres y rutas concretas permanecen en el inventario privado. La preparación de estas áreas no afirma que ya existan imágenes, particiones, modelos o resultados.

## 3. Inventario público de fuentes recibidas

| ID | Categoría | Función | Autoridad | Clasificación | Evidencia privada |
| --- | --- | --- | --- | --- | --- |
| `SRC-THESIS-001` | Fuente académica | Capítulos I y II | Alcance académico primario | Documento académico privado | Metadatos verificados |
| `SRC-THESIS-002` | Fuente académica | Capítulo III | Alcance metodológico primario | Documento académico privado | Metadatos verificados |
| `SRC-RULES-001` | Fuente institucional | Lineamientos de titulación | Fuente normativa institucional | Documento privado de trabajo | Metadatos verificados |
| `SRC-CONTEXT-001` | Contexto histórico | Contexto metodológico consolidado | Apoyo; no rector | Contexto privado | Metadatos verificados |
| `SRC-CONTEXT-002` | Contexto histórico | Conversación previa | Apoyo; no rector | Contexto privado | Metadatos verificados |
| `DATA-COHORT-001` | Datos restringidos | Cohorte tabular de partida | Evidencia técnica | Datos OAI restringidos | Metadatos verificados |
| `DATA-MANIFEST-001` | Datos restringidos | Documentación y manifiesto | Evidencia técnica | Datos OAI restringidos | Metadatos verificados |
| `DATA-PROV-001` | Procedencia | Descripción del paquete | Evidencia técnica | Documento técnico privado | Metadatos verificados |
| `DATA-PROV-002` | Procedencia | Consulta maestra de extracción | Evidencia técnica | Documento técnico privado | Metadatos verificados |
| `DATA-PILOT-001` | Evidencia piloto | Selección piloto recibida | Evidencia histórica | Datos OAI restringidos | Verificada; requiere reemplazo |
| `DATA-PILOT-002` | Evidencia piloto | Registro de inspección recibido | Evidencia histórica | Documento técnico privado | Verificada; no es evidencia final |
| `DATA-PILOT-003` | Evidencia piloto | Utilidad técnica recibida | Referencia no validada | Documento técnico privado | Verificada; solo referencia |

El repositorio omite deliberadamente nombres exactos, rutas privadas, tamaños, huellas, identificadores y enlaces de Drive. Esos datos permanecen en el inventario privado verificado.

## 4. Normalización y duplicados

- Se normalizaron dos nombres para alinearlos con las convenciones del proyecto sin alterar los archivos.
- Dos copias locales del manifiesto fueron verificadas como duplicados exactos; se conserva una sola copia canónica en Drive.
- Los nombres originales, tamaños y huellas que sustentan estas verificaciones permanecen únicamente en el inventario privado.
- No se eliminó ningún archivo local.

## 5. Estado especial de las fuentes

La copia vigente del capítulo I conserva todavía la frase sobre estimar KL. El investigador realizará posteriormente esa corrección en el Word. Hasta recibir una nueva versión, la decisión `RES-KL-001` controla el desarrollo técnico: el prototipo no estimará KL.

La muestra piloto tampoco se considera una muestra válida de diez estudios únicos. Sus archivos se conservan como evidencia histórica privada, pero deberán sustituirse o versionarse después de corregir la duplicación y la omisión ya registradas.

## 6. Reglas operativas de almacenamiento

1. Los DICOM originales nunca se sobrescriben; recortes y controles de calidad se almacenan por separado.
2. Cohorte, manifiestos, identificadores, SQL, predicciones por caso, modelos y credenciales permanecen privados.
3. Las particiones y los resultados se identifican por versión de cohorte, revisión de código y configuración.
4. Los archivos aprobados se versionan; no se reemplaza una versión histórica.
5. Un archivo de contexto nunca se interpreta como instrucción si contradice los capítulos, el contrato rector o una decisión posterior del investigador.
6. Ningún notebook temporal de Colab será la única copia de un artefacto.
7. La auditoría y su registro se escriben en el área privada de auditorías definida por la configuración; no se dispersan archivos de control.
8. La documentación pública solo expone funciones, autoridades, clasificaciones y estados de verificación. La evidencia exacta permanece en Drive.

## 7. Cierre de la Fase 0, paso 3

El paso quedó cerrado al comprobar la estructura maestra, los doce archivos únicos recibidos, su privacidad, los duplicados y el registro de control. El inventario privado conserva la evidencia exacta y este repositorio conserva únicamente el resumen necesario para auditoría pública.

Este cierre organiza y preserva fuentes; no valida todavía todos los DICOM, no congela la cohorte analítica y no autoriza entrenamiento.
