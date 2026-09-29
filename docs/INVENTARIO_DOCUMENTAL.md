# Inventario documental público

## Control

| Campo | Valor |
| --- | --- |
| Estado | Activo |
| Versión | 1.0 pública |
| Fecha de corte | 28 de septiembre de 2026 |
| Ámbito | Artefactos rectores y respaldos de los pasos 1 a 5 de la Fase 0 |
| Inventario exacto | Privado en Google Drive autorizado |

Este archivo es el resumen publicable del control documental. El inventario privado conserva nombres de archivo completos, enlaces, identificadores de Drive, tamaños, huellas SHA-256 y ubicación exacta de cada versión. Esa evidencia no se replica en GitHub.

## Versiones vigentes

| Documento | Versión/estado | Ubicación en Git | Respaldo privado |
| --- | --- | --- | --- |
| Contrato de alcance y trazabilidad | 1.1, aprobado | `docs/00_ALCANCE_Y_TRAZABILIDAD.md` | Verificado |
| Reglas metodológicas invariables | 1.0, aprobado | `docs/01_REGLAS_INVARIABLES.md` | Verificado |
| Contrato ejecutable del alcance | 1.0, aprobado | `configs/governance/scope_contract.json` | Verificado |
| Estructura e inventario público de fuentes | 1.1 pública, ejecutado | `docs/02_ESTRUCTURA_DRIVE_Y_FUENTES.md` | Inventario exacto preservado |
| Registro público resumido de fuentes | 1.1 pública, verificado | `configs/governance/source_registry.json` | Registro exacto preservado |
| Control de cambios metodológicos | 1.0, aprobado | `docs/03_CONTROL_CAMBIOS_METODOLOGICOS.md` | Verificado |
| Plantilla de solicitud MCR | 1.0, aprobada | `docs/templates/SOLICITUD_CAMBIO_METODOLOGICO.md` | Verificado |
| Registro de cambios MCR | 1.0, aprobado | `configs/governance/methodology_change_log.json` | Verificado |
| Arquitectura reproducible | 1.1, aprobada | `docs/04_ARQUITECTURA_REPRODUCIBLE.md` | Verificado |
| Notebook de cierre técnico en Colab | 1.1, aprobado para ejecución; ruta privada mediante secreto | `notebooks/00_arranque_colab.ipynb` | Verificado |

## Historial preservado

Drive conserva, sin sobrescritura, las revisiones previas del contrato de alcance, reglas invariables, contrato ejecutable, control de cambios, plantilla MCR, registro MCR, README, plan de implementación, guía de Fase 0 y registro de decisiones. La relación exacta de versiones, ubicaciones y huellas se mantiene en el inventario privado.

## Instantáneas del repositorio

Se verificaron copias privadas del README principal, README del prototipo, plan de implementación, guía de Fase 0, decisiones, arquitectura y notebook. Las revisiones superadas permanecen en el historial de Drive.

## Evidencia de verificación

- los artefactos vigentes y las versiones históricas están en sus áreas privadas esperadas;
- los metadatos de Drive indican que los archivos verificados no están compartidos públicamente;
- los doce archivos fuente únicos fueron controlados mediante el inventario privado;
- las pruebas locales incluyen contratos de gobernanza y del notebook de Colab;
- los pasos 1 a 5 están aprobados y versionados;
- la ejecución técnica equivalente en Colab/Drive del paso 6 continúa pendiente;
- no se ejecutó entrenamiento ni se abrió el conjunto de prueba reservado.

El inventario público se ampliará solo con información necesaria para reproducibilidad y gobierno. Credenciales, identificadores individuales, rutas privadas, huellas de fuentes restringidas y enlaces privados nunca se publicarán.
