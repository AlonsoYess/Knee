# Prototipo

El servicio FastAPI y la interfaz se implementarán en la fase 7. Importarán el paquete de inferencia y el preprocesamiento compartido desde `src/knee`, una vez congelados los modelos en desarrollo; no duplicarán transformaciones dentro de la API o de la interfaz. Las pruebas funcionales exigirán la misma salida que el pipeline experimental para entradas idénticas.

El prototipo recibirá DICOM basal, lateralidad, edad, sexo e IMC y mostrará la probabilidad calibrada de progresión estructural a 48 meses y su clasificación mediante el umbral congelado. Conforme a `RES-KL-001`, no estimará ni mostrará un grado Kellgren–Lawrence. Los pesos, radiografías, predicciones individuales y demás artefactos privados no se incluirán en este repositorio.
