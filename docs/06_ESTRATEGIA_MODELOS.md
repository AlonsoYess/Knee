# Estrategia de modelos aprobada para desarrollo

**Versión:** 1.0  
**Fecha:** 28 de septiembre de 2026  
**Decisión rectora:** `MCR-2026-002` / `RES-MOD-001`

## 1. Propósito y límites

Esta estrategia amplía las arquitecturas candidatas sin cambiar la pregunta de investigación. El sistema continúa siendo longitudinal y multimodal: utiliza la radiografía tibiofemoral V00 y edad, sexo e IMC de V00 para predecir `PROGRESION_48M`, construida en V06. La unidad de análisis sigue siendo la rodilla y la separación se realiza por participante.

No se agregan MRI, variables futuras, generación de radiografías ni estimación automática del grado KL. El presupuesto de Colab permite ejecutar candidatos exigentes, pero no justifica una búsqueda ilimitada: el registro cerrado y la selección fuera de pliegue protegen frente al sobreajuste y la selección oportunista.

## 2. Evidencia que sustenta la ampliación

- Tiulpin et al. demostraron la viabilidad de predecir progresión combinando radiografía simple y datos clínicos, con código de investigación disponible: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6934728/> y <https://github.com/imedslab/OAProgression>.
- La evaluación reciente en OAI y MOST mostró que la transferencia y la fusión multimodal deben contrastarse empíricamente porque la fusión no mejora de manera uniforme: <https://www.nature.com/articles/s41598-026-68440-7>.
- SKELEX es un modelo fundacional especializado en radiografías musculoesqueléticas con código y pesos publicados para investigación académica: <https://www.nature.com/articles/s41746-026-02826-9> y <https://github.com/skhoha/SKELEX>.
- ConvNeXt V2 ofrece una familia convolucional moderna con preentrenamiento auto-supervisado y pesos oficiales: <https://github.com/facebookresearch/ConvNeXt-V2>.
- DINOv3 ofrece representaciones visuales auto-supervisadas y variantes ViT compactas, con acceso y licencia propios que deben verificarse: <https://github.com/facebookresearch/dinov3>.
- La atención cruzada bilateral tiene evidencia específica para progresión de artrosis, pero se conservará como análisis complementario porque introduce la rodilla contralateral: <https://www.oarsijournal.com/article/S1063-4584%2823%2901014-2/fulltext>.

## 3. Comparación cerrada

### Rama clínica

Se mantienen regresión logística regularizada, XGBoost y MLP. La regresión logística no es un modelo descartable por ser sencilla: constituye la referencia interpretable adecuada para tres predictores. XGBoost evalúa relaciones no lineales y la MLP produce una representación neural compatible con la fusión.

### Rama radiográfica

DenseNet121 y ViT-B/16 son líneas base obligatorias. Se agregan ConvNeXt V2 Tiny, DINOv3 ViT-S/16 y SKELEX. El mejor codificador elegible se decide únicamente con desarrollo. La variante bilateral con atención cruzada es complementaria y solo podrá ejecutarse después de que la Fase 1 confirme pares, lateralidad y calidad.

### Fusión multimodal

La fusión intermedia continúa siendo el análisis principal. Se comparan concatenación y una variante con compuertas o FiLM, que permite que edad, sexo e IMC modulen la representación visual sin construir un transformer multimodal desproporcionado para la cohorte. La fusión tardía fija 0,5/0,5 permanece complementaria.

## 4. Selección por etapas

1. **Elegibilidad:** verificar pesos, licencia, riesgo de solapamiento de preentrenamiento, carga determinista y compatibilidad con Colab.
2. **Cribado:** ejecutar sonda lineal o codificador congelado usando solamente desarrollo.
3. **Promoción:** conservar las líneas base y promover como máximo dos candidatos modernos por AP fuera de pliegue, con ROC-AUC y Brier como desempates ya aprobados.
4. **Ajuste fino:** comparar descongelamiento gradual, decaimiento de tasa por capas y ajuste completo cuando la evidencia de desarrollo lo respalde.
5. **Estabilidad:** repetir los finalistas neuronales con semillas 2026, 2027 y 2028.
6. **Fusión:** entrenar los candidatos multimodales con pesos visuales generados sin participantes del pliegue de validación.
7. **Congelamiento:** elegir arquitectura, transformaciones, calibrador y umbral antes de abrir una sola vez la prueba reservada.

## 5. Técnicas comunes

Se utilizarán AdamW, precisión mixta, acumulación de gradiente cuando la VRAM lo exija, calentamiento y decaimiento cosenoidal, BCE ponderada calculada dentro de cada entrenamiento, parada temprana por AP y registro íntegro de recursos. Los aumentos radiográficos se definirán después del control visual de la Fase 1 y se aplicarán solo al entrenamiento.

Se conservará 224 × 224 para las líneas base comprometidas. En los candidatos promovidos se evaluará 384 × 384 como sensibilidad dentro de desarrollo. La mayor resolución no se asumirá superior: deberá demostrar ganancia estable y justificar su costo.

## 6. Condiciones pendientes antes del entrenamiento

- cerrar el piloto radiográfico con diez estudios únicos;
- verificar lectura, fotometría, lateralidad, pares y región tibiofemoral;
- congelar la cohorte común y las particiones agrupadas;
- revisar de forma documentada las licencias de pesos de ConvNeXt V2, DINOv3 y SKELEX;
- implementar y probar los cargadores sin utilizar el bloque de prueba;
- aprobar la política de aumentos después del control visual.

La aprobación de esta estrategia permite preparar configuración, código y pruebas. No autoriza todavía entrenamiento masivo ni apertura de la prueba reservada.
