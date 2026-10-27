"""Build notebook 41 from the verified, thin GPU campaign orchestrator."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path


CAMPAIGN = "remaining_encoders_frozen"
NOTEBOOK_PATH = "notebooks/41_entrenar_codificadores_congelados_cinco_pliegues.ipynb"
MODEL_NAMES = ("ViT-B/16", "ConvNeXtV2-Tiny", "Skelex", "BiomedCLIP", "MambaOut-Tiny")


def _previous():
    source = Path(__file__).resolve().with_name("build_methodology_campaign_notebooks.py")
    spec = importlib.util.spec_from_file_location("verified_methodology_notebook_builder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("verified_notebook41_template_changed")
    return source.replace(old, new, 1)


def build(code_revision, *, allow_unpublished_placeholder=False):
    value = _previous().build(code_revision, "densenet_frozen",
        allow_unpublished_placeholder=allow_unpublished_placeholder)
    value["cells"][0]["source"] = (
        "# 41 · Comparación de los demás codificadores radiográficos\n\n"
        "**Revisión 1.0 · Plan metodológico v1.2 · GPU T4.**\n\n"
        "Se ejecutarán **25 trabajos**, en este orden: **ViT-B/16, ConvNeXtV2-Tiny, Skelex, "
        "BiomedCLIP y MambaOut-Tiny**; cinco pliegues por codificador, con semilla 2026. "
        "Los codificadores permanecen congelados: solo se entrena la cabeza de predicción. "
        "Cada trabajo comienza desde el checkpoint externo registrado, sin reutilizar pesos entrenados de otro pliegue.\n\n"
        "Se mantienen el conjunto de desarrollo, las particiones por participante, las entradas de 224 × 224 "
        "y el aumento gamma v1.2 únicamente en entrenamiento. La validación conserva su transformación "
        "determinista y cada codificador utiliza su normalización registrada.\n\n"
        "Se necesitan los registros completos de las libretas **33 y 38** ya guardados en Drive. "
        "Se autentican automáticamente; no tienes que volver a ejecutarlas. Revisa las tres vistas del aumento "
        "de la libreta 38 antes de confirmar. Tras la confirmación, se comprobarán los pesos y un forward "
        "sintético en GPU de cada codificador. Después se hará una **prueba breve de dos actualizaciones "
        "por codificador en el pliegue 0 de desarrollo**, para verificar gradientes y recursos con la receta "
        "aprobada. Estos cinco controles se registran aparte, no participan en la selección del modelo y "
        "sus pesos no se reutilizan en los 25 trabajos completos.\n\n"
        "Cada trabajo completo se guarda y verifica en Drive antes de continuar. Una reconexión recupera "
        "solo trabajos completos de esta misma revisión y configuración; el trabajo interrumpido se reinicia. "
        "Ejecuta una sola sesión de esta campaña.\n\n"
        "DenseNet121 ya tiene su campaña en la libreta 40 y no se vuelve a entrenar aquí. DINOv3 permanece "
        "fuera por acceso pendiente. Esta libreta no descongela codificadores, no ejecuta fusión multimodal, "
        "calibración, selección de umbral ni ajuste final. Los originales y las particiones se conservan; "
        "la prueba reservada permanece cerrada. **El entrenamiento comienza solo con la confirmación indicada.**")
    bootstrap = value["cells"][1]["source"]
    bootstrap = _replace_once(bootstrap, "knee_methodology40_", "knee_image_screening41_")
    bootstrap = _replace_once(bootstrap, "CAMPAIGN = 'densenet_frozen'", "CAMPAIGN = 'remaining_encoders_frozen'")
    value["cells"][1]["source"] = bootstrap
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = cell["source"].replace("scripts/train_methodology_campaign_colab.py",
                                                   "scripts/train_image_screening_colab.py")
    value["cells"][4]["source"] = (
        "## 2. Comprobar la implementación con datos sintéticos\n\n"
        "Se comprueban la lista y el orden de los 25 trabajos, las autorizaciones, el aumento, "
        "los archivos de mayor tamaño y la recuperación. Estas pruebas calculan en CPU con la compilación "
        "de PyTorch para GPU; la GPU queda oculta en este paso. No se entrenan datos de participantes.")
    value["cells"][6]["source"] = (
        "## 3. Localizar las comprobaciones previas y reconstruir desarrollo\n\n"
        "Indica las mismas carpetas de registros privados y desarrollo, donde está `transport.json`. "
        "Se autentican las comprobaciones 33 y 38 y la integridad del paquete de desarrollo. "
        "Esta celda no descarga ni abre el paquete reservado de prueba.")
    value["cells"][8]["source"] = (
        "## 4. Confirmar y ejecutar los 25 trabajos\n\n"
        "Orden: **ViT-B/16 → ConvNeXtV2-Tiny → Skelex → BiomedCLIP → MambaOut-Tiny**. "
        "Primero se verifican los cinco codificadores, incluida una prueba breve de dos actualizaciones por "
        "codificador en el pliegue 0 de desarrollo. Después se completan cinco pliegues por codificador, "
        "siempre con el codificador congelado y aumento gamma v1.2. Cada trabajo completo comienza desde "
        "los pesos externos registrados; los controles breves no aportan pesos ni métricas a esa comparación. "
        "El progreso identifica modelo y pliegue. Los checkpoints grandes requieren espacio local y tiempo "
        "de copia a Drive. Cada trabajo se guarda y verifica antes de continuar.\n\n"
        "La confirmación cubre únicamente esta campaña de desarrollo. Si Colab se desconecta, vuelve a abrir "
        "esta misma revisión y ejecuta con las mismas carpetas: se recuperan los trabajos completos compatibles.")
    train = value["cells"][9]["source"]
    train = _replace_once(train, "if CAMPAIGN == 'densenet_frozen':", "if CAMPAIGN == 'remaining_encoders_frozen':")
    train = _replace_once(train,
        "orchestration._save(TEST_RECEIPT, CPU_TESTS)\norchestration._save(RUNTIME_RECEIPT, runtime)",
        "for receipt_path, payload in ((TEST_RECEIPT, CPU_TESTS), (RUNTIME_RECEIPT, runtime)):\n"
        "    if receipt_path.exists():\n"
        "        if orchestration._json(receipt_path) != payload:\n"
        "            raise RuntimeError('El recibo previo pertenece a otra preparación; inicia una sesión nueva.')\n"
        "    else:\n"
        "        orchestration._save(receipt_path, payload)")
    train = _replace_once(train, "Iniciando únicamente la campaña confirmada del plan v1.2. Cada trabajo completo se guardará en Drive…",
        "Iniciando los 25 trabajos confirmados; ViT-B/16 primero. Cada trabajo completo se guardará en Drive…")
    value["cells"][9]["source"] = train
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
            if cell["outputs"] or cell["execution_count"] is not None:
                raise ValueError("clean_unexecuted_notebook41_required")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--allow-unpublished-placeholder", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = build(args.code_revision, allow_unpublished_placeholder=args.allow_unpublished_placeholder)
    target = args.output or Path(__file__).resolve().parents[1] / NOTEBOOK_PATH
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
