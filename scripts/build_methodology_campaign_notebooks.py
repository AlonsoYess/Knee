"""Build thin, revision-pinned Colab notebooks for methodology v1.2 campaigns."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path


NOTEBOOKS = {
    "clinical": "notebooks/39_entrenar_clinicos_tres_escenarios_v1_2.ipynb",
    "densenet_frozen": "notebooks/40_entrenar_densenet121_aumento_v1_2.ipynb",
}


def _builder(name):
    path = Path(__file__).resolve().with_name(name)
    spec = importlib.util.spec_from_file_location("bounded_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source):
    ast.parse(source)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def _progress_function(revision, allow_placeholder):
    previous = _builder("build_clinical_fivefold_notebook.py").build(
        revision, allow_unpublished_placeholder=allow_placeholder)
    for cell in previous["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse(cell["source"]).body:
            if isinstance(node, ast.FunctionDef) and node.name == "training_run":
                return ast.get_source_segment(cell["source"], node)
    raise ValueError("verified_streaming_progress_function_not_found")


def build(code_revision, campaign, *, allow_unpublished_placeholder=False):
    if campaign not in NOTEBOOKS:
        raise ValueError("registered_methodology_campaign_required")
    clinical = campaign == "clinical"
    number, resource = (39, "CPU") if clinical else (40, "GPU")
    old_builder = "build_training_preparation_notebook.py" if clinical else "build_training_smoke_notebook.py"
    old_module = "training_preparation_colab" if clinical else "training_smoke_colab"
    old_prefix = "knee_training_preparation34_" if clinical else "knee_training_smoke35_"
    previous = _builder(old_builder).build(code_revision,
        allow_unpublished_placeholder=allow_unpublished_placeholder)
    bootstrap = previous["cells"][1]["source"]
    expected_import = "from knee import " + old_module + " as orchestration"
    if expected_import not in bootstrap or old_prefix not in bootstrap:
        raise ValueError("bounded_campaign_bootstrap_source_changed")
    bootstrap = bootstrap.replace(expected_import, "from knee import methodology_campaign_colab as orchestration")
    bootstrap = bootstrap.replace(old_prefix, f"knee_methodology{number}_")
    bootstrap += f"\nCAMPAIGN = {campaign!r}\n"
    title = ("39 · Modelos clínicos: tres escenarios y cinco pliegues" if clinical else
        "40 · DenseNet121 congelado con aumento de intensidad")
    scope = ("Se entrenarán regresión logística, XGBoost y MLP en los tres escenarios clínicos del Word: "
        "**edad, sexo e IMC**; esos datos más **KL inicial**; y esos datos más **cirugía previa y WOMAC**. "
        "Son **45 trabajos**: tres algoritmos × tres escenarios × cinco pliegues, con semilla 2026. "
        "La imputación y el escalado se ajustarán dentro del entrenamiento de cada pliegue."
        if clinical else
        "Se entrenará únicamente **DenseNet121 con el codificador congelado y la cabeza entrenable**, "
        "en los mismos **cinco pliegues** y con semilla 2026. Esta campaña aplica el **aumento leve de intensidad "
        "gamma del plan v1.2**, solo al entrenamiento y antes de la normalización del codificador. "
        "La validación conserva su transformación determinista. Cada pliegue comienza desde los pesos externos verificados.")
    history = ("Los trabajos de la libreta 37 pertenecen al escenario histórico de tres variables y no se "
        "reutilizan como resultados de esta campaña ampliada." if clinical else
        "Los resultados de la libreta 36 se conservan como campaña histórica sin este aumento. "
        "No se reutilizan sus pesos o predicciones como si pertenecieran a la receta v1.2.")
    prerequisite = ("Se necesita la comprobación **38 terminada y guardada en Colab**, incluidas sus tres vistas "
        "de aumento. Revisa esas vistas antes de confirmar. La libreta localiza sus registros automáticamente. "
        "Un ensayo local o una comprobación sin guardar no satisfacen este requisito.")
    if not clinical:
        prerequisite += " También se autentica la prueba GPU 35 ya completada."
    stages = [
        _markdown(f"# {title}\n\n**Revisión 1.0 · Plan metodológico v1.2. Selecciona {resource}.**\n\n"
            + scope + "\n\n" + prerequisite + "\n\n" + history + "\n\n"
            "Esta libreta queda preparada para una ejecución posterior por el investigador. "
            "Las primeras celdas preparan el entorno y comprueban sus entradas. **El entrenamiento solo comienza "
            "al escribir la confirmación específica en la celda indicada.**\n\n"
            "Se utilizan exclusivamente los datos de desarrollo y sus particiones selladas. "
            "Cada trabajo completo se guarda y verifica en la misma carpeta privada de Drive antes de continuar. "
            "Una reconexión recupera los trabajos completos compatibles; un trabajo interrumpido se reinicia. "
            "Ejecuta una sola sesión por campaña. Los originales se conservan intactos y la prueba reservada permanece cerrada. "
            "La calibración, el umbral y el modelo final corresponden a etapas posteriores."),
        _code(bootstrap),
        _markdown(f"## 1. Preparar el intérprete {resource} aislado\n\n"
            "Se conservan las dependencias fijadas y se verifica el recurso seleccionado. "
            "Esta celda no inicia entrenamiento."),
        _code("""print('Preparando el intérprete aislado; todavía no se inició entrenamiento…')
runtime = json.loads(checked_run([sys.executable, '-B', str(REPO_DIR / 'scripts/train_methodology_campaign_colab.py'), '--operation', 'setup', '--campaign', CAMPAIGN, '--repository-root', str(REPO_DIR), '--destination', str(SESSION_ROOT / 'isolated_runtime')]))
NEURAL_PYTHON = Path(runtime['python_path'])
RUN_ENV = orchestration.runtime_environment(REPO_DIR, CAMPAIGN)
print('Intérprete preparado. Todavía no se entrenó ningún trabajo.')
"""),
        _markdown("## 2. Comprobar la implementación con datos sintéticos\n\n"
            "Se prueban los escenarios clínicos, el aumento, la autorización y la recuperación de trabajos. "
            "Estas pruebas no ajustan modelos reales. Cualquier fallo u omisión detiene la libreta."),
        _code("""TEST_ENV = dict(RUN_ENV)
TEST_ENV.update(CUDA_VISIBLE_DEVICES='', COLAB_GPU='0')
CPU_TESTS = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/train_methodology_campaign_colab.py'), '--operation', 'tests', '--campaign', CAMPAIGN, '--repository-root', str(REPO_DIR)], cwd=REPO_DIR, env=TEST_ENV))
print(f"Pruebas sintéticas aprobadas: {CPU_TESTS['tests_run']}; omitidas: {CPU_TESTS['skipped']}.")
"""),
        _markdown("## 3. Localizar la comprobación 38 y reconstruir desarrollo\n\n"
            "Indica las dos carpetas habituales: registros privados y desarrollo, donde está `transport.json`. "
            "Se comprueban los registros completos de la libreta 38 y la integridad del paquete. "
            "Si la comprobación 38 no está terminada y guardada, completa primero su última celda."),
        _code("""from google.colab import drive
drive.mount('/content/drive')
PERSISTENT_ROOT = orchestration.checked_path(input('Misma carpeta existente de KneeOAI con los registros privados: ').strip())
REMOTE_PACKAGE_DIR = orchestration.checked_path(input('Misma carpeta privada de desarrollo, que contiene transport.json: ').strip())
for selected in (PERSISTENT_ROOT, REMOTE_PACKAGE_DIR):
    if not selected.is_dir() or not selected.is_relative_to(Path('/content/drive')):
        raise RuntimeError('Selecciona carpetas existentes dentro de tu Drive montado.')
proof = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/train_methodology_campaign_colab.py'), '--operation', 'locate', '--campaign', CAMPAIGN, '--repository-root', str(REPO_DIR), '--records-root', str(PERSISTENT_ROOT)], cwd=REPO_DIR, env=RUN_ENV))
PREPARATION_DIR = Path(proof['preparation_dir'])
LOCAL_PACKAGE_DIR = SESSION_ROOT / 'development'
REFERENCE_DIR = SESSION_ROOT / 'reference'
REFERENCE_DIR.mkdir(exist_ok=False)
LOCAL_REFERENCE = REFERENCE_DIR / 'model_inputs_loading_reference_v1.json'
transport = orchestration.stage_development(REMOTE_PACKAGE_DIR, LOCAL_PACKAGE_DIR, LOCAL_REFERENCE)
print('Comprobación 38 localizada. Paquete de desarrollo reconstruido y verificado.')
"""),
        _markdown("## 4. Confirmar la campaña y ejecutar sus trabajos\n\n"
            + scope + "\n\n"
            "**Revisa el alcance y las tres vistas del aumento de la libreta 38 antes de confirmar.** "
            "La confirmación cubre solo esta campaña. Se mostrará el progreso real por trabajo y una señal "
            "periódica de actividad. Cada trabajo completo se copia a Drive y se verifica antes del siguiente. "
            "Si hay una interrupción, conserva el mensaje y vuelve a ejecutar esta misma revisión con las mismas carpetas."),
    ]
    run = """confirmation = input('Escribe ' + orchestration.CONFIRMATIONS[CAMPAIGN] + ' para ejecutar solo esta campaña: ').strip()
if confirmation != orchestration.CONFIRMATIONS[CAMPAIGN]:
    raise RuntimeError('No se confirmó la campaña. No se inició entrenamiento.')
import queue, threading, time, uuid

__PROGRESS_FUNCTION__

LOCAL_OUTPUT = SESSION_ROOT / ('methodology_campaign_' + uuid.uuid4().hex)
TEST_RECEIPT = SESSION_ROOT / 'synthetic_test_receipt.json'
RUNTIME_RECEIPT = SESSION_ROOT / 'setup_receipt.json'
orchestration._save(TEST_RECEIPT, CPU_TESTS)
orchestration._save(RUNTIME_RECEIPT, runtime)
arguments = [str(NEURAL_PYTHON), '-u', '-B', str(REPO_DIR / 'scripts/train_methodology_campaign_colab.py'), '--operation', 'train', '--campaign', CAMPAIGN, '--repository-root', str(REPO_DIR), '--preparation-dir', str(PREPARATION_DIR), '--package-dir', str(LOCAL_PACKAGE_DIR), '--reference-path', str(LOCAL_REFERENCE), '--output-dir', str(LOCAL_OUTPUT), '--records-root', str(PERSISTENT_ROOT), '--code-revision', CODE_REVISION, '--test-receipt', str(TEST_RECEIPT), '--runtime-receipt', str(RUNTIME_RECEIPT), '--approval-statement', orchestration.APPROVAL_STATEMENTS[CAMPAIGN]]
if CAMPAIGN == 'densenet_frozen':
    WEIGHTS_ROOT = SESSION_ROOT / 'verified_public_weights'
    arguments.extend(['--weights-root', str(WEIGHTS_ROOT)])
print('Iniciando únicamente la campaña confirmada del plan v1.2. Cada trabajo completo se guardará en Drive…', flush=True)
training_log = training_run(arguments, cwd=REPO_DIR, env=RUN_ENV)
FINAL_CHECK = json.loads((LOCAL_OUTPUT / orchestration.SUMMARY_FILENAME).read_text(encoding='utf-8'))
orchestration.validate_result(FINAL_CHECK)
print('Campaña terminada. Resultados privados guardados y verificados en Drive.')
""".replace("__PROGRESS_FUNCTION__", _progress_function(code_revision, allow_unpublished_placeholder))
    stages += [_code(run),
        _markdown("## 5. Mostrar el resultado\n\n"
            "La selección utiliza la media no ponderada de AP de los cinco pliegues de cada candidato. "
            "La AP al reunir las predicciones fuera de pliegue se informa de forma descriptiva. "
            "Son resultados de desarrollo; la evaluación final se realizará posteriormente.\n\n"
            "Comparte únicamente el **JSON final**. Las predicciones individuales y los modelos permanecen privados."),
        _code("""print(json.dumps(FINAL_CHECK, ensure_ascii=False, indent=2))
print('Resultados de desarrollo guardados. No se abrió la prueba reservada ni se ajustó la calibración final.')
""")]
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if not clinical:
        metadata["accelerator"] = "GPU"
    return {"cells": stages, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--campaign", required=True, choices=tuple(NOTEBOOKS))
    parser.add_argument("--allow-unpublished-placeholder", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or Path(__file__).resolve().parents[1] / NOTEBOOKS[args.campaign]
    value = build(args.code_revision, args.campaign,
        allow_unpublished_placeholder=args.allow_unpublished_placeholder)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
