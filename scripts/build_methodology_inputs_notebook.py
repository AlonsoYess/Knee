"""Build notebook38 for development baseline and augmentation checks on CPU."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path


NOTEBOOK = "notebooks/38_comprobar_complementos_clinicos_y_aumento_sin_entrenar.ipynb"
UNPUBLISHED = "PENDING_PUBLISHED_COMMIT"


def _previous_builder():
    spec = importlib.util.spec_from_file_location("knee_notebook38_cpu_bootstrap_source",
        Path(__file__).resolve().with_name("build_training_preparation_notebook.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    ast.parse(source)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def build(code_revision, *, allow_unpublished_placeholder=False):
    previous = _previous_builder().build(code_revision,
        allow_unpublished_placeholder=allow_unpublished_placeholder)
    bootstrap = previous["cells"][1]["source"].replace("knee_training_preparation34_", "knee_methodology38_")
    bootstrap = bootstrap.replace("from knee import training_preparation_colab as orchestration",
        "from knee import methodology_inputs_colab as orchestration")
    cells = [markdown("""# 38 · Comprobar variables clínicas complementarias y aumento de datos

**Revisión 1.0. Selecciona CPU** y ejecuta las celdas en orden en una sesión nueva. Se comprobarán **KL inicial, cirugía previa y WOMAC total** de las mismas rodillas de desarrollo y la aplicación del aumento leve de intensidad previsto en el plan.

La revisión conserva los tres escenarios clínicos: **edad, sexo e IMC**; esos tres datos más **KL inicial**; y esos tres datos más **cirugía previa y WOMAC**. La imputación y el escalado se calcularán únicamente con los participantes de entrenamiento de cada uno de los **cinco pliegues**. Preparar estas transformaciones no ajusta un modelo predictivo.

Necesitarás las mismas dos carpetas de Drive: la de **registros privados** y la de **desarrollo que contiene `transport.json` y la referencia de la libreta 32**. Las entradas basales adicionales ya están preparadas en esa carpeta de desarrollo; la libreta las localiza y comprueba automáticamente.

Se verifican las imágenes ya preparadas y unas vistas comparativas del aumento. **No se entrena ningún modelo, no se descargan pesos y no se abre la prueba reservada.** Los originales y las particiones se conservan. La comprobación usa CPU; deja GPU y TPU desactivadas."""), code(bootstrap),
        markdown("""## 1. Preparar el intérprete CPU aislado

Se instalan las mismas versiones fijadas del entorno de comprobación en un intérprete temporal separado del intérprete principal de Colab. Esta preparación puede tardar varios minutos."""),
        code("""print('Preparando el intérprete CPU aislado para la comprobación, sin entrenamiento…')
runtime = json.loads(checked_run([sys.executable, '-B', str(REPO_DIR / 'scripts/check_methodology_inputs_colab.py'), '--operation', 'setup', '--repository-root', str(REPO_DIR), '--destination', str(SESSION_ROOT / 'cpu_runtime')]))
NEURAL_PYTHON = Path(runtime['python_path'])
RUN_ENV = orchestration.runtime_environment(REPO_DIR)
print('Intérprete CPU preparado. No se cargaron pesos ni se entrenaron modelos.')
"""), markdown("""## 2. Verificar el código con datos sintéticos

Se comprueban la correspondencia de las variables, la separación por participante, las transformaciones por pliegue y las restricciones del aumento de datos. Cualquier fallo o prueba omitida detiene la libreta."""),
        code("""CPU_TESTS = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_methodology_inputs_colab.py'), '--operation', 'tests', '--repository-root', str(REPO_DIR)], cwd=REPO_DIR, env=RUN_ENV))
print(f"Pruebas sintéticas aprobadas: {CPU_TESTS['tests_run']}; omitidas: {CPU_TESTS['skipped']}.")
"""), markdown("""## 3. Localizar las mismas carpetas de Drive y verificar las entradas

Pega la carpeta habitual de registros privados y la carpeta de desarrollo que contiene `transport.json`. La libreta reconstruye únicamente el paquete de desarrollo, conserva la referencia de la libreta 32 y comprueba los archivos basales adicionales antes de utilizarlos. No es necesario volver a ejecutar las libretas anteriores."""),
        code("""from google.colab import drive
drive.mount('/content/drive')
PERSISTENT_ROOT = orchestration.checked_path(input('Misma carpeta existente de KneeOAI con los registros privados: ').strip())
REMOTE_PACKAGE_DIR = orchestration.checked_path(input('Misma carpeta privada de desarrollo, que contiene transport.json: ').strip())
for selected in (PERSISTENT_ROOT, REMOTE_PACKAGE_DIR):
    if not selected.is_dir() or not selected.is_relative_to(Path('/content/drive')):
        raise RuntimeError('Selecciona carpetas existentes dentro de tu Drive montado.')
LOCAL_PACKAGE_DIR = SESSION_ROOT / 'development'
REFERENCE_DIR = SESSION_ROOT / 'reference'
REFERENCE_DIR.mkdir(exist_ok=False)
LOCAL_REFERENCE = REFERENCE_DIR / 'model_inputs_loading_reference_v1.json'
transport = orchestration.stage_development(REMOTE_PACKAGE_DIR, LOCAL_PACKAGE_DIR, LOCAL_REFERENCE)
LOCAL_BASELINE_DIR = SESSION_ROOT / 'baseline'
baseline = orchestration.stage_baseline(REMOTE_PACKAGE_DIR, LOCAL_BASELINE_DIR)
LOCAL_PROJECTION = Path(baseline['projection_path'])
LOCAL_DESCRIPTOR = Path(baseline['descriptor_path'])
print('Paquete de desarrollo, referencia y variables basales adicionales verificados.')
"""), markdown("""## 4. Comprobar las variables clínicas y el aumento de intensidad

Se verifican las entradas de los tres escenarios en cada pliegue. Los valores ausentes se tratan con estadísticas calculadas solo dentro de su entrenamiento. Se revisa que el aumento cambie únicamente la intensidad de copias de entrenamiento, conserve geometría, lateralidad y relleno, y sea reproducible. Validación conserva su transformación determinista.

Esta comprobación no calcula el rendimiento predictivo ni demuestra que el aumento vaya a mejorar un modelo. No inicia ningún entrenamiento."""),
        code("""import uuid
LOCAL_OUTPUT = SESSION_ROOT / ('methodology_check_' + uuid.uuid4().hex)
print('Comprobando las variables basales de desarrollo, los cinco pliegues y el aumento de intensidad. Solo CPU; sin entrenamiento…')
verification_log = checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_methodology_inputs_colab.py'), '--operation', 'verify', '--repository-root', str(REPO_DIR), '--package-dir', str(LOCAL_PACKAGE_DIR), '--reference-path', str(LOCAL_REFERENCE), '--projection-path', str(LOCAL_PROJECTION), '--descriptor-path', str(LOCAL_DESCRIPTOR), '--output-dir', str(LOCAL_OUTPUT), '--execution-environment', 'google_colab_cpu'], cwd=REPO_DIR, env=RUN_ENV)
result = json.loads((LOCAL_OUTPUT / orchestration.CORE_FILENAME).read_text(encoding='utf-8'))
orchestration.validate_result(result, 'google_colab_cpu')
print('Comprobación terminada. Se prepararon y verificaron las entradas; no se entrenaron modelos.')
"""), markdown("""## 5. Revisar las vistas del aumento

Cada vista muestra una imagen de desarrollo antes y después de aplicar el aumento de intensidad. Sirven para revisar que el cambio es leve y conserva el encuadre; no muestran resultados predictivos. Estas vistas y los archivos por rodilla permanecen en el almacenamiento privado."""),
        code("""from IPython.display import display, Image
for relative in orchestration.PREVIEW_FILES:
    preview = orchestration.checked_path(LOCAL_OUTPUT / relative)
    if not preview.is_file() or not preview.is_relative_to(LOCAL_OUTPUT):
        raise RuntimeError('Falta una vista verificada del aumento de intensidad.')
    display(Image(filename=str(preview)))
"""), markdown("""## 6. Guardar y verificar el resultado en la misma carpeta de Drive

Se guarda la comprobación completa en una subcarpeta privada nueva, junto a los registros anteriores. El guardado incluye comprobación de integridad y lectura de vuelta. No se reemplazan resultados previos ni se concede autorización para entrenar."""),
        code("""TEST_RECEIPT = SESSION_ROOT / 'synthetic_test_receipt.json'
RUNTIME_RECEIPT = SESSION_ROOT / 'cpu_setup_receipt.json'
orchestration._save(TEST_RECEIPT, CPU_TESTS)
orchestration._save(RUNTIME_RECEIPT, runtime)
FINAL_CHECK = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_methodology_inputs_colab.py'), '--operation', 'save', '--repository-root', str(REPO_DIR), '--output-dir', str(LOCAL_OUTPUT), '--records-root', str(PERSISTENT_ROOT), '--code-revision', CODE_REVISION, '--test-receipt', str(TEST_RECEIPT), '--runtime-receipt', str(RUNTIME_RECEIPT)], cwd=REPO_DIR, env=RUN_ENV))
orchestration.validate_summary(FINAL_CHECK)
print(json.dumps(FINAL_CHECK, ensure_ascii=False, indent=2))
print('Comprobación guardada y verificada. No se entrenó ni se abrió la prueba reservada.')
"""), markdown("""Comparte únicamente el **JSON final**. No necesitas enviar las imágenes ni los archivos privados. Las vistas quedan disponibles para tu revisión; el resultado de esta libreta acredita preparación y comprobaciones de entradas, no entrenamiento ni rendimiento predictivo.""")]
    metadata = dict(previous["metadata"])
    metadata.pop("accelerator", None)
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--allow-unpublished-placeholder", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / NOTEBOOK)
    args = parser.parse_args()
    value = build(args.code_revision, allow_unpublished_placeholder=args.allow_unpublished_placeholder)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
