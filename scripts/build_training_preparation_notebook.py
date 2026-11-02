"""Build the thin CPU-only notebook34 from an exact published code revision."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path


NOTEBOOK = "notebooks/34_preparar_ejecutor_entrenamiento_sin_entrenar.ipynb"
UNPUBLISHED = "PENDING_PUBLISHED_COMMIT"


def _previous_builder():
    path = Path(__file__).resolve().with_name("build_final_models_notebook.py")
    spec = importlib.util.spec_from_file_location("knee_notebook33_bootstrap_source", path)
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
    bootstrap = previous["cells"][1]["source"].replace("knee_final_models33_", "knee_training_preparation34_")
    bootstrap = bootstrap.replace("from knee import final_models_colab as orchestration",
        "from knee import training_preparation_colab as orchestration")
    cells = [markdown("""# 34 · Preparar el ejecutor de entrenamiento, sin entrenar

**Revisión 1.0.** Selecciona **CPU** y ejecuta las celdas en orden. Esta libreta verifica el código del ejecutor y su configuración, usando pruebas sintéticas y la comprobación de modelos que ya guardaste con la libreta 33.

Al pedir la carpeta de Drive, pega **la misma carpeta de registros privados donde guardaste la comprobación 33**. La libreta busca únicamente sus archivos JSON dentro de las subcarpetas directas; selecciona el registro completo y válido más reciente. No hace falta volver a ejecutar la 33 ni buscar su subcarpeta manualmente.

No se descargarán pesos, no se abrirán paquetes de participantes y no hay ninguna celda que inicie entrenamiento. La prueba corta en GPU y el entrenamiento completo corresponden a una etapa posterior, con autorización propia. Todo el código queda en el repositorio; Drive conserva los resultados de esta comprobación en una carpeta nueva."""), code(bootstrap),
        markdown("""## 1. Preparar el intérprete CPU aislado

Se conservan las versiones verificadas de las bibliotecas y se añaden las dependencias clínicas fijadas del ejecutor. La preparación puede tardar varios minutos; no altera el intérprete principal de Colab."""),
        code("""print('Preparando el intérprete CPU aislado del ejecutor…')
runtime = json.loads(checked_run([sys.executable, '-B', str(REPO_DIR / 'scripts/check_training_preparation_colab.py'), '--operation', 'setup', '--repository-root', str(REPO_DIR), '--destination', str(SESSION_ROOT / 'cpu_runtime')]))
NEURAL_PYTHON = Path(runtime['python_path'])
RUN_ENV = orchestration.runtime_environment(REPO_DIR)
print('Entorno CPU aislado preparado. No se descargaron pesos ni se abrieron imágenes.')
"""), markdown("""## 2. Ejecutar las pruebas sintéticas del código

Se verifican las reglas de autorización, la separación por participante, las entradas del ejecutor, los modelos y el guardado de los registros con datos sintéticos. No se ejecutan pasos de entrenamiento, optimización ni cálculo de gradientes. Cualquier error o prueba omitida detiene la preparación."""),
        code("""CPU_TESTS = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_training_preparation_colab.py'), '--operation', 'tests', '--repository-root', str(REPO_DIR)], cwd=REPO_DIR, env=RUN_ENV))
print(f"Pruebas aprobadas: {CPU_TESTS['tests_run']}; omitidas: {CPU_TESTS['skipped']}.")
"""), markdown("""## 3. Localizar la comprobación aprobada de la libreta 33

Monta tu Drive y pega la carpeta privada de registros donde guardaste el resultado anterior. Solo se leerán los dos JSON de esa comprobación; no selecciones una carpeta de imágenes ni el paquete reservado."""),
        code("""from google.colab import drive
drive.mount('/content/drive')
PERSISTENT_ROOT = orchestration.checked_path(input('Carpeta existente de KneeOAI donde guardaste la comprobación de la libreta 33: ').strip())
if not PERSISTENT_ROOT.is_dir() or not PERSISTENT_ROOT.is_relative_to(Path('/content/drive')):
    raise RuntimeError('Selecciona una carpeta existente dentro del Drive montado.')
previous_record = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_training_preparation_colab.py'), '--operation', 'locate', '--repository-root', str(REPO_DIR), '--records-root', str(PERSISTENT_ROOT)], cwd=REPO_DIR, env=RUN_ENV))
SOURCE_PREFLIGHT = Path(previous_record['source_preflight_dir'])
print('Comprobación completa de la libreta 33 localizada y verificada.')
"""), markdown("""## 4. Verificar que el ejecutor queda preparado

Se comprueba el registro de los modelos, la configuración aprobada y las barreras que exigirán autorización antes de entrenar. La verificación usa únicamente los registros anteriores y entradas sintéticas. No genera un permiso para entrenar."""),
        code("""import uuid
LOCAL_OUTPUT = SESSION_ROOT / ('preparation_' + uuid.uuid4().hex)
print('Verificando la preparación del ejecutor. Sin entrenamiento…')
preparation_log = checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_training_preparation_colab.py'), '--operation', 'prepare', '--repository-root', str(REPO_DIR), '--source-preflight-dir', str(SOURCE_PREFLIGHT), '--output-dir', str(LOCAL_OUTPUT), '--execution-environment', 'google_colab_cpu'], cwd=REPO_DIR, env=RUN_ENV)
result = json.loads((LOCAL_OUTPUT / 'training_runner_preparation.json').read_text(encoding='utf-8'))
orchestration.validate_result(result, 'google_colab_cpu')
print('Ejecutor preparado; prueba corta en GPU y entrenamiento pendientes de autorización.')
"""), markdown("""## 5. Guardar la comprobación en la misma carpeta de Drive

Se crean dos archivos JSON en una subcarpeta nueva junto a los registros anteriores. No se reemplaza ningún resultado previo."""),
        code("""FINAL_CHECK = orchestration.publish_result(LOCAL_OUTPUT, PERSISTENT_ROOT, CODE_REVISION, CPU_TESTS, runtime)
print(json.dumps(FINAL_CHECK, ensure_ascii=False, indent=2))
print('Preparación guardada. No se entrenó ni se abrió la prueba reservada.')
"""), markdown("""Comparte únicamente el **JSON final**. No necesitas enviar imágenes ni paquetes privados. Que esta preparación pase no significa que ya se haya ejecutado la prueba corta en GPU o autorizado el entrenamiento.""")]
    return {"cells": cells, "metadata": previous["metadata"], "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--allow-unpublished-placeholder", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / NOTEBOOK)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.code_revision,
        allow_unpublished_placeholder=args.allow_unpublished_placeholder), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
