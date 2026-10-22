"""Build notebook35: one explicit head-only two-update GPU smoke on development."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path


NOTEBOOK = "notebooks/35_prueba_breve_entrenamiento_gpu_desarrollo.ipynb"
UNPUBLISHED = "PENDING_PUBLISHED_COMMIT"


def _previous_builder():
    spec = importlib.util.spec_from_file_location("knee_notebook35_bootstrap_source",
        Path(__file__).resolve().with_name("build_final_models_notebook.py"))
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
    bootstrap = previous["cells"][1]["source"].replace("knee_final_models33_", "knee_training_smoke35_")
    old_guard = """if os.environ.get('COLAB_GPU', '0') == '1' or any(os.environ.get(k) for k in ('COLAB_TPU_ADDR', 'TPU_NAME', 'COLAB_TPU_TASK_ID')):
    raise RuntimeError('Selecciona CPU en Entorno de ejecución > Cambiar tipo de entorno de ejecución.')"""
    new_guard = """if os.environ.get('COLAB_GPU', '0') != '1' or any(os.environ.get(k) for k in ('COLAB_TPU_ADDR', 'TPU_NAME', 'COLAB_TPU_TASK_ID')):
    raise RuntimeError('Selecciona GPU en Entorno de ejecución > Cambiar tipo de entorno de ejecución. No uses CPU ni TPU.')"""
    if old_guard not in bootstrap:
        raise ValueError("previous_bounded_bootstrap_gpu_guard_changed")
    bootstrap = bootstrap.replace(old_guard, new_guard)
    bootstrap = bootstrap.replace("from knee import final_models_colab as orchestration",
        "from knee import training_smoke_colab as orchestration")
    bootstrap = bootstrap.replace("La comprobación se detuvo. No ejecutes entrenamiento.",
        "La prueba breve se detuvo. Conserva el error; no intentes entrenamiento completo.")
    cells = [markdown("""# 35 · Prueba breve de entrenamiento en GPU, solo desarrollo

**Revisión 1.0.** Selecciona **GPU** y ejecuta las celdas en orden. Esta prueba sí actualiza pesos, pero su alcance es únicamente **DenseNet121, cabeza de clasificación, pliegue 0, semilla 2026 y dos actualizaciones del optimizador**. El codificador preentrenado permanece congelado. No se inicia el entrenamiento completo, no se comparan modelos y no se abre la prueba reservada.

Se comprobarán las referencias de las libretas 32 y 34, se reconstruirá únicamente el paquete de desarrollo ya aprobado y se descargará solo el checkpoint público de DenseNet121 en el almacenamiento temporal de Colab. No se necesitan imágenes nuevas ni volver a ejecutar las libretas anteriores. Se conservan la geometría de 224 × 224 y las particiones por participante.

Necesitarás dos rutas de Drive: la carpeta de registros privados usada en la 34 y la misma carpeta privada de desarrollo usada en las libretas 31–32, que contiene `transport.json` y `model_inputs_loading_reference_v1.json`. No selecciones una carpeta de prueba reservada.

Al final se guardarán los registros y los pesos de esta prueba en una **subcarpeta privada nueva**. Son artefactos de compatibilidad; no constituyen un modelo final ni evidencia válida para seleccionar modelos. La prueba no autoriza otros pliegues, descongelar el codificador o entrenar otros modelos."""), code(bootstrap),
        markdown("""## 1. Preparar el intérprete GPU aislado

Se instala la versión fijada de las bibliotecas CUDA dentro de un intérprete temporal independiente. No se altera el intérprete principal de Colab. La compatibilidad real de la GPU se verifica antes de acceder a los datos; si falla, la libreta se detiene sin sustituir la GPU por CPU."""),
        code("""print('Preparando el intérprete GPU aislado; aún no se abrieron datos ni se entrenó…')
runtime = json.loads(checked_run([sys.executable, '-B', str(REPO_DIR / 'scripts/check_training_smoke_colab.py'), '--operation', 'setup', '--repository-root', str(REPO_DIR), '--destination', str(SESSION_ROOT / 'gpu_runtime')]))
NEURAL_PYTHON = Path(runtime['python_path'])
RUN_ENV = orchestration.runtime_environment(REPO_DIR)
print('Intérprete y GPU verificados. Todavía no se ejecutó la prueba con datos.')
"""), markdown("""## 2. Ejecutar las pruebas sintéticas del código

La suite usa datos sintéticos y sustituciones de las operaciones de entrenamiento; no actualiza pesos reales. En este subproceso se oculta la GPU. Se verifica el código vigente compatible con las bibliotecas CUDA; los dos módulos específicos del intérprete CPU histórico (verificación de pesos CPU y preparación 34) no se repiten aquí: el resultado anterior de 313 pruebas se autentica mediante los dos JSON guardados. No se presentan como pruebas omitidas de esta suite ni como pruebas repetidas. Cualquier error u omisión de la suite actual detiene el proceso antes de abrir el paquete de desarrollo."""),
        code("""TEST_ENV = dict(RUN_ENV)
TEST_ENV.update(CUDA_VISIBLE_DEVICES='', COLAB_GPU='0')
CPU_TESTS = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_training_smoke_colab.py'), '--operation', 'tests', '--repository-root', str(REPO_DIR)], cwd=REPO_DIR, env=TEST_ENV))
print(f"Pruebas sintéticas aprobadas: {CPU_TESTS['tests_run']}; omitidas: {CPU_TESTS['skipped']}.")
"""), markdown("""## 3. Localizar la preparación 34 y reconstruir solo desarrollo

Indica primero la misma carpeta de registros privados de la libreta 34. Se selecciona su comprobación completa más reciente mediante una búsqueda limitada a JSON de subcarpetas directas. Después indica la carpeta de transporte de desarrollo usada en la 32. Se verifican todas las huellas al copiar; los archivos de Drive no se modifican."""),
        code("""from google.colab import drive
drive.mount('/content/drive')
PERSISTENT_ROOT = orchestration.checked_path(input('Carpeta existente de KneeOAI con los registros de la libreta 34: ').strip())
REMOTE_PACKAGE_DIR = orchestration.checked_path(input('Misma carpeta privada de desarrollo de la libreta 32, que contiene transport.json: ').strip())
for selected in (PERSISTENT_ROOT, REMOTE_PACKAGE_DIR):
    if not selected.is_dir() or not selected.is_relative_to(Path('/content/drive')):
        raise RuntimeError('Selecciona carpetas existentes dentro de tu Drive montado.')
preparation = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_training_smoke_colab.py'), '--operation', 'locate', '--repository-root', str(REPO_DIR), '--records-root', str(PERSISTENT_ROOT)], cwd=REPO_DIR, env=RUN_ENV))
PREPARATION_DIR = Path(preparation['preparation_dir'])
LOCAL_PACKAGE_DIR = SESSION_ROOT / 'development'
REFERENCE_DIR = SESSION_ROOT / 'reference'
REFERENCE_DIR.mkdir(exist_ok=False)
LOCAL_REFERENCE = REFERENCE_DIR / 'model_inputs_loading_reference_v1.json'
transport = orchestration.stage_development(REMOTE_PACKAGE_DIR, LOCAL_PACKAGE_DIR, LOCAL_REFERENCE)
print('Preparación 34 localizada. Solo desarrollo se reconstruyó y verificó; los originales siguen sin cambios.')
"""), markdown("""## 4. Ejecutar únicamente las dos actualizaciones autorizadas

**Esta celda sí realiza entrenamiento breve.** Escribe `AUTORIZO PRUEBA GPU` para confirmar el alcance mostrado. Solo se actualiza la cabeza de DenseNet121 durante dos pasos; no se reutilizan estos pesos para seleccionar un modelo ni se inicia el entrenamiento completo. Se mide la memoria GPU y se comprueba que el codificador permanezca congelado. La validación usa exclusivamente el pliegue de desarrollo correspondiente.

Si aparece un error, no cambies el modelo, el pliegue o el número de pasos por tu cuenta. Conserva el mensaje para revisarlo."""),
        code("""confirmation = input('Escribe AUTORIZO PRUEBA GPU para ejecutar solo la prueba descrita: ').strip()
if confirmation != 'AUTORIZO PRUEBA GPU':
    raise RuntimeError('No se confirmó la prueba breve. No se inició entrenamiento.')
import uuid
WEIGHTS_ROOT = SESSION_ROOT / 'public_weight_cache'
WEIGHTS_ROOT.mkdir(exist_ok=False)
LOCAL_OUTPUT = SESSION_ROOT / ('gpu_smoke_' + uuid.uuid4().hex)
print('Ejecutando solo DenseNet121, codificador congelado y cabeza entrenable, pliegue 0 y dos actualizaciones…')
smoke_log = checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_training_smoke_colab.py'), '--operation', 'smoke', '--repository-root', str(REPO_DIR), '--preparation-dir', str(PREPARATION_DIR), '--package-dir', str(LOCAL_PACKAGE_DIR), '--reference-path', str(LOCAL_REFERENCE), '--weights-root', str(WEIGHTS_ROOT), '--output-dir', str(LOCAL_OUTPUT), '--approval-statement', orchestration.APPROVAL_STATEMENT], cwd=REPO_DIR, env=RUN_ENV)
result = json.loads((LOCAL_OUTPUT / 'training_smoke.json').read_text(encoding='utf-8'))
orchestration.validate_result(result)
print('Prueba breve finalizada. El entrenamiento completo sigue pendiente de autorización.')
"""), markdown("""## 5. Guardar y verificar la prueba en Drive

Se copia la carpeta privada de la prueba a una subcarpeta nueva junto a los registros anteriores. Se verifican las huellas de cada archivo después de escribirlo. No se sobrescriben datos, resultados anteriores o paquetes de participantes. Los pesos de esta prueba no se publican en GitHub."""),
        code("""FINAL_CHECK = orchestration.publish_result(LOCAL_OUTPUT, PERSISTENT_ROOT, CODE_REVISION, CPU_TESTS, runtime)
print(json.dumps(FINAL_CHECK, ensure_ascii=False, indent=2))
print('Prueba breve guardada y verificada. No se abrió la prueba reservada ni se ejecutó entrenamiento completo.')
"""), markdown("""Comparte únicamente el **JSON final**, no los pesos ni las imágenes. Que esta prueba pase acredita ejecución técnica y recursos GPU en este alcance; no demuestra desempeño predictivo ni autoriza el entrenamiento completo.""")]
    metadata = {**previous["metadata"], "accelerator": "GPU"}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


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
