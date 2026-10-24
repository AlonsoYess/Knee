"""Build notebook37 for the three registered clinical models on development CPU."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path


NOTEBOOK = "notebooks/37_entrenar_modelos_clinicos_cinco_pliegues.ipynb"
UNPUBLISHED = "PENDING_PUBLISHED_COMMIT"


def _previous_builder():
    spec = importlib.util.spec_from_file_location("knee_notebook37_cpu_bootstrap_source",
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
    bootstrap = previous["cells"][1]["source"].replace("knee_training_preparation34_", "knee_clinical37_")
    bootstrap = bootstrap.replace("from knee import training_preparation_colab as orchestration",
        "from knee import clinical_fivefold_colab as orchestration")
    bootstrap = bootstrap.replace("La comprobación se detuvo. No ejecutes entrenamiento.",
        "La ejecución se detuvo. Conserva el error y los trabajos completos guardados en Drive; no cambies la receta.")
    cells = [markdown("""# 37 · Tres modelos clínicos en los cinco pliegues de desarrollo

**Revisión 1.0. Selecciona CPU** y ejecuta las celdas en orden en una sesión nueva. Esta libreta entrena los tres comparadores clínicos registrados: **regresión logística, XGBoost y MLP**. Cada modelo utiliza los mismos **cinco pliegues por participante**, la semilla **2026** y la receta fijada. Son **15 trabajos** en secuencia.

Las únicas variables clínicas son **edad, sexo e IMC de la evaluación inicial**. La imputación y las transformaciones se ajustan dentro del entrenamiento de cada pliegue. La comparación empleará las mismas rodillas y particiones utilizadas en la etapa radiográfica. No se cambian los originales ni las asignaciones.

Necesitarás las mismas dos carpetas de Drive: **registros privados**, donde está la comprobación 34, y **desarrollo**, que contiene `transport.json` y `model_inputs_loading_reference_v1.json`. La libreta localiza el registro 34 automáticamente; no es necesario repetir las libretas anteriores. Para verificar que el paquete de desarrollo es el mismo, también se comprueba la integridad de sus imágenes ya preparadas, pero los modelos de esta etapa utilizan únicamente las variables clínicas.

**No se abre la prueba reservada.** No se entrenan modelos radiográficos ni multimodales, no se descargan pesos preentrenados y no se ajustan calibración, umbral o modelo final con toda la cohorte.

**Cada trabajo completo de modelo y pliegue se guarda y verifica en Drive antes de continuar.** Si se desconecta Colab, vuelve a ejecutar esta misma revisión con las mismas carpetas. Solo se reutilizan trabajos completos y autenticados; un trabajo interrumpido se reinicia desde el principio. No se retoma un optimizador parcial. Conserva las carpetas de resultados y **no ejecutes dos sesiones simultáneas sobre la misma campaña**.

Los resultados serán de validación interna de desarrollo y servirán para comparar candidatos. La evaluación final con la prueba reservada corresponde a una etapa posterior."""), code(bootstrap),
        markdown("""## 1. Preparar el intérprete CPU aislado

Se conservan las versiones verificadas de las dependencias del ejecutor. La instalación se realiza en un intérprete temporal separado del intérprete principal de Colab. La GPU y la TPU deben estar desactivadas."""),
        code("""print('Preparando el intérprete CPU aislado; aún no se inició entrenamiento…')
runtime = json.loads(checked_run([sys.executable, '-B', str(REPO_DIR / 'scripts/train_clinical_fivefold_colab.py'), '--operation', 'setup', '--repository-root', str(REPO_DIR), '--destination', str(SESSION_ROOT / 'cpu_runtime')]))
NEURAL_PYTHON = Path(runtime['python_path'])
RUN_ENV = orchestration.runtime_environment(REPO_DIR)
print('Intérprete CPU preparado. Todavía no se entrenaron modelos de participantes.')
"""), markdown("""## 2. Verificar el código con pruebas sintéticas

Se comprueban las restricciones de alcance, la separación por participante y el guardado y recuperación de trabajos mediante pruebas sintéticas. Estas comprobaciones no entrenan con datos reales de la cohorte. Cualquier fallo u omisión detiene la libreta."""),
        code("""CPU_TESTS = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/train_clinical_fivefold_colab.py'), '--operation', 'tests', '--repository-root', str(REPO_DIR)], cwd=REPO_DIR, env=RUN_ENV))
print(f"Pruebas sintéticas aprobadas: {CPU_TESTS['tests_run']}; omitidas: {CPU_TESTS['skipped']}.")
"""), markdown("""## 3. Localizar la preparación 34 y reconstruir desarrollo

Indica las dos carpetas habituales. Se autentica la preparación del ejecutor de la libreta 34 y se reconstruye solo el transporte aprobado de desarrollo. Las fuentes de Drive se conservan intactas."""),
        code("""from google.colab import drive
drive.mount('/content/drive')
PERSISTENT_ROOT = orchestration.checked_path(input('Carpeta existente de KneeOAI con los registros privados de la libreta 34: ').strip())
REMOTE_PACKAGE_DIR = orchestration.checked_path(input('Misma carpeta privada de desarrollo, que contiene transport.json: ').strip())
for selected in (PERSISTENT_ROOT, REMOTE_PACKAGE_DIR):
    if not selected.is_dir() or not selected.is_relative_to(Path('/content/drive')):
        raise RuntimeError('Selecciona carpetas existentes dentro de tu Drive montado.')
proof = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/train_clinical_fivefold_colab.py'), '--operation', 'locate', '--repository-root', str(REPO_DIR), '--records-root', str(PERSISTENT_ROOT)], cwd=REPO_DIR, env=RUN_ENV))
PREPARATION_DIR = Path(proof['preparation_dir'])
LOCAL_PACKAGE_DIR = SESSION_ROOT / 'development'
REFERENCE_DIR = SESSION_ROOT / 'reference'
REFERENCE_DIR.mkdir(exist_ok=False)
LOCAL_REFERENCE = REFERENCE_DIR / 'model_inputs_loading_reference_v1.json'
transport = orchestration.stage_development(REMOTE_PACKAGE_DIR, LOCAL_PACKAGE_DIR, LOCAL_REFERENCE)
print('Preparación 34 autenticada. Paquete de desarrollo reconstruido y verificado.')
"""), markdown("""## 4. Confirmar y entrenar los tres modelos clínicos

Escribe **`AUTORIZO CLINICOS 5 PLIEGUES`** para ejecutar los 15 trabajos de esta etapa con la configuración registrada. Esta única confirmación cubre los tres modelos y sus cinco pliegues; no se solicitará otra entre trabajos.

Se mostrará el comienzo y cierre de cada trabajo, junto con una señal periódica de actividad y el tiempo transcurrido real. Esa señal no estima el porcentaje de avance ni la hora de finalización. Cada trabajo terminado se copia a Drive y se verifica antes de iniciar el siguiente.

Si hay una interrupción, conserva el error y los trabajos ya guardados. Al repetir la misma revisión se verificarán los trabajos completos antes de reutilizarlos. No cambies las variables, las semillas ni los parámetros para continuar."""),
        code("""confirmation = input('Escribe AUTORIZO CLINICOS 5 PLIEGUES para ejecutar solo la etapa descrita: ').strip()
if confirmation != 'AUTORIZO CLINICOS 5 PLIEGUES':
    raise RuntimeError('No se confirmó la etapa. No se inició entrenamiento.')
import queue, threading, time, uuid

def training_run(arguments, *, cwd, env):
    # Forward actual stderr progress; stdout contains the final structured record.
    process = subprocess.Popen(arguments, cwd=cwd, env=env, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    messages = queue.Queue()
    def forward(stream, channel):
        try:
            for line in iter(stream.readline, ''):
                messages.put((channel, line))
        finally:
            stream.close()
            messages.put((channel, None))
    for stream, channel in ((process.stdout, 'stdout'), (process.stderr, 'stderr')):
        threading.Thread(target=forward, args=(stream, channel), daemon=True).start()
    started, last_heartbeat, closed = time.monotonic(), time.monotonic(), set()
    stdout_lines = []
    try:
        while len(closed) < 2:
            try:
                channel, line = messages.get(timeout=1)
            except queue.Empty:
                channel, line = None, None
            if channel is not None:
                if line is None:
                    closed.add(channel)
                elif channel == 'stdout':
                    stdout_lines.append(line)
                else:
                    safe = re.sub(r'https?://\\S+', '<url>', line.rstrip())
                    safe = re.sub(r'(?i)(authorization|token|password|secret)\\s*[:=]\\s*\\S+', r'\\1=<redacted>', safe)
                    print(safe, flush=True)
            now = time.monotonic()
            if now - last_heartbeat >= 30 and process.poll() is None:
                print(f'Subproceso activo; tiempo transcurrido: {int(now - started)} s. No es un porcentaje de avance.', flush=True)
                last_heartbeat = now
        status = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if status:
        raise RuntimeError('El entrenamiento se detuvo. Conserva el error y los trabajos completos guardados en Drive; no cambies la receta.')
    return ''.join(stdout_lines)

LOCAL_OUTPUT = SESSION_ROOT / ('clinical_fivefold_' + uuid.uuid4().hex)
TEST_RECEIPT = SESSION_ROOT / 'synthetic_test_receipt.json'
RUNTIME_RECEIPT = SESSION_ROOT / 'cpu_setup_receipt.json'
orchestration._save(TEST_RECEIPT, CPU_TESTS)
orchestration._save(RUNTIME_RECEIPT, runtime)
print('Entrenando regresión logística, XGBoost y MLP en CPU, cinco pliegues de desarrollo y semilla 2026. Cada trabajo completo se guardará antes de continuar…', flush=True)
training_log = training_run([str(NEURAL_PYTHON), '-u', '-B', str(REPO_DIR / 'scripts/train_clinical_fivefold_colab.py'), '--operation', 'train', '--repository-root', str(REPO_DIR), '--preparation-dir', str(PREPARATION_DIR), '--package-dir', str(LOCAL_PACKAGE_DIR), '--reference-path', str(LOCAL_REFERENCE), '--output-dir', str(LOCAL_OUTPUT), '--records-root', str(PERSISTENT_ROOT), '--code-revision', CODE_REVISION, '--test-receipt', str(TEST_RECEIPT), '--runtime-receipt', str(RUNTIME_RECEIPT), '--approval-statement', orchestration.APPROVAL_STATEMENT], cwd=REPO_DIR, env=RUN_ENV)
FINAL_CHECK = json.loads((LOCAL_OUTPUT / orchestration.SUMMARY_FILENAME).read_text(encoding='utf-8'))
orchestration.validate_result(FINAL_CHECK)
print('Los 15 trabajos clínicos terminaron; los resultados privados están guardados y verificados en Drive.')
"""), markdown("""## 5. Mostrar el registro final

Para cada modelo se presenta la media no ponderada de las métricas de sus cinco pliegues. Las métricas calculadas al reunir todas las predicciones fuera de pliegue se describen por separado y no sustituyen la métrica de selección registrada. Esta etapa no elige todavía el sistema final ni compara por sí sola todas las alternativas radiográficas y multimodales.

Comparte únicamente el **JSON final**. Las predicciones individuales y los modelos entrenados se conservan en Drive; no es necesario enviar imágenes o archivos privados."""),
        code("""print(json.dumps(FINAL_CHECK, ensure_ascii=False, indent=2))
print('Regresión logística, XGBoost y MLP: cinco pliegues de desarrollo por modelo guardados y verificados. Prueba reservada sin abrir.')
""")]
    metadata = dict(previous["metadata"])
    metadata.pop("accelerator", None)
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--allow-unpublished-placeholder", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / NOTEBOOK)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.code_revision,
        allow_unpublished_placeholder=args.allow_unpublished_placeholder), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
