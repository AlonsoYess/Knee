"""Build notebook36 with the exact published frozen DenseNet five-fold code."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path


NOTEBOOK = "notebooks/36_entrenar_densenet121_congelado_cinco_pliegues.ipynb"
UNPUBLISHED = "PENDING_PUBLISHED_COMMIT"


def _previous_builder():
    spec = importlib.util.spec_from_file_location("knee_notebook36_bootstrap_source",
        Path(__file__).resolve().with_name("build_training_smoke_notebook.py"))
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
    bootstrap = previous["cells"][1]["source"].replace("knee_training_smoke35_", "knee_densenet36_")
    bootstrap = bootstrap.replace("from knee import training_smoke_colab as orchestration",
        "from knee import densenet_fivefold_colab as orchestration")
    bootstrap = bootstrap.replace("La prueba breve se detuvo. Conserva el error; no intentes entrenamiento completo.",
        "La ejecución se detuvo. Conserva el error y los pliegues completos de Drive; no cambies la receta.")
    cells = [markdown("""# 36 · DenseNet121 congelado: cinco pliegues de desarrollo

**Revisión 1.0. Esta libreta sí ejecuta entrenamiento, no es otra prueba de dos pasos.** Selecciona **GPU** y ejecuta las celdas en orden en una sesión nueva. El único modelo es **DenseNet121**, con su **codificador congelado** y la cabeza de clasificación entrenable, semilla **2026**, en los cinco pliegues ya establecidos de desarrollo.

Cada pliegue comienza desde el checkpoint público original verificado y una cabeza nueva. No se utilizan los pesos de la prueba breve 35. Se aplica la receta registrada: máximo de **60 épocas**, parada temprana con paciencia **8**, micro-lote de **2** y grupo efectivo de **16** rodillas. Los cinco pliegues se ejecutan en secuencia; esta etapa puede durar varias horas según la GPU y la parada temprana. No se entrenan otros modelos, no se descongela el codificador, no se cambian las particiones y **no se abre la prueba reservada**.

La libreta verifica la prueba GPU 35 ya guardada y copia únicamente el paquete aprobado de desarrollo. No se modifica ningún original. La normalización y el procesamiento clínico se limitan a lo ya registrado; no se incorporan variables o imágenes nuevas.

**Cada pliegue completo se guarda y verifica en Drive antes de iniciar el siguiente.** Si se pierde la sesión, vuelve a abrir esta misma revisión y ejecuta todas las celdas desde el inicio, con las mismas carpetas. Se recuperan únicamente los pliegues completos, autenticados y compatibles con el código y los datos exactos. Un pliegue interrumpido se reinicia desde el checkpoint público; no se continúan su optimizador ni sus pesos parciales. No borres las carpetas de resultados.

**No ejecutes dos sesiones o copias de esta libreta simultáneamente sobre las mismas carpetas.** Cada campaña admite un solo proceso escritor. Las publicaciones finales se preparan en archivos temporales únicos y solo reciben su nombre final después de verificar todas sus huellas. Un archivo temporal interrumpido se conserva, pero no se usa como resultado completo.

Necesitarás las mismas dos carpetas de Drive: registros privados y desarrollo (la que contiene `transport.json` y `model_inputs_loading_reference_v1.json`). No selecciones el paquete reservado. Los pesos y las predicciones se guardan únicamente en la carpeta privada de Drive, nunca en GitHub.

Los resultados de esta etapa describen validación interna de desarrollo. No constituyen una evaluación final, una comparación con los demás modelos o una validación clínica."""), code(bootstrap),
        markdown("""## 1. Preparar el intérprete GPU aislado

Se utiliza el mismo entorno CUDA verificado en la libreta 35. La GPU real y las versiones se comprueban antes de abrir datos. No se modifica el intérprete principal de Colab ni se permite sustituir silenciosamente GPU por CPU."""),
        code("""print('Preparando el intérprete GPU aislado; aún no se abrieron datos ni se entrenó…')
runtime = json.loads(checked_run([sys.executable, '-B', str(REPO_DIR / 'scripts/train_densenet_fivefold_colab.py'), '--operation', 'setup', '--repository-root', str(REPO_DIR), '--destination', str(SESSION_ROOT / 'gpu_runtime')]))
NEURAL_PYTHON = Path(runtime['python_path'])
RUN_ENV = orchestration.runtime_environment(REPO_DIR)
print('Intérprete y GPU verificados. Todavía no se entrenó ningún pliegue.')
"""), markdown("""## 2. Verificar el código mediante pruebas sintéticas

En este subproceso se oculta la GPU. Las pruebas emplean datos sintéticos y sustituciones de las operaciones de entrenamiento; no entrenan modelos reales ni abren el paquete de participantes. Cualquier error u omisión detiene la libreta."""),
        code("""TEST_ENV = dict(RUN_ENV)
TEST_ENV.update(CUDA_VISIBLE_DEVICES='', COLAB_GPU='0')
CPU_TESTS = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/train_densenet_fivefold_colab.py'), '--operation', 'tests', '--repository-root', str(REPO_DIR)], cwd=REPO_DIR, env=TEST_ENV))
print(f"Pruebas sintéticas aprobadas: {CPU_TESTS['tests_run']}; omitidas: {CPU_TESTS['skipped']}.")
"""), markdown("""## 3. Localizar la prueba 35 y reconstruir solo desarrollo

Indica las dos carpetas habituales. Se autentican los registros completos de la prueba GPU 35 con revisión AMP corregida y se reconstruye únicamente el transporte de desarrollo. Las referencias, partes y huellas se comprueban sin modificar las fuentes de Drive. No es necesario repetir las libretas anteriores."""),
        code("""from google.colab import drive
drive.mount('/content/drive')
PERSISTENT_ROOT = orchestration.checked_path(input('Carpeta existente de KneeOAI con los registros privados de la libreta 35: ').strip())
REMOTE_PACKAGE_DIR = orchestration.checked_path(input('Misma carpeta privada de desarrollo, que contiene transport.json: ').strip())
for selected in (PERSISTENT_ROOT, REMOTE_PACKAGE_DIR):
    if not selected.is_dir() or not selected.is_relative_to(Path('/content/drive')):
        raise RuntimeError('Selecciona carpetas existentes dentro de tu Drive montado.')
proof = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/train_densenet_fivefold_colab.py'), '--operation', 'locate', '--repository-root', str(REPO_DIR), '--records-root', str(PERSISTENT_ROOT)], cwd=REPO_DIR, env=RUN_ENV))
SMOKE_DIR = Path(proof['smoke_dir'])
LOCAL_PACKAGE_DIR = SESSION_ROOT / 'development'
REFERENCE_DIR = SESSION_ROOT / 'reference'
REFERENCE_DIR.mkdir(exist_ok=False)
LOCAL_REFERENCE = REFERENCE_DIR / 'model_inputs_loading_reference_v1.json'
transport = orchestration.stage_development(REMOTE_PACKAGE_DIR, LOCAL_PACKAGE_DIR, LOCAL_REFERENCE)
print('Prueba 35 autenticada. Solo desarrollo reconstruido y verificado; originales intactos.')
"""), markdown("""## 4. Confirmar y entrenar únicamente DenseNet121 congelado

Escribe **`AUTORIZO DENSENET 5 PLIEGUES`** para autorizar una sola ejecución de la receta mostrada en los cinco pliegues de desarrollo. No se solicitará otra confirmación entre pliegues.

Mantén la sesión conectada y permite que el proceso termine. Se mostrará cuándo empieza y termina cada pliegue y una señal periódica de que el subproceso sigue vivo, con tiempo transcurrido real; **esa señal no representa un porcentaje de avance**. No se inventa un progreso de épocas ni una hora de finalización. Los pliegues terminados se copian a Drive y se verifican inmediatamente. Una reconexión no retoma un pliegue a medias.

El codificador permanece congelado. AMP conserva las mismas rodillas si debe reducir su escala, sin descartar ejemplos. Si hay un error de memoria u otro fallo, conserva el mensaje y los resultados ya guardados; no modifiques el lote, las épocas o el modelo por tu cuenta."""),
        code("""confirmation = input('Escribe AUTORIZO DENSENET 5 PLIEGUES para ejecutar solo la etapa descrita: ').strip()
if confirmation != 'AUTORIZO DENSENET 5 PLIEGUES':
    raise RuntimeError('No se confirmó la etapa. No se inició entrenamiento.')
import queue, threading, time, uuid

def training_run(arguments, *, cwd, env):
    # Stream real stderr; stdout is the structured final record, not a progress estimate.
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
    stdout_lines, error_lines = [], []
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
                    error_lines.append(safe)
                    if len(error_lines) > 100:
                        error_lines.pop(0)
                    print(safe, flush=True)
            now = time.monotonic()
            if now - last_heartbeat >= 30 and process.poll() is None:
                print(f'Subproceso activo; tiempo transcurrido: {int(now - started)} s. No es un porcentaje de avance.', flush=True)
                last_heartbeat = now
        status = process.wait()
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        raise
    if status:
        raise RuntimeError('El entrenamiento se detuvo. Conserva el error y los pliegues completos guardados en Drive; no cambies la receta.')
    return ''.join(stdout_lines)

WEIGHTS_ROOT = SESSION_ROOT / 'public_weight_cache'
WEIGHTS_ROOT.mkdir(exist_ok=False)
LOCAL_OUTPUT = SESSION_ROOT / ('fivefold_' + uuid.uuid4().hex)
TEST_RECEIPT = SESSION_ROOT / 'synthetic_test_receipt.json'
RUNTIME_RECEIPT = SESSION_ROOT / 'gpu_setup_receipt.json'
orchestration._save(TEST_RECEIPT, CPU_TESTS)
orchestration._save(RUNTIME_RECEIPT, runtime)
print('Entrenando DenseNet121 con codificador congelado, cinco pliegues de desarrollo y semilla 2026. Se guardará cada pliegue antes de continuar…', flush=True)
training_log = training_run([str(NEURAL_PYTHON), '-u', '-B', str(REPO_DIR / 'scripts/train_densenet_fivefold_colab.py'), '--operation', 'train', '--repository-root', str(REPO_DIR), '--smoke-dir', str(SMOKE_DIR), '--package-dir', str(LOCAL_PACKAGE_DIR), '--reference-path', str(LOCAL_REFERENCE), '--weights-root', str(WEIGHTS_ROOT), '--output-dir', str(LOCAL_OUTPUT), '--records-root', str(PERSISTENT_ROOT), '--code-revision', CODE_REVISION, '--test-receipt', str(TEST_RECEIPT), '--runtime-receipt', str(RUNTIME_RECEIPT), '--approval-statement', orchestration.APPROVAL_STATEMENT], cwd=REPO_DIR, env=RUN_ENV)
FINAL_CHECK = json.loads((LOCAL_OUTPUT / orchestration.SUMMARY_FILENAME).read_text(encoding='utf-8'))
orchestration.validate_result(FINAL_CHECK)
print('Los cinco pliegues terminaron; los resultados privados ya están guardados y verificados en Drive.')
"""), markdown("""## 5. Mostrar el registro final

La selección interna se resume mediante la media de las métricas de los cinco pliegues; las predicciones agrupadas se describen por separado. Esta etapa no determina aún cuál es el mejor modelo entre todas las alternativas. No se ajustan probabilidades finales, umbrales o modelos con la prueba reservada.

Comparte únicamente el **JSON final**. No compartas imágenes, pesos privados ni identificadores de participantes."""),
        code("""print(json.dumps(FINAL_CHECK, ensure_ascii=False, indent=2))
print('DenseNet121 congelado: cinco pliegues de desarrollo guardados y verificados. Prueba reservada sin abrir; no se entrenaron otros modelos ni se descongeló el codificador.')
""")]
    return {"cells": cells, "metadata": {**previous["metadata"], "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--allow-unpublished-placeholder", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / NOTEBOOK)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.code_revision,
        allow_unpublished_placeholder=args.allow_unpublished_placeholder), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
