"""Build notebook33 with a published exact code revision; no private inputs."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re


NOTEBOOK = "notebooks/33_verificar_modelos_finales_sin_entrenar.ipynb"
UNPUBLISHED = "PENDING_PUBLISHED_COMMIT"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    ast.parse(source)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def build(code_revision, *, allow_unpublished_placeholder=False):
    if not (isinstance(code_revision, str) and re.fullmatch(r"[0-9a-f]{40}", code_revision)
            and code_revision != "0" * 40):
        if not (allow_unpublished_placeholder and code_revision == UNPUBLISHED):
            raise ValueError("published_exact_code_revision_required")
    bootstrap = """from pathlib import Path, PurePosixPath
import json, os, re, shutil, stat, subprocess, sys, tempfile, urllib.request, zipfile

CODE_REVISION = __REVISION__
if not re.fullmatch(r'[0-9a-f]{40}', CODE_REVISION) or CODE_REVISION == '0' * 40:
    raise RuntimeError('Esta copia no tiene una revisión publicada. Usa la libreta final entregada.')
if os.environ.get('COLAB_GPU', '0') == '1' or any(os.environ.get(k) for k in ('COLAB_TPU_ADDR', 'TPU_NAME', 'COLAB_TPU_TASK_ID')):
    raise RuntimeError('Selecciona CPU en Entorno de ejecución > Cambiar tipo de entorno de ejecución.')
if not os.environ.get('COLAB_RELEASE_TAG'):
    raise RuntimeError('Esta libreta debe ejecutarse personalmente en Google Colab; no es un ensayo local.')
if 'knee' in sys.modules or any(name.startswith('knee.') for name in sys.modules):
    raise RuntimeError('Reinicia el entorno para cargar la revisión fijada de esta libreta.')
SESSION_ROOT = Path(tempfile.mkdtemp(prefix='knee_final_models33_'))
REPO_DIR = SESSION_ROOT / 'repository'
archive_path = SESSION_ROOT / 'code.zip'
url = 'https://codeload.github.com/AlonsoYess/Knee/zip/' + CODE_REVISION
with urllib.request.urlopen(url, timeout=60) as response, archive_path.open('xb') as target:
    if response.geturl() != url:
        raise RuntimeError('La descarga del código cambió de ubicación.')
    total = 0
    while True:
        chunk = response.read(1024**2)
        if not chunk:
            break
        total += len(chunk)
        if total > 128 * 1024**2:
            raise RuntimeError('El archivo del repositorio supera el límite autorizado.')
        target.write(chunk)
REPO_DIR.mkdir()
with zipfile.ZipFile(archive_path) as archive:
    members = archive.infolist()
    if not members or len(members) > 20000 or sum(i.file_size for i in members) > 512 * 1024**2:
        raise RuntimeError('El contenido del repositorio supera los límites de extracción.')
    seen = set()
    for item in members:
        name = item.filename.rstrip('/')
        parts = name.split('/')
        if not name or '\\\\' in name or ':' in name or '\\0' in name or any(p in ('', '.', '..') for p in parts) or parts[0] != 'Knee-' + CODE_REVISION or name in seen or stat.S_ISLNK(item.external_attr >> 16):
            raise RuntimeError('Se encontró una ruta no permitida en el repositorio.')
        seen.add(name)
        destination = REPO_DIR.joinpath(*parts[1:])
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            if len(parts) < 2:
                raise RuntimeError('El archivo del repositorio no tiene una ruta relativa válida.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024**2)
sys.path.insert(0, str(REPO_DIR / 'src'))
from knee import final_models_colab as orchestration

def checked_run(arguments, **kwargs):
    result = subprocess.run(arguments, capture_output=True, text=True, **kwargs)
    if result.returncode:
        detail = re.sub(r'https?://\\S+', '<url>', (result.stdout + result.stderr)[-12000:])
        detail = re.sub(r'(?i)(authorization|token|password|secret)\\s*[:=]\\s*\\S+', r'\\1=<redacted>', detail)
        print(detail)
        raise RuntimeError('La comprobación se detuvo. No ejecutes entrenamiento.')
    return result.stdout

print('Revisión exacta del código preparada. No se abrieron datos de participantes.')
""".replace("__REVISION__", repr(code_revision))
    cells = [markdown("""# 33 · Verificar los modelos finales, sin entrenar

Selecciona **CPU** y ejecuta las celdas en orden. La lista queda cerrada con BiomedCLIP y MambaOut-Tiny como candidatos adicionales. Esta libreta comprueba **seis encoders accesibles de siete registrados**; DINOv3 sigue pendiente de acceso autorizado y no se sustituye por otro checkpoint.

Solo se generan dos imágenes sintéticas en memoria. **No necesitas el paquete de desarrollo ni el reservado**, ni ejecutar las libretas anteriores. No se entrena, no se crean optimizadores y no se modifican originales o particiones.

Se descargarán aproximadamente **2,7 GB de pesos públicos** al almacenamiento temporal local de Colab, además del intérprete y las bibliotecas. Los pesos se reutilizan si repites una comprobación dentro de la misma sesión; no se guardan en el repositorio ni se redistribuyen en Drive. Allí solo se guardarán los registros de verificación. Todo el cálculo se ejecuta mediante código Python del repositorio en una revisión fija.

Esta prueba acredita carga e inferencia sintética en CPU: **no es entrenamiento, una prueba GPU, validación clínica ni una comparación de desempeño predictivo**."""), code(bootstrap), markdown("""## 1. Preparar el entorno CPU aislado

No se altera el intérprete principal de Colab. Se instala la versión fijada de Python y las dependencias de los encoders. La primera preparación puede tardar varios minutos."""), code("""print('Preparando el intérprete aislado para los modelos finales…')
runtime = json.loads(checked_run([sys.executable, '-B', str(REPO_DIR / 'scripts/check_final_models_colab.py'), '--operation', 'setup', '--repository-root', str(REPO_DIR), '--destination', str(SESSION_ROOT / 'cpu_runtime')]))
NEURAL_PYTHON = Path(runtime['python_path'])
RUN_ENV = orchestration.runtime_environment(REPO_DIR)
print('Entorno CPU aislado preparado; aún no se descargaron pesos ni se abrieron datos.')
"""), markdown("""## 2. Verificar las pruebas del código

Se ejecutan pruebas sintéticas del transporte, los adaptadores, la configuración, las reglas del registro y esta libreta. Cualquier error u omisión detiene el proceso."""), code("""CPU_TESTS = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_final_models_colab.py'), '--operation', 'tests', '--repository-root', str(REPO_DIR)], cwd=REPO_DIR, env=RUN_ENV))
print(f"Pruebas aprobadas: {CPU_TESTS['tests_run']}; omitidas: {CPU_TESTS['skipped']}.")
"""), markdown("""## 3. Comprobar pesos e inferencia sintética

Se verifican la procedencia y las huellas de los archivos, las dimensiones de salida y que dos pasadas idénticas produzcan resultados idénticos sin cambiar los pesos. BiomedCLIP se usa solo como codificador visual: no se ingresan textos ni datos clínicos. Cada ejecución escribe una carpeta nueva; ninguna anterior se reemplaza."""), code("""import uuid
WEIGHTS_ROOT = SESSION_ROOT / 'public_weight_cache'
LOCAL_OUTPUT = SESSION_ROOT / ('verification_' + uuid.uuid4().hex)
print('Verificando seis encoders en CPU con imágenes sintéticas. Sin entrenamiento…')
verification_log = checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/check_final_models_colab.py'), '--operation', 'verify', '--repository-root', str(REPO_DIR), '--weights-root', str(WEIGHTS_ROOT), '--output-dir', str(LOCAL_OUTPUT), '--allow-download', '--execution-environment', 'google_colab_cpu'], cwd=REPO_DIR, env=RUN_ENV)
result = json.loads((LOCAL_OUTPUT / 'checkpoint_verification.json').read_text(encoding='utf-8'))
orchestration.validate_result(result, 'google_colab_cpu')
print('Verificados:', result['pretrained_models_loaded'], 'de', result['registered_image_models'], 'registrados. DINOv3 conserva su acceso pendiente.')
"""), markdown("""## 4. Guardar el registro en tu Drive

Indica una carpeta existente dentro de tu espacio privado de KneeOAI. Se crearán únicamente dos archivos JSON en una subcarpeta nueva. No selecciones ni abras paquetes de imágenes; no se necesita leerlos para esta comprobación."""), code("""from google.colab import drive
drive.mount('/content/drive')
PERSISTENT_ROOT = orchestration.checked_path(input('Carpeta existente de KneeOAI en Drive donde guardar la comprobación: ').strip())
if not PERSISTENT_ROOT.is_dir() or not PERSISTENT_ROOT.is_relative_to(Path('/content/drive')):
    raise RuntimeError('Selecciona una carpeta existente dentro del Drive montado.')
FINAL_CHECK = orchestration.publish_result(LOCAL_OUTPUT, PERSISTENT_ROOT, CODE_REVISION, CPU_TESTS, runtime)
print(json.dumps(FINAL_CHECK, ensure_ascii=False, indent=2))
print('Comprobación guardada. No se entrenó, no se abrieron datos de participantes ni la prueba reservada.')
"""), markdown("""Comparte únicamente el **JSON final**. No necesitas enviar imágenes, pesos ni archivos privados. Aunque la comprobación pase, el entrenamiento y la prueba corta en GPU siguen pendientes de autorización y ejecución separadas. La condición pendiente de DINOv3 no se presenta como aprobada.""")]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--allow-unpublished-placeholder", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / NOTEBOOK)
    args = parser.parse_args()
    notebook = build(args.code_revision, allow_unpublished_placeholder=args.allow_unpublished_placeholder)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
