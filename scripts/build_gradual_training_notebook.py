"""Build the thin, revision-pinned notebook42 without changing notebook41."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

CAMPAIGN = "gradual_four_models"
NOTEBOOK_PATH = "notebooks/42_entrenar_ajuste_gradual_cinco_pliegues.ipynb"
MODEL_NAMES = ("DenseNet121", "ViT-B/16", "Skelex", "MambaOut-Tiny")


def _previous():
    source = Path(__file__).resolve().with_name("build_image_screening_notebook.py")
    spec = importlib.util.spec_from_file_location("verified_image_screening_builder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("verified_gradual_notebook_template_changed")
    return source.replace(old, new, 1)


def _markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source):
    ast.parse(source)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def build(code_revision, *, supervision_revision=None, allow_unpublished_placeholder=False):
    supervision_revision = supervision_revision or code_revision
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", supervision_revision) or supervision_revision == "0" * 40:
        raise ValueError("published_supervision_revision_required")
    previous = _previous().build(code_revision, allow_unpublished_placeholder=allow_unpublished_placeholder)
    value = {**previous, "cells": previous["cells"][:8]}
    value["cells"][0] = _markdown(
        "# 42 · Ajuste gradual de los codificadores seleccionados\n\n"
        "**Revisión 1.2 · Plan metodológico v1.2 · GPU.**\n\n"
        "La campaña autorizada comprende **20 trabajos**: **DenseNet121, ViT-B/16, Skelex y MambaOut-Tiny**, "
        "cada uno en los mismos cinco pliegues de desarrollo, con semilla 2026. La selección se vuelve a "
        "comprobar contra los resultados completos y privados de las libretas 40 y 41; esta libreta no "
        "agrega candidatos ni cambia los criterios de selección.\n\n"
        "Cada pliegue nuevo parte de los pesos externos registrados y una cabeza nueva: cinco épocas con "
        "el codificador congelado, cinco con el último bloque entrenable y después ajuste del codificador "
        "completo, hasta el límite y la parada temprana del plan. Se mantienen las particiones por participante, "
        "las imágenes de 224 × 224 y el aumento gamma únicamente en entrenamiento.\n\n"
        "Antes de ejecutar un modelo pendiente se comprueban su asignación de capas, gradientes y memoria "
        "en GPU. Un ajuste autorizado del tamaño de los microlotes conserva el lote efectivo de 16; estas "
        "comprobaciones no aportan pesos ni métricas a los pliegues completos.\n\n"
        "Los trabajos completos se conservan sin copiar sus checkpoints al entorno de Colab. Los trabajos "
        "nuevos guardan el estado necesario al final de cada época para poder recuperarse. Se mantienen "
        "las dos generaciones recientes verificadas de cada trabajo; una época interrumpida antes de "
        "guardarse se repite. Para retomar un pliegue, conserva esta revisión y el mismo tipo de GPU y entorno; "
        "si no son compatibles, la libreta se detiene y conserva el avance. Ejecuta una sola sesión de esta campaña.\n\n"
        "La supervisión guarda en Drive el consumo de RAM y GPU, los mensajes y el cierre de cada intento. "
        "Un fallo HTTP 502, 503 o 504 identificado admite un único reintento. Una interrupción desconocida, "
        "un problema de memoria o una incompatibilidad detienen la ejecución con un registro; no se cambia "
        "la receta para eludirlos. Si desaparece el entorno completo de Colab, vuelve a ejecutar las etapas "
        "desde el inicio para consultar el último registro y recuperar el avance. El código no puede "
        "reiniciar por sí mismo un entorno de Colab desconectado.\n\n"
        "Las primeras cuatro etapas preparan y comprueban las entradas. **El entrenamiento comienza al "
        "ejecutar la etapa 5**, de acuerdo con la autorización ya otorgada; no se solicita una nueva frase "
        "ni es necesario escribir rutas. Los originales y las particiones no se modifican. La prueba "
        "reservada, la fusión multimodal, la calibración y el ajuste final quedan fuera de esta campaña.")
    bootstrap = value["cells"][1]["source"]
    for old, new in (("knee_image_screening41_", "knee_gradual42_"),
            ("from knee import image_campaign_recovery as orchestration",
             "from knee import gradual_campaign_colab as orchestration"),
            ("CAMPAIGN = 'remaining_encoders_frozen'", "CAMPAIGN = 'gradual_four_models'")):
        bootstrap = _replace_once(bootstrap, old, new)
    support_paths = ("scripts/colab_training_supervisor.py", "tests/test_colab_training_supervisor.py")
    root = Path(__file__).resolve().parents[1]
    support_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in support_paths}
    bootstrap += """\n# Operational helper is independently pinned; the frozen training checkout stays unchanged.
import hashlib, importlib.util
SUPERVISION_REVISION = __SUPERVISION_REVISION__
SUPPORT_HASHES = __SUPPORT_HASHES__
SUPPORT_DIR = SESSION_ROOT / 'supervision'
for relative, expected_hash in SUPPORT_HASHES.items():
    support_url = 'https://raw.githubusercontent.com/AlonsoYess/Knee/' + SUPERVISION_REVISION + '/' + relative
    with urllib.request.urlopen(support_url, timeout=60) as response:
        if response.geturl() != support_url:
            raise RuntimeError('La descarga de la supervisión cambió de ubicación.')
        payload = response.read(1024**2 + 1)
    if len(payload) > 1024**2 or hashlib.sha256(payload).hexdigest() != expected_hash:
        raise RuntimeError('La supervisión descargada no coincide con la revisión publicada.')
    destination = SUPPORT_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(payload)
support_spec = importlib.util.spec_from_file_location('knee_colab_supervision42', SUPPORT_DIR / 'scripts/colab_training_supervisor.py')
supervision = importlib.util.module_from_spec(support_spec)
sys.modules[support_spec.name] = supervision
support_spec.loader.exec_module(supervision)
print('Supervisión operativa verificada; el código de entrenamiento conserva su revisión original.')
""".replace("__SUPERVISION_REVISION__", repr(supervision_revision)).replace("__SUPPORT_HASHES__", repr(support_hashes))
    value["cells"][1] = _code(bootstrap)
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = cell["source"].replace("scripts/train_image_screening_colab.py",
                "scripts/train_gradual_campaign_colab.py")
    value["cells"][4] = _markdown("## 2. Comprobar la implementación con datos sintéticos\n\n"
        "Se comprueban el alcance, la selección, la recuperación y la separación entre preparación y "
        "entrenamiento. También se prueba la supervisión con procesos pequeños sintéticos. Las pruebas "
        "utilizan CPU con la GPU oculta y no abren datos de participantes.")
    value["cells"][5] = _code(value["cells"][5]["source"] + """
SUPERVISION_TEST_ENV = dict(os.environ)
SUPERVISION_TEST_ENV.update(CUDA_VISIBLE_DEVICES='', COLAB_GPU='0', PYTHONDONTWRITEBYTECODE='1', KNEE_FROZEN_REPOSITORY=str(REPO_DIR))
supervision_tests = checked_run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(SUPPORT_DIR / 'tests'), '-p', 'test_colab_training_supervisor.py', '-v'], cwd=SUPPORT_DIR, env=SUPERVISION_TEST_ENV)
print('Pruebas de supervisión aprobadas: procesos sintéticos en CPU, sin entrenamiento de participantes.')
""")
    value["cells"][6] = _markdown("## 3. Reconstruir únicamente las entradas de desarrollo\n\n"
        "Las carpetas se leen de la misma configuración privada utilizada en la libreta 41. Se comprueban "
        "las partes del paquete y la referencia de entradas. Esta etapa no selecciona datos nuevos ni "
        "descarga el paquete reservado.")
    locate = value["cells"][7]["source"]
    start = locate.index("proof = json.loads(checked_run(")
    stop = locate.index("LOCAL_PACKAGE_DIR = SESSION_ROOT")
    locate = locate[:start] + locate[stop:]
    locate = _replace_once(locate,
        "print('Comprobación 38 localizada. Paquete de desarrollo reconstruido y verificado.')",
        "print('Paquete de desarrollo reconstruido y verificado. No se inició entrenamiento.')")
    locate = _replace_once(locate, "LOCAL_PACKAGE_DIR = SESSION_ROOT",
        "supervision.show_previous(PERSISTENT_ROOT)\nLOCAL_PACKAGE_DIR = SESSION_ROOT")
    value["cells"][7] = _code(locate)
    value["cells"] += [_markdown("## 4. Verificar la selección y preparar los 20 trabajos\n\n"
        "Se localizan los resultados completos de las libretas 40 y 41, se comprueban las predicciones y "
        "métricas de desarrollo y se reconstruye la selección conforme al plan. Se reconoce el cierre "
        "autorizado aunque existan carpetas de intentos incompletos anteriores. El conjunto resultante "
        "debe ser DenseNet121, ViT-B/16, Skelex y MambaOut-Tiny. Si la evidencia no coincide, la preparación "
        "se detiene. Esta etapa no descarga pesos ni entrena; guarda una preparación verificable en Drive."),
        _code("""if CAMPAIGN != 'gradual_four_models':
    raise RuntimeError('Esta libreta solo prepara la campaña gradual autorizada.')
import uuid
TEST_RECEIPT = SESSION_ROOT / 'synthetic_test_receipt.json'
RUNTIME_RECEIPT = SESSION_ROOT / 'setup_receipt.json'
for receipt_path, payload in ((TEST_RECEIPT, CPU_TESTS), (RUNTIME_RECEIPT, runtime)):
    if receipt_path.exists():
        if orchestration._json(receipt_path) != payload:
            raise RuntimeError('El recibo pertenece a otra preparación; inicia una sesión nueva.')
    else:
        orchestration._save(receipt_path, payload)
PREPARATION_DIR = SESSION_ROOT / ('gradual_preparation_' + uuid.uuid4().hex)
print('Comprobando las fuentes 40 y 41 y la selección de desarrollo. Sin entrenamiento…')
PREPARATION_CHECK = json.loads(checked_run([str(NEURAL_PYTHON), '-B', str(REPO_DIR / 'scripts/train_gradual_campaign_colab.py'), '--operation', 'prepare', '--campaign', CAMPAIGN, '--repository-root', str(REPO_DIR), '--records-root', str(PERSISTENT_ROOT), '--package-dir', str(LOCAL_PACKAGE_DIR), '--reference-path', str(LOCAL_REFERENCE), '--output-dir', str(PREPARATION_DIR), '--code-revision', CODE_REVISION, '--test-receipt', str(TEST_RECEIPT), '--runtime-receipt', str(RUNTIME_RECEIPT)], cwd=REPO_DIR, env=RUN_ENV))
orchestration.validate_preparation_receipt(REPO_DIR, PREPARATION_DIR, code_revision=CODE_REVISION)
print('Preparación verificada y guardada. Esta etapa no ejecutó entrenamiento.')
""")]
    run = """if CAMPAIGN != 'gradual_four_models':
    raise RuntimeError('Esta libreta no autoriza otra campaña.')
orchestration.validate_preparation_receipt(REPO_DIR, PREPARATION_DIR, code_revision=CODE_REVISION)
import uuid

def training_run(arguments, *, cwd, env):
    outcome = supervision.run(arguments, cwd=cwd, env=env, records_root=PERSISTENT_ROOT,
        session_root=SESSION_ROOT, code_revision=CODE_REVISION, sample_interval=2,
        heartbeat_interval=30, max_retries=1, retry_delay=30)
    globals()['LOCAL_OUTPUT'] = outcome.output_dir
    return outcome.stdout

WEIGHTS_ROOT = SESSION_ROOT / 'verified_public_weights'
WEIGHTS_ROOT.mkdir(exist_ok=True)
LOCAL_OUTPUT = SESSION_ROOT / ('gradual_campaign_' + uuid.uuid4().hex)
print('Iniciando solo los trabajos pendientes del ajuste gradual autorizado…', flush=True)
training_log = training_run([str(NEURAL_PYTHON), '-u', '-B', str(REPO_DIR / 'scripts/train_gradual_campaign_colab.py'), '--operation', 'train', '--campaign', CAMPAIGN, '--repository-root', str(REPO_DIR), '--preparation-dir', str(PREPARATION_DIR), '--package-dir', str(LOCAL_PACKAGE_DIR), '--reference-path', str(LOCAL_REFERENCE), '--output-dir', str(LOCAL_OUTPUT), '--records-root', str(PERSISTENT_ROOT), '--weights-root', str(WEIGHTS_ROOT), '--code-revision', CODE_REVISION], cwd=REPO_DIR, env=RUN_ENV)
FINAL_CHECK = json.loads((LOCAL_OUTPUT / orchestration.SUMMARY_FILENAME).read_text(encoding='utf-8'))
orchestration.validate_result(FINAL_CHECK)
print('Campaña gradual terminada y guardada en Drive. La prueba reservada permanece cerrada.')
"""
    value["cells"] += [_markdown("## 5. Ejecutar el ajuste gradual autorizado\n\n"
        "Esta celda inicia la comprobación GPU y los trabajos pendientes. Se conserva la receta aprobada "
        "y cada pliegue mantiene sus participantes separados. El progreso muestra modelo, pliegue y época; "
        "los guardados permiten recuperar las épocas completas después de una interrupción. Los resultados "
        "se guardan en la misma carpeta privada de registros. Si el entorno desaparece sin cierre, la "
        "supervisión conserva el último consumo observado sin atribuir automáticamente la causa a memoria. "
        "No inicies otra sesión de esta campaña en paralelo."), _code(run),
        _markdown("## 6. Mostrar el resultado\n\n"
        "Comparte únicamente el JSON final. Las predicciones individuales, los modelos y los estados "
        "de recuperación permanecen privados. Las métricas corresponden a desarrollo y no sustituyen "
        "la evaluación final del conjunto reservado."),
        _code("print(json.dumps(FINAL_CHECK, ensure_ascii=False, indent=2))\n")]
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse(cell["source"])
            if cell["outputs"] or cell["execution_count"] is not None:
                raise ValueError("clean_unexecuted_gradual_notebook_required")
            if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "input" for node in ast.walk(tree)):
                raise ValueError("authorized_gradual_notebook_must_not_prompt")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--supervision-revision", required=True)
    parser.add_argument("--allow-unpublished-placeholder", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = build(args.code_revision, supervision_revision=args.supervision_revision,
                  allow_unpublished_placeholder=args.allow_unpublished_placeholder)
    target = args.output or Path(__file__).resolve().parents[1] / NOTEBOOK_PATH
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
