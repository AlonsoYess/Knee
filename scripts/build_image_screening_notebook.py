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
        "**Revisión 2.0 · Plan metodológico v1.2 · GPU T4.**\n\n"
        "La campaña contiene **25 trabajos**; se ejecutan solo los pendientes, en este orden: **ViT-B/16, ConvNeXtV2-Tiny, Skelex, "
        "BiomedCLIP y MambaOut-Tiny**; cinco pliegues por codificador, con semilla 2026. "
        "Los codificadores permanecen congelados: solo se entrena la cabeza de predicción. "
        "Cada pliegue nuevo comienza desde el checkpoint externo registrado. Si existe un guardado compatible de una época "
        "de ese mismo pliegue, se restaura sin reutilizar pesos de otro pliegue.\n\n"
        "Se mantienen el conjunto de desarrollo, las particiones por participante, las entradas de 224 × 224 "
        "y el aumento gamma v1.2 únicamente en entrenamiento. La validación conserva su transformación "
        "determinista y cada codificador utiliza su normalización registrada.\n\n"
        "Se necesitan los registros completos de las libretas **33 y 38** ya guardados en Drive. "
        "Se autentican automáticamente; no tienes que volver a ejecutarlas. Revisa las tres vistas del aumento "
        "de la libreta 38 antes de iniciar la campaña. Al ejecutar la celda 4, se comprobarán los pesos y un forward "
        "sintético en GPU únicamente de los codificadores pendientes. Se recuperan las pruebas breves compatibles ya "
        "guardadas; solo si faltan se ejecutan dos actualizaciones en el pliegue 0 de desarrollo para verificar gradientes "
        "y recursos con la receta "
        "aprobada. Estos cinco controles se registran aparte, no participan en la selección del modelo y "
        "sus pesos no se reutilizan en los 25 trabajos completos.\n\n"
        "Cada trabajo completo se guarda y verifica en Drive antes de continuar. Una reconexión recupera "
        "los trabajos completos compatibles guardados por la revisión anterior o esta revisión, sin copiar sus pesos. "
        "Los trabajos nuevos guardan al final de cada época el estado necesario para retomarse después de un corte. "
        "Una época interrumpida antes del guardado debe repetirse. El progreso dentro de un pliegue antiguo no guardado "
        "no se puede recuperar. Ejecuta una sola sesión de esta campaña.\n\n"
        "DenseNet121 ya tiene su campaña en la libreta 40 y no se vuelve a entrenar aquí. DINOv3 permanece "
        "fuera por acceso pendiente. Esta libreta no descongela codificadores, no ejecuta fusión multimodal, "
        "calibración, selección de umbral ni ajuste final. Los originales y las particiones se conservan; "
        "la prueba reservada permanece cerrada. **Esta campaña ya fue autorizada. El entrenamiento comienza "
        "al ejecutar la celda 4; no tienes que escribir rutas ni repetir una frase de autorización.**")
    bootstrap = value["cells"][1]["source"]
    bootstrap = _replace_once(bootstrap, "knee_methodology40_", "knee_image_screening41_")
    bootstrap = _replace_once(bootstrap, "CAMPAIGN = 'densenet_frozen'", "CAMPAIGN = 'remaining_encoders_frozen'")
    bootstrap = _replace_once(bootstrap, "from knee import methodology_campaign_colab as orchestration", "from knee import image_campaign_recovery as orchestration")
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
        "Las carpetas de registros privados y desarrollo se leen de la configuración privada guardada en Drive. "
        "No es necesario escribir rutas. Si esa configuración no es válida, la celda se detiene sin buscar otras carpetas. "
        "Se autentican las comprobaciones 33 y 38 y la integridad del paquete de desarrollo. "
        "Esta celda no descarga ni abre el paquete reservado de prueba.")
    locate = value["cells"][7]["source"]
    start = locate.index("PERSISTENT_ROOT = orchestration.checked_path(input(")
    stop = locate.index("proof = json.loads(checked_run(")
    locate = locate[:start] + """MY_DRIVE_ROOT = Path('/content/drive/MyDrive').resolve()
PROJECT_ROOT = orchestration.checked_path(os.environ.get('KNEE_DATA_ROOT') or MY_DRIVE_ROOT / 'KneeOAI').resolve()
if not PROJECT_ROOT.is_dir() or not PROJECT_ROOT.is_relative_to(MY_DRIVE_ROOT):
    raise RuntimeError('No se encontró la carpeta privada configurada dentro de tu Drive montado.')
SETTINGS_PATH = orchestration.checked_path(PROJECT_ROOT / 'colab_runtime_paths.json')
if not SETTINGS_PATH.is_file() or not 0 < SETTINGS_PATH.stat().st_size <= 16384:
    raise RuntimeError('Falta la configuración privada de las carpetas de Colab.')
settings = orchestration._json(SETTINGS_PATH)
if not isinstance(settings, dict) or set(settings) != {'version', 'records_relative', 'development_relative'} or settings.get('version') != 'knee_colab_runtime_paths_v1.0':
    raise RuntimeError('La configuración privada de carpetas no tiene el formato esperado.')
selected_paths = []
for key in ('records_relative', 'development_relative'):
    relative = settings[key]
    if not isinstance(relative, str) or not relative or len(relative) > 1024 or '\\\\' in relative or ':' in relative:
        raise RuntimeError('La configuración debe contener rutas relativas válidas.')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(part in ('', '.', '..') for part in relative.split('/')):
        raise RuntimeError('La configuración no puede salir de la carpeta del proyecto.')
    selected = orchestration.checked_path(PROJECT_ROOT.joinpath(*parts.parts)).resolve()
    if not selected.is_dir() or not selected.is_relative_to(PROJECT_ROOT):
        raise RuntimeError('Una de las carpetas privadas configuradas no existe dentro del proyecto.')
    selected_paths.append(selected)
PERSISTENT_ROOT, REMOTE_PACKAGE_DIR = selected_paths
if PERSISTENT_ROOT.is_relative_to(REMOTE_PACKAGE_DIR) or REMOTE_PACKAGE_DIR.is_relative_to(PERSISTENT_ROOT):
    raise RuntimeError('Las carpetas de registros y desarrollo deben estar separadas.')
if not orchestration.checked_path(REMOTE_PACKAGE_DIR / 'transport.json').is_file():
    raise RuntimeError('La carpeta de desarrollo configurada no contiene transport.json.')
print('Carpetas privadas localizadas automáticamente. No se inició entrenamiento.')
""" + locate[stop:]
    value["cells"][7]["source"] = locate
    value["cells"][8]["source"] = (
        "## 4. Iniciar la campaña autorizada: cinco pruebas breves y 25 trabajos\n\n"
        "Orden: **ViT-B/16 → ConvNeXtV2-Tiny → Skelex → BiomedCLIP → MambaOut-Tiny**. "
        "Se reconocen y omiten los trabajos completos guardados. Solo se verifican en GPU los codificadores pendientes, "
        "reutilizando sus pruebas breves ya guardadas cuando correspondan. Después se completan únicamente los pliegues pendientes, "
        "siempre con el codificador congelado y aumento gamma v1.2. Un trabajo pendiente retoma su última época guardada "
        "si existe; de lo contrario comienza desde los pesos externos registrados. Los controles breves no aportan "
        "pesos ni métricas a esa comparación. "
        "El progreso identifica modelo y pliegue. Los checkpoints grandes requieren espacio local y tiempo "
        "de copia a Drive. Cada trabajo se guarda y verifica antes de continuar.\n\n"
        "**Al ejecutar esta celda se retoma la campaña que ya autorizaste; no se solicita otra frase.** "
        "La autorización se limita a los cinco codificadores y los cinco pliegues descritos. Si Colab se desconecta, "
        "vuelve a abrir esta misma revisión y ejecuta las celdas: se recuperan los trabajos completos compatibles.")
    train = value["cells"][9]["source"]
    train = _replace_once(train,
        "confirmation = input('Escribe ' + orchestration.CONFIRMATIONS[CAMPAIGN] + ' para ejecutar solo esta campaña: ').strip()\n"
        "if confirmation != orchestration.CONFIRMATIONS[CAMPAIGN]:\n"
        "    raise RuntimeError('No se confirmó la campaña. No se inició entrenamiento.')",
        "if CAMPAIGN != 'remaining_encoders_frozen':\n"
        "    raise RuntimeError('Esta libreta no autoriza una campaña distinta.')")
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
        "Buscando trabajos completos para omitirlos; se entrenarán solo los pendientes…")
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
