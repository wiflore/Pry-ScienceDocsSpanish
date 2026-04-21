"""Carga de configuraciones YAML y rutas del proyecto."""

from pathlib import Path
import yaml

# Directorio raíz del proyecto (dos niveles arriba de src/config.py)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _cargar_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels_config() -> dict:
    return _cargar_yaml(CONFIGS_DIR / "labels.yaml")


def load_models_config() -> dict:
    return _cargar_yaml(CONFIGS_DIR / "models.yaml")


def load_data_config() -> dict:
    return _cargar_yaml(CONFIGS_DIR / "data.yaml")


def load_prompt(nombre: str) -> str:
    """Carga una plantilla de prompt desde configs/prompts/ (sin extensión)."""
    return (CONFIGS_DIR / "prompts" / f"{nombre}.txt").read_text(encoding="utf-8")


def get_label_id2name(labels_config: dict) -> dict:
    return {l["id"]: l["name"] for l in labels_config["labels"]}


def get_label_name2id(labels_config: dict) -> dict:
    return {l["name"]: l["id"] for l in labels_config["labels"]}


def get_pubmed_to_imrad_mapping(labels_config: dict) -> dict:
    return {int(k): int(v) for k, v in labels_config["pubmed_label_mapping"].items()}


def resolve_path(ruta: str) -> Path:
    return PROJECT_ROOT / ruta
