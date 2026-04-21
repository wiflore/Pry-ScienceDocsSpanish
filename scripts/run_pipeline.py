#!/usr/bin/env python3
"""
Pipeline principal de clasificación retórica IMRaD (Etapa 1).

Uso:
    python scripts/run_pipeline.py [--skip-download] [--skip-translate]
                                   [--skip-train] [--models scibeto qwen_zero qwen_few]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Agregar raíz del proyecto al PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.config import (
    load_data_config,
    load_labels_config,
    load_models_config,
    resolve_path,
    validate_configs,
)
from src.data.download import download_pubmed_rct
from src.data.preprocessing import map_pubmed_labels_to_imrad, preprocess_dataset, sample_balanced
from src.data.splitting import create_splits
from src.data.translation import MarianTranslator
from src.evaluation.metrics import compare_models, compute_metrics
from src.evaluation.visualization import plot_confusion_matrix, plot_model_comparison
from src.models.device import device_info, get_device
from src.models.encoder import SciBETOClassifier
from src.models.llm_prompting import QwenPrompter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline retórico IMRaD — Etapa 1")
    parser.add_argument("--skip-download", action="store_true", help="Omitir descarga del dataset")
    parser.add_argument("--skip-translate", action="store_true", help="Omitir traducción")
    parser.add_argument("--skip-train", action="store_true", help="Omitir entrenamiento SciBETO")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["scibeto", "qwen_zero", "qwen_few"],
        choices=["scibeto", "qwen_zero", "qwen_few"],
        help="Modelos a evaluar",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validar configuración ────────────────────────────────────────────────
    logger.info("Validando configuraciones...")
    validate_configs()

    data_cfg = load_data_config()
    labels_cfg = load_labels_config()
    models_cfg = load_models_config()

    raw_dir = resolve_path(data_cfg["data"]["paths"]["raw"])
    translated_dir = resolve_path(data_cfg["data"]["paths"]["translated"])
    splits_dir = resolve_path(data_cfg["data"]["paths"]["splits"])
    reports_dir = resolve_path("reports/entrega3")
    reports_dir.mkdir(parents=True, exist_ok=True)

    seed = data_cfg["data"]["random_seed"]
    examples_per_label = data_cfg["data"]["examples_per_imrad_label"]

    device = get_device()
    logger.info("Dispositivo: %s", device_info(device))

    # ── Paso 1: Descarga ─────────────────────────────────────────────────────
    logger.info("=== PASO 1: Descarga PubMed RCT ===")
    dataset = download_pubmed_rct(
        raw_dir=raw_dir,
        split=data_cfg["data"]["source_split"],
        force_redownload=False,
    )
    logger.info("Dataset cargado: %d ejemplos", len(dataset))

    # ── Paso 2: Preprocesamiento y muestreo ──────────────────────────────────
    logger.info("=== PASO 2: Preprocesamiento ===")
    dataset = map_pubmed_labels_to_imrad(dataset)
    dataset = preprocess_dataset(dataset)
    dataset = sample_balanced(dataset, examples_per_label=examples_per_label, seed=seed)

    # ── Paso 3: Traducción ───────────────────────────────────────────────────
    logger.info("=== PASO 3: Traducción MarianMT ===")
    translator = MarianTranslator(batch_size=data_cfg["data"]["translation"]["batch_size"])
    dataset = translator.translate_dataset(
        dataset,
        text_column="text",
        output_column="text_es",
        save_path=translated_dir / "balanced_translated",
        force_retranslate=False,
    )

    # ── Paso 4: Splits ───────────────────────────────────────────────────────
    logger.info("=== PASO 4: Creación de splits ===")
    splits = create_splits(
        dataset,
        train_ratio=data_cfg["data"]["train_ratio"],
        val_ratio=data_cfg["data"]["val_ratio"],
        seed=seed,
        save_dir=splits_dir / "imrad_splits",
    )

    results = {}

    # ── Paso 5: SciBETO ──────────────────────────────────────────────────────
    if "scibeto" in args.models:
        logger.info("=== PASO 5: Fine-tuning SciBETO-large ===")
        scibeto_cfg = models_cfg["models"]["scibeto-large"]
        classifier = SciBETOClassifier(
            device=device,
            batch_size=scibeto_cfg["batch_size"],
            learning_rate=scibeto_cfg["learning_rate"],
            num_epochs=scibeto_cfg["num_epochs"],
            warmup_ratio=scibeto_cfg["warmup_ratio"],
            weight_decay=scibeto_cfg["weight_decay"],
            output_dir=resolve_path("models/scibeto-large-imrad"),
        )
        if not args.skip_train:
            classifier.train(splits)
        else:
            logger.info("Cargando SciBETO desde disco...")
            classifier = SciBETOClassifier.load(
                resolve_path("models/scibeto-large-imrad"), device
            )

        test_texts = splits["test"]["text_es"]
        test_labels = splits["test"]["imrad_label"]
        preds = classifier.predict(test_texts)
        metrics = compute_metrics(test_labels, preds)
        results["SciBETO-large (fine-tuned)"] = metrics
        logger.info("SciBETO — macro_f1=%.4f", metrics["macro_f1"])

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title="SciBETO-large — Matriz de Confusión (test)",
            save_path=reports_dir / "scibeto_confusion_matrix.png",
        )

    # ── Paso 6: Qwen zero-shot ───────────────────────────────────────────────
    if "qwen_zero" in args.models:
        logger.info("=== PASO 6: Qwen2.5:3b — Zero-shot ===")
        prompter = QwenPrompter(
            prompt_mode="zero_shot",
            prompts_dir=resolve_path("configs/prompts"),
        )
        test_texts = splits["test"]["text_es"]
        test_labels = splits["test"]["imrad_label"]
        preds = prompter.predict(test_texts)
        metrics = compute_metrics(test_labels, preds)
        results["Qwen2.5:3b (zero-shot)"] = metrics
        logger.info("Qwen zero-shot — macro_f1=%.4f", metrics["macro_f1"])

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title="Qwen2.5:3b Zero-shot — Matriz de Confusión (test)",
            save_path=reports_dir / "qwen_zero_confusion_matrix.png",
        )

    # ── Paso 7: Qwen few-shot ────────────────────────────────────────────────
    if "qwen_few" in args.models:
        logger.info("=== PASO 7: Qwen2.5:3b — Few-shot ===")
        prompter_few = QwenPrompter(
            prompt_mode="few_shot",
            prompts_dir=resolve_path("configs/prompts"),
        )
        test_texts = splits["test"]["text_es"]
        test_labels = splits["test"]["imrad_label"]
        preds = prompter_few.predict(test_texts)
        metrics = compute_metrics(test_labels, preds)
        results["Qwen2.5:3b (few-shot)"] = metrics
        logger.info("Qwen few-shot — macro_f1=%.4f", metrics["macro_f1"])

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title="Qwen2.5:3b Few-shot — Matriz de Confusión (test)",
            save_path=reports_dir / "qwen_few_confusion_matrix.png",
        )

    # ── Paso 8: Comparación ──────────────────────────────────────────────────
    if results:
        logger.info("=== PASO 8: Comparación de modelos ===")
        table = compare_models(results)
        print("\n" + table + "\n")

        plot_model_comparison(
            results,
            metric="macro_f1",
            save_path=reports_dir / "model_comparison_macro_f1.png",
        )

        # Guardar resultados en JSON
        results_path = reports_dir / "results.json"
        # Eliminar classification_report del JSON (no es serializable limpiamente)
        json_results = {
            k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
            for k, v in results.items()
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        logger.info("Resultados guardados en %s", results_path)

    logger.info("Pipeline completado exitosamente.")


if __name__ == "__main__":
    main()
