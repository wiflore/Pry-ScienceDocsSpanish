"""Traducción inglés→español con MarianMT (Helsinki-NLP/opus-mt-en-es)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_ID = "Helsinki-NLP/opus-mt-en-es"


class MarianTranslator:
    """Wrapper para traducción en lote con MarianMT."""

    def __init__(self, device: str = "cpu", batch_size: int = 32, max_length: int = 512):
        """
        Args:
            device: 'cpu', 'cuda' o 'mps'. MarianMT se ejecuta en CPU por estabilidad.
            batch_size: tamaño de lote para traducción.
            max_length: tokens máximos de salida.
        """
        # MarianMT es estable en CPU; no requiere MPS
        self.device = "cpu"
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info("Cargando MarianMT tokenizer desde %s", MODEL_ID)
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)
        logger.info("Cargando MarianMT model desde %s", MODEL_ID)
        self.model = MarianMTModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Traduce una lista de textos inglés→español."""
        # Truncar al tokenizador para evitar error por texto largo
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            translated_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )

        return self.tokenizer.batch_decode(translated_ids, skip_special_tokens=True)

    def translate_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        output_column: str = "text_es",
        save_path: str | Path | None = None,
        force_retranslate: bool = False,
    ) -> Dataset:
        """
        Traduce todos los textos de un Dataset y agrega columna con la traducción.

        Args:
            dataset: Dataset con columna de texto en inglés.
            text_column: nombre de la columna fuente.
            output_column: nombre de la columna destino en español.
            save_path: si se indica, guarda el dataset traducido en disco.
            force_retranslate: si False y save_path existe, carga desde disco.

        Returns:
            Dataset con columna output_column agregada.
        """
        if save_path is not None:
            save_path = Path(save_path)
            if save_path.exists() and not force_retranslate:
                logger.info("Cargando traducción desde caché: %s", save_path)
                return Dataset.load_from_disk(str(save_path))

        texts = dataset[text_column]
        translated: List[str] = []

        logger.info(
            "Traduciendo %d textos en lotes de %d...", len(texts), self.batch_size
        )
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Traduciendo"):
            batch = texts[start : start + self.batch_size]
            translated.extend(self.translate_batch(batch))

        result = dataset.add_column(output_column, translated)

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            result.save_to_disk(str(save_path))
            logger.info("Dataset traducido guardado en %s", save_path)

        return result
