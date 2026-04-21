"""Detección automática del dispositivo de cómputo: MPS / CUDA / CPU."""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Retorna el mejor dispositivo disponible con soporte para float32.

    Orden de preferencia: MPS (Apple Silicon) → CUDA → CPU.
    Para MPS se recomienda definir la variable de entorno
    PYTORCH_ENABLE_MPS_FALLBACK=1 antes de ejecutar.

    Returns:
        torch.device listo para usar.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        # Asegurar fallback para operaciones no soportadas en MPS
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        logger.info("Dispositivo seleccionado: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "Dispositivo seleccionado: CUDA (%s)", torch.cuda.get_device_name(0)
        )
    else:
        device = torch.device("cpu")
        logger.info("Dispositivo seleccionado: CPU")

    return device


def device_info(device: torch.device) -> dict:
    """Retorna información básica sobre el dispositivo."""
    info = {"device": str(device), "pytorch_version": torch.__version__}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    elif device.type == "mps":
        info["mps_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()
    return info
