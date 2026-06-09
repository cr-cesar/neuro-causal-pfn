"""Ensamblado de lotes para el transformer.

Convierte el lote numpy que produce la cohorte en tensores de torch. Cada
paciente de contexto lleva sus covariables, su tratamiento y su resultado; cada
paciente de consulta lleva solo sus covariables y el tratamiento de interes. La
construccion concreta de los tokens (la concatenacion y la proyeccion lineal) la
realiza el modelo; aqui solo se preparan los tensores.

En modo completo, las proyecciones lineales de este esqueleto se sustituirian
por la codificacion tabular de columna y luego fila descrita en el plan.
"""
from typing import Dict

import torch


def to_tensors(batch_np: Dict[str, "object"], device: str = "cpu",
               dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=dtype, device=device) for k, v in batch_np.items()}
