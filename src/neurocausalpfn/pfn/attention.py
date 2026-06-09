"""Enmascaramiento de atencion asimetrico de la Etapa 2.

Tanto los tokens de contexto como los de consulta atienden solo al contexto.
Esto vuelve mutuamente independientes las predicciones por consulta y concuerda
con la semantica predictiva posterior: cada consulta se resuelve a partir del
contexto, no de las demas consultas.
"""
import torch


def context_only_mask(n_context: int, n_total: int,
                      device=None, dtype=torch.float32) -> torch.Tensor:
    """Mascara aditiva [n_total, n_total]. Las columnas de consulta se bloquean
    con menos infinito, de modo que ningun token puede atender a una posicion de
    consulta. Las primeras n_context columnas quedan a cero, asi que ninguna fila
    es enteramente menos infinito y el softmax permanece estable."""
    mask = torch.zeros(n_total, n_total, device=device, dtype=dtype)
    mask[:, n_context:] = float("-inf")
    return mask
