"""Cohorte del Neuro-Prior.

Itera sobre procesos generadores de datos sinteticos, los filtra con el
verificador de identificabilidad y apila contextos y consultas en lotes para
entrenar el transformer. Devuelve arreglos numpy; la conversion a tensores se
hace en la capa del modelo para que este modulo no dependa de torch.
"""
from typing import Dict, Sequence

import numpy as np

from .intersynth import MECHANISMS, SyntheticDGP, make_dataset
from .verify_identifiability import verify_identifiability


class NeuroPrior:
    def __init__(self, d_x: int, n_context: int, n_query: int, seed: int = 0,
                 mechanisms: Sequence[str] = MECHANISMS):
        self.d_x = int(d_x)
        self.n_context = int(n_context)
        self.n_query = int(n_query)
        self.rng = np.random.default_rng(seed)
        self.mechanisms = tuple(mechanisms)

    def _one(self) -> Dict[str, np.ndarray]:
        # reintenta hasta obtener un proceso que pasa la puerta R1/R2.
        # Los procesos son ignorables por construccion (U=None); el filtro
        # rechaza sobre todo violaciones de positividad en la muestra.
        for _ in range(64):
            mech = str(self.rng.choice(self.mechanisms))
            dgp = SyntheticDGP(self.d_x, self.rng, mechanism=mech)
            data = make_dataset(dgp, self.n_context, self.n_query, self.rng)
            if verify_identifiability(data["Tc"], data["Xc"], None, None, U=None):
                return data
        return data  # devuelve el ultimo si el reintento se agota

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        items = [self._one() for _ in range(batch_size)]
        keys = ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1")
        return {k: np.stack([it[k] for it in items], axis=0) for k in keys}
