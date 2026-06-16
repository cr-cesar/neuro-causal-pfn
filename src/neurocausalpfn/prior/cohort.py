"""Cohorte del Neuro-Prior.

Itera sobre procesos generadores de datos sinteticos, los filtra con el
verificador de identificabilidad y apila contextos y consultas en lotes para
entrenar el transformer. Devuelve arreglos numpy; la conversion a tensores se
hace en la capa del modelo para que este modulo no dependa de torch.
"""
from typing import Dict, Sequence

import numpy as np

from .atlas import FunctionalAtlas, _centroid
from .intersynth import MECHANISMS, SyntheticDGP, make_dataset
from .intersynth_atlas import InterSynthDGP, compute_overlaps, make_intersynth_dataset
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


def build_synthetic_lesion_pool(n: int, shape=(48, 56, 48), seed: int = 0) -> np.ndarray:
    """Conjunto de mascaras sinteticas para ejecutar InterSynth sin datos reales.
    En la tuberia real, este conjunto se sustituye por las mascaras de Giles."""
    from ..data.nifti_dataset import LesionMaskDataset

    ds = LesionMaskDataset(root=None, in_shape=shape, n_synth=n, seed=seed)
    return np.stack([ds._load_volume(i) for i in range(n)], axis=0)


class NeuroPriorInterSynth:
    """Cohorte del Neuro-Prior con sustrato anatomico (InterSynth).

    Precomputa, una sola vez, los solapamientos y centroides de cada lesion del
    conjunto. Cada lote muestrea una condicion (un proceso del prior) y, a partir
    de indices del conjunto, construye contexto y consultas con resultados
    potenciales conocidos. El covariable X es el latente del encoder si se
    proporciona z_pool; en su defecto, las fracciones de solapamiento observadas
    mas el centroide normalizado, que es una covariable puramente observada y por
    tanto mantiene la ignorabilidad por construccion.
    """

    def __init__(self, atlas: FunctionalAtlas, lesion_pool: np.ndarray, seed: int = 0,
                 z_pool=None, n_context: int = 128, n_query: int = 16,
                 unobserved_strength: float = 0.0):
        self.atlas = atlas
        self.rng = np.random.default_rng(seed)
        self.n_context = int(n_context)
        self.n_query = int(n_query)
        self.unobserved_strength = float(unobserved_strength)
        self.m = len(lesion_pool)

        self.overlaps = np.stack([compute_overlaps(atlas, lesion_pool[i]) for i in range(self.m)], axis=0)  # [m, K, 2]
        self.centroids = np.stack([_centroid(lesion_pool[i]) for i in range(self.m)], axis=0)               # [m, 3]
        if z_pool is not None:
            self.X = np.asarray(z_pool, dtype=np.float64)
        else:
            geo = self.centroids / np.array(atlas.shape, dtype=np.float64)
            self.X = np.concatenate([self.overlaps.reshape(self.m, -1), geo], axis=1)                        # [m, 2K + 3]
        self.d_x = int(self.X.shape[1])

    def _indices(self, n: int) -> np.ndarray:
        return self.rng.integers(0, self.m, size=n)

    def _one(self, n_context: int) -> Dict[str, np.ndarray]:
        for _ in range(64):
            dgp = InterSynthDGP(self.atlas, self.rng, unobserved_strength=self.unobserved_strength)
            ci, qi = self._indices(n_context), self._indices(self.n_query)
            k = dgp.network_idx
            data = make_intersynth_dataset(
                dgp,
                self.overlaps[ci, k, :], self.centroids[ci], self.X[ci],
                self.overlaps[qi, k, :], self.centroids[qi], self.X[qi],
                self.rng)
            if verify_identifiability(data["Tc"], data["Xc"], None, None, U=None):
                return data
        return data

    def sample_batch(self, batch_size: int, n_context=None) -> Dict[str, np.ndarray]:
        n_ctx = self.n_context if n_context is None else int(n_context)
        items = [self._one(n_ctx) for _ in range(batch_size)]
        keys = ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1")
        return {k: np.stack([it[k] for it in items], axis=0) for k in keys}
