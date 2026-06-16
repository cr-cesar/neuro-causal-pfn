"""InterSynth con sustrato anatomico (la version fiel del mecanismo).

A diferencia del generador ligero de intersynth.py (que muestrea covariables
gaussianas desde cero), este modulo cruza cada lesion con la parcelacion
funcional para fabricar la verdad de terreno semi-sintetica, siguiendo el marco
de Giles:

- Deficit: una lesion que cubre al menos el 5% de una subred causa el deficit
  correspondiente.
- Susceptibilidad verdadera al tratamiento: depende de cual de las dos subredes
  (transcriptomica o receptomica) de la red causal toca predominantemente la
  lesion. Esta susceptibilidad nunca se le muestra al modelo, es solo verdad de
  terreno.
- Desenlace: se combinan un efecto del tratamiento (TE) y una recuperacion
  espontanea (RE) como variables de Bernoulli, lo que da resultados potenciales
  esperados condicionales conocidos mu_0 y mu_1, y por tanto un CATE conocido.
- Asignacion de tratamiento: confusion observada por la distancia del centroide
  de la lesion a la subred sensible (hiperparametro b); opcionalmente confusion
  no observada por fuga desde la susceptibilidad real (que viola la
  identificabilidad y debe rechazar el verificador).

Cada condicion muestreada (red causal, mapeo subred a tratamiento, TE, RE, sesgo
de asignacion) es un proceso del prior. El covariable X que ve el transformer es
la representacion de la lesion: el latente del encoder congelado si se
proporciona, o, en su defecto, las fracciones de solapamiento observadas.
"""
import numpy as np

from .atlas import FunctionalAtlas, _centroid


def compute_overlaps(atlas: FunctionalAtlas, lesion: np.ndarray) -> np.ndarray:
    """Fracciones de solapamiento [K, 2] de la lesion con las dos subredes de
    cada red del atlas."""
    out = np.zeros((atlas.n_networks, 2), dtype=np.float32)
    for i, k in enumerate(atlas.networks):
        out[i] = atlas.subnetwork_overlap(lesion, k)
    return out


class InterSynthDGP:
    """Una condicion (un proceso del prior) muestreada del marco InterSynth."""

    def __init__(self, atlas: FunctionalAtlas, rng: np.random.Generator,
                 unobserved_strength: float = 0.0):
        self.atlas = atlas
        self.network_idx = int(rng.integers(0, atlas.n_networks))
        self.network = atlas.networks[self.network_idx]
        self.optimal_for_A = int(rng.integers(0, 2))       # subred A responde a este tratamiento
        self.p_te = float(rng.uniform(0.4, 0.8))           # prob de efecto del tratamiento
        self.p_re = float(rng.uniform(0.1, 0.4))           # prob de recuperacion espontanea
        self.bias_b = float(rng.uniform(0.5, 2.0))         # fuerza de la confusion observada
        self.unobserved_strength = float(unobserved_strength)
        self.deficit_threshold = 0.05
        self._scale = float(np.prod(atlas.shape) ** (1.0 / 3.0))

    def susceptibility(self, oa: float, ob: float):
        """0 si la lesion toca predominantemente la subred A, 1 si la B, None si
        no toca la red (no diferenciable, el tratamiento no cambia el desenlace)."""
        if max(oa, ob) < self.deficit_threshold:
            return None
        return 0 if oa >= ob else 1

    def optimal_treatment(self, s):
        if s is None:
            return None
        return self.optimal_for_A if s == 0 else (1 - self.optimal_for_A)

    def mu(self, s, t: int) -> float:
        """Resultado potencial esperado condicional: probabilidad de buen
        desenlace bajo el tratamiento t. Combina TE (si el tratamiento es el
        adecuado para la susceptibilidad) y RE (recuperacion espontanea)."""
        works = (s is not None) and (t == self.optimal_treatment(s))
        p_treat = self.p_te if works else 0.0
        return 1.0 - (1.0 - p_treat) * (1.0 - self.p_re)

    def cate(self, s) -> float:
        return self.mu(s, 1) - self.mu(s, 0)

    def propensity(self, lesion_centroid: np.ndarray, s) -> float:
        """Probabilidad de recibir el tratamiento 1. La confusion observada
        depende solo de la geometria de la lesion (su centroide); la fuga no
        observada, opcional, depende de la susceptibilidad real."""
        ca = self.atlas.centroid(self.network, 0)
        cb = self.atlas.centroid(self.network, 1)
        d = float(np.linalg.norm(lesion_centroid - ca) - np.linalg.norm(lesion_centroid - cb))
        signal = -self.bias_b * d / self._scale
        if self.unobserved_strength > 0.0 and s is not None:
            signal += self.unobserved_strength * (1.0 if s == 1 else -1.0)
        return float(1.0 / (1.0 + np.exp(-signal)))


def make_intersynth_dataset(dgp: InterSynthDGP,
                            overlaps_ctx: np.ndarray, cent_ctx: np.ndarray, X_ctx: np.ndarray,
                            overlaps_qry: np.ndarray, cent_qry: np.ndarray, X_qry: np.ndarray,
                            rng: np.random.Generator):
    """Dataset observacional de contexto y conjunto de consultas con los
    resultados potenciales conocidos. overlaps_* son [n, 2] (las dos subredes de
    la red causal del proceso); cent_* son [n, 3]; X_* son [n, d_x]."""
    s_ctx = [dgp.susceptibility(o[0], o[1]) for o in overlaps_ctx]
    e = np.array([dgp.propensity(cent_ctx[i], s_ctx[i]) for i in range(len(s_ctx))])
    Tc = (rng.uniform(size=len(s_ctx)) < e).astype(np.float64)
    mu_assigned = np.array([dgp.mu(s_ctx[i], int(Tc[i])) for i in range(len(s_ctx))])
    Yc = (rng.uniform(size=len(s_ctx)) < mu_assigned).astype(np.float64)

    s_qry = [dgp.susceptibility(o[0], o[1]) for o in overlaps_qry]
    Tq = rng.integers(0, 2, size=len(s_qry)).astype(np.float64)
    mu0 = np.array([dgp.mu(s, 0) for s in s_qry])
    mu1 = np.array([dgp.mu(s, 1) for s in s_qry])
    mu_q = np.where(Tq == 1, mu1, mu0)

    return {"Xc": np.asarray(X_ctx, np.float64), "Tc": Tc, "Yc": Yc,
            "Xq": np.asarray(X_qry, np.float64), "Tq": Tq,
            "mu_q": mu_q, "mu0": mu0, "mu1": mu1}
