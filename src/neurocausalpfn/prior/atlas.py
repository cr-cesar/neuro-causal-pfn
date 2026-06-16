"""Atlas funcional y sus subdivisiones para InterSynth.

La parcelacion (las 16 redes funcionales derivadas de NeuroQuery) y sus
subdivisiones por transcriptoma (Allen) o receptoma (Hansen) son plantillas de
referencia en espacio MNI. Aqui se cargan desde un directorio de NIfTI y se
exponen las utilidades que necesita el mecanismo de InterSynth: la fraccion de
cada subred cubierta por una lesion, y los centroides de las subredes para
modelar la asignacion de tratamiento con confusion observada.

Si el directorio no existe, se sintetiza un atlas de juguete (regiones
etiquetadas en un volumen pequeno) para que el modulo sea ejecutable y
verificable sin los atlas reales. La logica del mecanismo es identica en ambos
casos; solo cambia la procedencia de las etiquetas.
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np


class FunctionalAtlas:
    def __init__(self, network_labels: np.ndarray,
                 subnetworks: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        # network_labels: volumen entero [D, H, W], 0 = fondo, 1..K = redes.
        # subnetworks[k] = (mascara_subred_A, mascara_subred_B) de la red k.
        self.network_labels = network_labels
        self.subnetworks = subnetworks
        self.shape = tuple(network_labels.shape)
        self.networks = sorted(subnetworks.keys())
        self._centroids = {}
        for k, (a, b) in subnetworks.items():
            self._centroids[(k, 0)] = _centroid(a)
            self._centroids[(k, 1)] = _centroid(b)

    @property
    def n_networks(self) -> int:
        return len(self.networks)

    def subnetwork_overlap(self, lesion: np.ndarray, network: int) -> Tuple[float, float]:
        """Fraccion de cada subred (A, B) de una red cubierta por la lesion."""
        a, b = self.subnetworks[network]
        return _overlap_fraction(lesion, a), _overlap_fraction(lesion, b)

    def centroid(self, network: int, sub: int) -> np.ndarray:
        return self._centroids[(network, sub)]

    @classmethod
    def from_dir(cls, atlas_dir: Optional[str], shape=(48, 56, 48), seed: int = 0,
                 modality: str = "receptor"):
        if atlas_dir and os.path.isdir(atlas_dir):
            return cls._load(atlas_dir, modality=modality)
        return cls.synthetic(shape=shape, seed=seed)

    @classmethod
    def _load(cls, atlas_dir: str, modality: str = "receptor"):
        """Carga el atlas real de Giles.

        Estructura esperada:
        - functional_parcellation_2mm.nii.gz: volumen con las redes etiquetadas
          1..K (0 = fondo).
        - 2mm_parcellations/{modality}/: un archivo por red, con las dos subredes
          codificadas como etiquetas 1 (subred A) y 2 (subred B); el numero al
          inicio del nombre del archivo es el indice de la red.
        modality es 'receptor' (subdivision receptomica, Hansen) o 'genetics'
        (subdivision transcriptomica, Allen)."""
        import glob
        import nibabel as nib

        net_path = None
        for ext in (".nii.gz", ".nii"):
            cand = os.path.join(atlas_dir, "functional_parcellation_2mm" + ext)
            if os.path.exists(cand):
                net_path = cand
                break
        if net_path is None:
            raise FileNotFoundError(
                "no se encontro functional_parcellation_2mm en " + atlas_dir)
        network_labels = np.asarray(nib.load(net_path).get_fdata()).astype(np.int64)

        sub_dir = os.path.join(atlas_dir, "2mm_parcellations", modality)
        files = sorted(glob.glob(os.path.join(sub_dir, "*.nii"))
                       + glob.glob(os.path.join(sub_dir, "*.nii.gz")))
        if not files:
            raise FileNotFoundError("no se encontraron subredes en " + sub_dir)
        subnetworks = {}
        for f in files:
            stem = os.path.basename(f)
            for ext in (".nii.gz", ".nii"):
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            try:
                k = int(stem.split("_")[0])
            except ValueError:
                continue
            vol = np.asarray(nib.load(f).get_fdata())
            subnetworks[k] = ((vol == 1).astype(np.float32), (vol == 2).astype(np.float32))
        return cls(network_labels, subnetworks)

    @classmethod
    def synthetic(cls, shape=(48, 56, 48), n_networks: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        network_labels = np.zeros(shape, dtype=np.int64)
        subnetworks = {}
        zz, yy, xx = np.indices(shape)
        for k in range(1, n_networks + 1):
            center = np.array([int(rng.integers(int(0.25 * s), int(0.75 * s) + 1)) for s in shape])
            radius = np.array([int(rng.integers(int(0.12 * s), int(0.22 * s) + 1)) for s in shape])
            ell = (((zz - center[0]) / radius[0]) ** 2
                   + ((yy - center[1]) / radius[1]) ** 2
                   + ((xx - center[2]) / radius[2]) ** 2) <= 1.0
            network_labels[ell] = k
            axis = int(rng.integers(0, 3))            # subdivide por un plano que pasa por el centroide
            coord = (zz, yy, xx)[axis]
            half = coord <= center[axis]
            subnetworks[k] = ((ell & half).astype(np.float32), (ell & (~half)).astype(np.float32))
        return cls(network_labels, subnetworks)


def _overlap_fraction(lesion: np.ndarray, region: np.ndarray) -> float:
    region_size = float(region.sum())
    if region_size < 1.0:
        return 0.0
    return float((lesion * region).sum()) / region_size


def _centroid(mask: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask > 0.5)
    if len(idx) == 0:
        return np.array(mask.shape, dtype=np.float32) / 2.0
    return idx.mean(axis=0).astype(np.float32)
