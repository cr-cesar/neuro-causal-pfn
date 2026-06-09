import numpy as np

from neurocausalpfn.prior.confounding import make_unobserved_confounded
from neurocausalpfn.prior.intersynth import SyntheticDGP, make_dataset
from neurocausalpfn.prior.verify_identifiability import verify_identifiability


def test_accept_ignorable_process():
    rng = np.random.default_rng(0)
    dgp = SyntheticDGP(8, rng, mechanism="mixed")
    data = make_dataset(dgp, n_context=300, n_query=10, rng=rng)
    assert verify_identifiability(data["Tc"], data["Xc"], None, None, U=None)


def test_reject_unobserved_confounding():
    rng = np.random.default_rng(1)
    W, X, Y0, Y1, U = make_unobserved_confounded(8, 400, rng, strength=3.0)
    assert not verify_identifiability(W, X, Y0, Y1, U=U)


def test_reject_positivity_violation():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 8))
    W = np.ones(50)  # todos tratados, sin solapamiento
    assert not verify_identifiability(W, X, None, None, U=None)
