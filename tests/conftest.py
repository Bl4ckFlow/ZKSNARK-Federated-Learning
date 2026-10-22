"""Shared fixtures.

Anything that needs the compiled circuit is marked ``circuit`` and skipped when
the artifacts are absent, so ``pytest`` is useful on a machine without circom
while ``make test`` still exercises the cryptography.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zkfl.config import FixedPointConfig, ProtocolConfig  # noqa: E402
from zkfl.crypto import Identity, PeerRegistry  # noqa: E402
from zkfl.model import Model  # noqa: E402
from zkfl.trainer import LocalDataset  # noqa: E402
from zkfl.zk import CircuitArtifacts, Groth16Backend, NullProofBackend  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "circuit: needs the compiled Groth16 circuit")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


@pytest.fixture(scope="session")
def fp() -> FixedPointConfig:
    return FixedPointConfig()


@pytest.fixture(scope="session")
def protocol() -> ProtocolConfig:
    return ProtocolConfig(n_nodes=10, f_byzantine=3, collect_window=0.05,
                          strict_zk=False)


@pytest.fixture(scope="session")
def circuit_available(fp) -> bool:
    return not CircuitArtifacts.for_config(fp).missing()


@pytest.fixture(scope="session")
def groth16(fp, circuit_available):
    if not circuit_available:
        pytest.skip("circuit artifacts absent; run scripts/setup_circuit.py")
    backend = Groth16Backend(fp)
    yield backend
    backend.close()


@pytest.fixture
def null_backend():
    return NullProofBackend(i_understand_this_is_not_a_proof=True)


@pytest.fixture
def identities() -> list[Identity]:
    return [Identity.from_seed(bytes([i]) * 32) for i in range(10)]


@pytest.fixture
def dataset(fp) -> LocalDataset:
    from zkfl.datasets import synthetic

    split = synthetic(n_samples=64, n_features=fp.n_features, seed=3)
    return LocalDataset(split.x_train.tolist(), split.y_train.tolist(), fp)


@pytest.fixture
def registry(identities, dataset) -> PeerRegistry:
    reg = PeerRegistry()
    for identity in identities:
        payload = PeerRegistry.registration_payload(
            identity.node_id, dataset.root, dataset.n_samples
        )
        reg.register(identity.node_id, dataset.root, dataset.n_samples,
                     identity.sign("registration", payload))
    return reg


@pytest.fixture
def genesis(fp) -> Model:
    return Model.zeros(fp)
