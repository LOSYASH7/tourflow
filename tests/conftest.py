from datetime import date

import pytest

from tourflow.config import get_config
from tourflow.features import build_features
from tourflow.sources.synthetic import world_inputs

AS_OF = date(2026, 9, 15)


@pytest.fixture(scope="session")
def cfg():
    return get_config()


@pytest.fixture(scope="session")
def inputs(cfg):
    return world_inputs(cfg, AS_OF)


@pytest.fixture(scope="session")
def features(cfg, inputs):
    return build_features(inputs, cfg)
