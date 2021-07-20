import logging

import pytest

from minerl_wrappers.utils import load_means
from tests.common import DIAMOND_COMPETITION_IDS, build_and_run_step

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.parametrize("gym_id", DIAMOND_COMPETITION_IDS)
def test_diamond_competition_envs(gym_id):
    logging.debug("Loading kmeans")
    means = load_means()
    build_and_run_step(
        gym_id, pfrl_2020=True, pfrl_2020_config={"action_choices": means}
    )
    logging.debug("Finished test!")
