import logging

import pytest

from tests.common import BASIC_IDS, build_and_run_step

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.parametrize("gym_id", BASIC_IDS)
def test_intro_competition_envs(gym_id):
    build_and_run_step(gym_id, pfrl_2019=True, pfrl_2019_config={})
    logging.debug("Finished test!")
