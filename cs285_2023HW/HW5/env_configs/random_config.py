


import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from typing import Optional

def random_agent_config(
    env_name: str,
    exp_name: Optional[str] = None,
    **kwargs
):
    log_string = "{env_name}_random".format(
        env_name=env_name
    )

    return {
        "agent": "random",
        "agent_kwargs": {},
        "make_env": lambda: RecordEpisodeStatistics(TimeLimit(gym.make(env_name), 100)),
        "env_name": env_name,
        "log_name": log_string,
        "batch_size": 0,
        **kwargs,
    }