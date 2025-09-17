# python3
# Create Date: 2025-08-10
# Func: 
#   from scripting_utils import make_config
# =============================================================================
import yaml 
from .dqn_basic_config import basic_dqn_config
from .random_config import random_agent_config
from .rnd_config import rnd_config
# from .cql_config import cql_config
from .awac_config import awac_config
from .iql_config import iql_config


configs = {
    "dqn": basic_dqn_config,
    "random": random_agent_config,
    "rnd": rnd_config,
    # "cql": cql_config,
    "awac": awac_config,
    "iql": iql_config,
}


def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    base_config_name = config_kwargs.pop("base_config")
    return configs[base_config_name](**config_kwargs)


