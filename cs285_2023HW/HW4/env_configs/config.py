# python3
# Create Date: 2025-08-10
# Func: 
#   from scripting_utils import make_config
# =============================================================================
import yaml 
from .mpc_config import  mpc_config
from .sac_config import sac_config


configs = {
    "mpc": mpc_config,
    "sac": sac_config,
}



def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    base_config_name = config_kwargs.pop("base_config")
    return configs[base_config_name](**config_kwargs)


