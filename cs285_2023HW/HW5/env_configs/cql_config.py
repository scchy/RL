
from .dqn_basic_config import basic_dqn_config


def cql_config(
    cql_alpha: float = 1.0,
    cql_temperature: float = 1.0,
    total_steps: int = 50000,
    discount: float = 0.95,
    **kwargs,
):
    config = basic_dqn_config(total_steps=total_steps, discount=discount, **kwargs)
    config["log_name"] = "{env_name}_cql{cql_alpha}".format(
        env_name=config["env_name"], cql_alpha=cql_alpha
    )
    config["agent"] = "cql"

    config["agent_kwargs"]["cql_alpha"] = cql_alpha
    config["agent_kwargs"]["cql_temperature"] = cql_temperature

    return config