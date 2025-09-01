from gymnasium.envs.registration import register

def register_envs():
    register(
        id='cheetah-cs285-v0',
        entry_point='env_configs.env.cheetah:HalfCheetahEnv',
        max_episode_steps=1000,
    )
    register(
        id='obstacles-cs285-v0',
        entry_point='env_configs.env.obstacles:Obstacles',
        kwargs={'max_episode_steps': 500},
        max_episode_steps=500,
    )
    register(
        id='reacher-cs285-v0',
        entry_point='env_configs.env.reacher:Reacher7DOFEnv',
        max_episode_steps=500,
    )

