from posggym import register

register(
    id="Driving14x14WideRoundAbout-n2-v0",
    entry_point="posggym.envs.grid_world.driving:DrivingEnv",
    max_episode_steps=50,
    kwargs={
        "grid": "14x14RoundAbout",
        "num_agents": 2,
        "obs_dim": (3, 1, 1),
    },
)