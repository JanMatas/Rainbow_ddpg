from gym.envs.registration import register
from itertools import chain, combinations
env_args = {
    "pusher_fix": {
        "env_id": "MicoEnv-pusher_fix-{}-{}-{}-{}-v1",
        'entry_point': 'micoenv.mico_robot_env:MicoEnv'
    },
}

env_kwargs = {
    "pusher_fix": {
        "doneAfter": 300,
        'target_in_the_air': False,
        "has_object": True
    },
}


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


for obs_type in ["low_dim", "pixels", "pixels_stacked"]:

    for reward_type in ["positive"]:
        for randomness in map(lambda a: "acton" + "".join(a), powerset(["x"])):
            for useVelos in [False]:
                for env_type in ["pusher_fix"]:
                    velos_id = "velos" if useVelos else "ik"
                    env_id = env_args[env_type]["env_id"].format(
                        obs_type, reward_type, velos_id, randomness)
                    kwargs = env_kwargs[env_type].copy()
                    kwargs["observation_type"] = obs_type
                    if obs_type == "pixels_stacked":
                        kwargs["frameMemoryLen"] = 4
                    kwargs["reward_type"] = reward_type
                    kwargs["randomizeArm"] = "a" in randomness
                    kwargs["randomizeCamera"] = "c" in randomness
                    kwargs["randomizeTextures"] = "t" in randomness
                    kwargs["randomizeObjects"] = "o" in randomness
                    kwargs["normalTextures"] = "n" in randomness

                    kwargs["velocityControl"] = useVelos
                    register(
                        id=env_id,
                        entry_point=env_args[env_type]["entry_point"],
                        kwargs=kwargs)
register(
    id="MicoReal-v1",
    entry_point="micoenv.mico_real_env:MicoRealEnv",
    kwargs={})

register(
    id="MicoSimReal-v1",
    entry_point="micoenv.mico_simreal_env:MicoSimRealEnv",
    kwargs={
        'target_in_the_air': False,
        "has_object": True,
        "fixed_goal": True,
        "use_gui": True,
        "randomizeArm": True,
        "randomizeCamera": True,
        "randomizeTextures": True
    },
)
