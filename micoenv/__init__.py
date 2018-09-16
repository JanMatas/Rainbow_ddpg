from gym.envs.registration import register
register(
    id="Pusher-v1",
    entry_point="micoenv.mico_robot_env:MicoEnv",
    kwargs={
        "randomize_arm": True,
        "randomize_camera": True,
        "randomize_textures": True,
        "randomize_objects": True,
        "normal_textures": True,
        "done_after": 300,
        'target_in_the_air': False,
        "has_object": True,
        "reward_type": "positive",
        "observation_type": "pixels",


    }
)


