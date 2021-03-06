from Environment.MultiAgentCowdEnv import MultiAgentCrowdEnv
import random
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback

# Params config
LOG_DIR_NAME = "A2C"
ALGORITHM = "A2C"
NUM_AGENTS = 6
FRAMEWORK = "torch"
STOP_REWARD = 0.92 * 1000 * NUM_AGENTS
STOP_ITERS = 200000
STOP_TIMESTEPS = 10000000

# environment config
dims = (20, 20)
config = {"num_agents": NUM_AGENTS,
          "state_radius": 4,
          "map_dim": dims,
          "time_limit": 500,
          "time_step_penalty": -1,
          "success_reward": 1000,
          "collision_penalty": -50,
          "stop_on_collision": True,
          "output_shared_state_and_actions": False}

env = MultiAgentCrowdEnv(config)

# config for experiment
config = {
    "env": MultiAgentCrowdEnv,
    "env_config": config,
    "multiagent": {
        "policies":
            {
                str(i): (None, env.observation_space, env.action_space, {"gamma": 0.99}) for i in range(NUM_AGENTS)
            },
        "policy_mapping_fn":
            lambda agent_id: random.choice([str(i) for i in range(NUM_AGENTS)])
    },
    "evaluation_interval": 2,
    "evaluation_num_episodes": 1,
    "evaluation_num_workers": 1,
    "evaluation_config": {
        "record_env": True,
        "render_env": True,
    },
    "framework": FRAMEWORK,
}
stop = {
    "episode_reward_mean": STOP_REWARD,
    "timesteps_total": STOP_TIMESTEPS,
    "training_iteration": STOP_ITERS,
}

# callbacks = [WandbLoggerCallback(project="Hri", api_key="ceb8d4ecf459f66ca402d1515a3df16ba5debf31", log_config=True)]
callbacks = []
results = tune.run(ALGORITHM, name=LOG_DIR_NAME, stop=stop, config=config, verbose=1, callbacks=callbacks)
