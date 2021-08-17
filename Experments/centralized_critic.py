import os
from ray.rllib.models import ModelCatalog
from Environment.MultiAgentCowdEnv import MultiAgentCrowdEnv
from ray import tune
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
import gym

torch, nn = try_import_torch()

# Params config
LOG_DIR_NAME = "A2C"
ALGORITHM = "A2C"
NUM_AGENTS = 6
FRAMEWORK = "torch"
STOP_REWARD = 0.92 * 1000 * NUM_AGENTS
STOP_ITERS = 200000
STOP_TIMESTEPS = 1000000

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
          "output_shared_state_and_actions": True}
env = MultiAgentCrowdEnv(config)


# custom model with centralized critic
class CentralizedCriticCrowdNavModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.action_model = TorchFC(
            gym.spaces.Box(low=-max(env.map_dim),
                           high=max(env.map_dim),
                           shape=(1, env.obs_dim * env.obs_dim + 2)),
            action_space,
            num_outputs,
            model_config,
            name + "_action")

        self.value_model = TorchFC(obs_space, action_space, 1, model_config,
                                   name + "_vf")
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        # Store all the observations for critic
        self._model_in = [input_dict["obs_flat"], state, seq_lens]

        # Run actor only on own observations
        return self.action_model({
            "obs": input_dict["obs"]["own_obs"]
        }, state, seq_lens)

    def value_function(self):
        # run critic on all observations
        value_out, _ = self.value_model({
            "obs": self._model_in[0]
        }, self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])


ModelCatalog.register_custom_model("centralized_critic_crowd_nav", CentralizedCriticCrowdNavModel)

# config for experiment
config = {
    "env": MultiAgentCrowdEnv,
    "env_config": config,
    "no_done_at_end": True,
    "batch_mode": "complete_episodes",
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "multiagent": {
        "policies": {
            str(i) : (None, env.observation_space, env.action_space, {}) for i in range(NUM_AGENTS)
        },
        "policy_mapping_fn": (lambda aid, **kwargs: str(aid)),
    },
    "model": {
        "custom_model": "centralized_critic_crowd_nav",
    },
    "framework": FRAMEWORK,
}

stop = {
    "episode_reward_mean": STOP_REWARD,
    "timesteps_total": STOP_TIMESTEPS,
    "training_iteration": STOP_ITERS,
}

results = tune.run(ALGORITHM, name=LOG_DIR_NAME, config=config, stop=stop, verbose=1)
