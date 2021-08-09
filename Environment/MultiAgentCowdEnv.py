from typing import Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, AgentID
import numpy as np
import gym
from Agent import Agent
from GymInfo import Nothing, Collision, ReachGoal, Discomfort, Timeout


def generate_map(dims):
    return np.zeros(dims)


class MultiAgentCrowdEnv(MultiAgentEnv):
    def __init__(self, config):
        # store configs
        self.num_agents = config["num_agents"]
        self.state_radius = config["state_radius"]
        self.map_dim = config["map_dim"]
        self.collision_penalty = config["collision_penalty"]
        self.success_reward = config["success_reward"]
        self.time_step_penalty = config["time_step_penalty"]
        self.time_limit = config["time_limit"]

        # initialise vars
        self.map = None
        self.global_time = 0
        obstacle_shape = (self.state_radius * 2 + 1, self.state_radius * 2 + 1)
        spaces = {
            'view': gym.spaces.Box(low=0, high=2, shape=obstacle_shape),
            'goal_position': gym.spaces.Box(low=np.array([0, 0]), high=np.array(self.map_dim))
        }
        # state is nearby spaces with 0 for empty, 1 for obstacle and 2 for other agent
        self.observation_space = gym.spaces.Dict(spaces)

        # action space is 2 numbers, -1, 0, 1 in x and y
        self.action_space = gym.spaces.MultiDiscrete([3, 3])

        # create agents
        self.agents = [Agent(id) for id in range(self.num_agents)]

    @override(MultiAgentEnv)
    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self._step_for_agent(self.agents[i], action)

        done["__all__"] = sum([1 if a.done else 0 for a in self.agents]) == len(self.agents)

        self.global_time += 1

        return obs, rew, done, info

    @override(MultiAgentEnv)
    def reset(self) -> MultiAgentDict:
        self._generate_new_map()
        self.global_time = 0

        for agent in self.agents:
            agent.reset(*self._generate_start_goal())
            self.map[tuple(agent.position)] = 2

        return {a.id: self._state_for_agent(a) for a in self.agents}

    @override(MultiAgentEnv)
    def render(self, mode=None):
        pass

    def _generate_new_map(self):
        self.map = generate_map(self.map_dim)

    def _state_for_agent(self, agent):
        return {"view": self._state_view(agent),
                "goal_position": self._state_goal_position(agent)}

    def _random_position(self):
        return np.random.randint((0, 0), self.map_dim, (2))

    def _is_obstacle(self, pos):
        return self.map[tuple(pos)] == 1 or self.map[tuple(pos)] == 2

    def _generate_start_goal(self):
        starting_pos = self._random_position()
        # generate random positions until not an obstacle
        while self._is_obstacle(starting_pos):
            starting_pos = self._random_position()

        goal_pos = self._random_position()
        # generate random positions until not an obstacle and not starting position
        while self._is_obstacle(goal_pos) or np.array_equal(starting_pos, goal_pos):
            goal_pos = self._random_position()

        # todo : check if agents are too close when setting start
        # todo : minimum distance from goal for starting position?
        return starting_pos, goal_pos

    def _state_view(self, agent):
        pos = agent.position
        view_dims = self.state_radius * 2 + 1

        # calculate x and y indexes of map section that is part of agents state
        x_start = int(max(0, pos[0] - self.state_radius))
        x_end = int(min(self.map_dim[0], pos[0] + self.state_radius + 1))
        y_start = int(max(0, pos[1] - self.state_radius))
        y_end = int(min(self.map_dim[1], pos[1] + self.state_radius + 1))

        # cut out slice of map
        view = self.map[x_start:x_end, y_start:y_end].copy()

        # pad state with obstacles if outside of map
        padded = np.ones((view_dims, view_dims))
        pad_dims = [0, view_dims, 0, view_dims]  # xmin, xmax, ymin, ymax

        # padding left
        if x_start == 0 and view.shape[0] < view_dims:
            pad_dims[0] = view_dims - view.shape[0]

        # padding right
        elif x_end == self.map_dim[0] and view.shape[0] < view_dims:
            pad_dims[1] = view.shape[0]

        # padding bottom
        if y_start == 0 and view.shape[1] < view_dims:
            pad_dims[2] = view_dims - view.shape[1]

        # padding right
        elif y_end == self.map_dim[1] and view.shape[1] < view_dims:
            pad_dims[3] = view.shape[1]

        # replace padded grid with state
        padded[pad_dims[0]:pad_dims[1], pad_dims[2]:pad_dims[3]] = view

        # todo: center of obs is 2 (because agent is in it) should it be replaced to a 0?
        # todo: instead of 0,1,2 for state, should it be 1 hot encoded?
        return padded

    def _state_goal_position(self, agent):
        return agent.goal - agent.position

    def _inside_map_boundaries(self, pos):
        return 0 <= pos[0] < self.map_dim[0] and 0 <= pos[1] < self.map_dim[1]

    def _execute_agent_action(self, agent, action):
        target_pos = agent.preview_move(action)

        # check for collision when pos outside of map
        if not self._inside_map_boundaries(target_pos):
            return True

        # check for collision when pos is an obstacle
        if self.map[tuple(target_pos)] == 1:
            return True

        # check for collision with another agent
        if self.map[tuple(target_pos)] == 2:
            return True

        # no collision -> move agent + update map
        self.map[tuple(agent.position)] = 0
        agent.move(action)
        self.map[tuple(agent.position)] = 2

        return False

    def _step_for_agent(self, agent, action):
        if not agent.done:
            collision = self._execute_agent_action(agent, action)

            if self.global_time >= self.time_limit - 1:
                reward = 0
                agent.done = True
                info = Timeout()
            elif collision:
                reward = self.collision_penalty
                agent.done = True
                info = Collision()

            elif agent.reached_goal():
                reward = self.success_reward
                agent.done = True
                info = ReachGoal()
            # elif dmin < self.discomfort_dist:
            #     # adjust the reward based on FPS
            #     reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            #     done = False
            #     info = Discomfort(dmin)
            else:
                reward = self.time_step_penalty
                info = Nothing()

            return self._state_for_agent(agent), reward, agent.done, info
        else:
            self._state_for_agent(agent), 0, True, Nothing()

    def _testing_set_agent(self, agent):
        self.agents.append(agent)
        self.map[tuple(agent.position)] = 2
