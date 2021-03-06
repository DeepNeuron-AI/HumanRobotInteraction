from typing import Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import MultiAgentDict
import numpy as np
import gym
from Agent import Agent
from GymInfo import Nothing, Collision, ReachGoal, Discomfort, Timeout
import matplotlib.lines as mlines

def generate_map(dims):
    return np.zeros(dims, dtype=np.float32)


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
        self.stop_on_collision = config["stop_on_collision"]
        self.output_shared_state_and_actions = config["output_shared_state_and_actions"]

        # initialise vars
        self.map = None
        self.global_time = 0
        self._reset_history()

        # observation space = flattened radius of nearby squares + distance to goal
        obs_dim = self.state_radius * 2 + 1
        self.obs_dim = obs_dim
        own_observation_space = gym.spaces.Box(low=-max(self.map_dim),
                                               high=max(self.map_dim),
                                               shape=(1, obs_dim * obs_dim + 2))

        # use shared obs and actions for multi agent
        if self.output_shared_state_and_actions:
            self._initialise_shared_spaces()

            shared_observation_space = gym.spaces.Box(low=-max(self.map_dim),
                                                      high=max(self.map_dim),
                                                      shape=(self.num_agents, obs_dim * obs_dim + 2))
            shared_action_space = gym.spaces.Box(low=0, high=9, shape=(self.num_agents, 1))

            self.observation_space = gym.spaces.Dict({
                "own_obs": own_observation_space,
                "shared_obs": shared_observation_space,
                "shared_action": shared_action_space,
            })

        # or only local view for regular rl
        else:
            self.observation_space = own_observation_space

        # ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))
        self.action_space = gym.spaces.Discrete(9)

        self.agents = [Agent(id) for id in range(self.num_agents)]

    # ------------- PUBLIC INTERFACE -------------------
    @override(MultiAgentEnv)
    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        if self.output_shared_state_and_actions:
            self._initialise_shared_spaces()
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self._step_for_agent(self.agents[i], action)

        done["__all__"] = sum([1 if a.done else 0 for a in self.agents]) == len(self.agents)

        self._record_history()
        self.global_time += 1

        return obs, rew, done, info

    @override(MultiAgentEnv)
    def reset(self) -> MultiAgentDict:
        self._generate_new_map()
        self._reset_history()
        self.global_time = 0

        if self.output_shared_state_and_actions:
            self._initialise_shared_spaces()

        for agent in self.agents:
            agent.reset(*self._generate_start_and_goal())
            self.map[tuple(agent.position)] = 2

        return {a.id: self._state_for_agent(a) for a in self.agents}

    @override(MultiAgentEnv)
    def render(self, mode='video', output_file=None):
        # only render once all agents are done
        if sum([1 if a.done else 0 for a in self.agents]) != self.num_agents:
            return

        from matplotlib import animation
        import matplotlib.pyplot as plt

        cmap = plt.cm.get_cmap('hsv', len(self.agents) * 2)

        fig, ax = plt.subplots(figsize=(9,9))
        ax.tick_params(labelsize=12)
        ax.set_xlim(-1, self.map_dim[0])
        ax.set_ylim(-1, self.map_dim[1])
        ax.set_xlabel('x(m)', fontsize=14)
        ax.set_ylabel('y(m)', fontsize=14)

        # add obstacle markers
        obstacles = np.transpose((self.map == 1).nonzero())
        for obstacle in obstacles:
            marker = mlines.Line2D([obstacle[0]], [obstacle[1]],
                                   color='k',
                                   marker='s', linestyle='None', markersize=8)
            ax.add_artist(marker)
        # add start positions and goals
        colors = [cmap(i) for i in range(len(self.agents))]
        for i in range(len(self.agents)):
            agent = self.agents[i]
            agent_goal = mlines.Line2D([agent.goal[0]], [agent.goal[1]],
                                       color=colors[i],
                                       marker='*', linestyle='None', markersize=8)
            ax.add_artist(agent_goal)
            human_start = mlines.Line2D([agent.starting_pos[0]], [agent.starting_pos[1]],
                                        color=colors[i],
                                        marker='P', linestyle='None', markersize=8)
            ax.add_artist(human_start)

        # add agent positions and their numbers
        agent_positions = [[(self.history[t, i, 0], self.history[t, i, 1]) for i in range(len(self.agents))] for t in
                           range(self.time_limit)]
        agent_width = max(self.map_dim) / 100
        agents = [plt.Circle((agent_positions[0][i][0], agent_positions[0][i][1]), agent_width , fill=False, color=colors[i]) for i
                  in range(len(self.agents))]

        for i, agent in enumerate(agents):
            ax.add_artist(agent)

        # add time annotation
        time = plt.text(0.4, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
        ax.add_artist(time)

        global_step = 0

        def update(frame_num):
            nonlocal global_step
            global_step = frame_num

            for i, agent in enumerate(agents):
                agent.center = (agent_positions[frame_num][i][0], agent_positions[frame_num][i][1])
                if self.history[frame_num, i, 2] > 0:
                    agent.fill = True
                    if self.history[frame_num, i, 2] == 1:
                        agent.set_color('k')
            time.set_text('Time: {:.2f}'.format(frame_num * 1))

        anim = animation.FuncAnimation(fig, update, frames=self.global_time, interval=1 * 230)
        anim.running = True

        if output_file is not None:
            # save as video
            ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(output_file, writer=ffmpeg_writer)

            # save output file as gif if imagemagic is installed
            # anim.save(output_file, writer='imagemagic', fps=12)
        else:
            plt.show()

    # ------------ MAP FUNCTIONS --------------------

    def _generate_new_map(self):
        self.map = generate_map(self.map_dim)

    def _is_obstacle(self, pos):
        return self.map[tuple(pos)] == 1 or self.map[tuple(pos)] == 2

    def _generate_start_and_goal(self):
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

    def _random_position(self):
        return np.random.randint((0, 0), self.map_dim, (2))

    def _inside_map_boundaries(self, pos):
        return 0 <= pos[0] < self.map_dim[0] and 0 <= pos[1] < self.map_dim[1]

    # -------------- STATE FUNCTIONS -----------------
    def _state_for_agent(self, agent):
        # calculate own observation
        view = self._state_view(agent)
        goal = self._state_goal_position(agent)
        own_obs = np.expand_dims(np.concatenate((view.flatten(), goal)), axis=0)

        if self.output_shared_state_and_actions:
            # add to shared observations
            self.shared_obs[agent.id, :] = own_obs

            return {"own_obs": own_obs,
                    "shared_obs": self.shared_obs,
                    "shared_action": self.shared_actions}
        else:
            return own_obs

    def _initialise_shared_spaces(self):
        self.shared_obs = np.zeros((self.num_agents, self.obs_dim * self.obs_dim + 2), dtype=np.float32)
        self.shared_actions = np.zeros((self.num_agents, 1), dtype=np.float32)

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
        padded = np.ones((view_dims, view_dims), dtype=np.float32)
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
        return (agent.goal - agent.position).astype(dtype=np.float32)

    # -------------- AGENT FUNCTION ------------------

    def _reset_history(self):
        self.history = np.zeros((self.time_limit + 1, self.num_agents, 3))

    def _record_history(self):
        for i, agent in enumerate(self.agents):
            self.history[self.global_time, i, 0] = agent.position[0]
            self.history[self.global_time, i, 1] = agent.position[1]
            if agent.done:
                self.history[self.global_time, i, 2] = 1
                if np.array_equal(agent.position,agent.goal):
                    self.history[self.global_time, i, 2] = 2

    def _handle_discrete_action(self, action):
        actions = []
        values = (-1, 0, 1)
        for x in values:
            for y in values:
                actions.append([x, y])
        return np.array(actions[action])

    def _execute_agent_action(self, agent, action):
        action = self._handle_discrete_action(action)
        target_pos = agent.preview_move(action)

        # check for collision when pos outside of map
        if not self._inside_map_boundaries(target_pos):
            return True

        # check for collision when pos is an obstacle
        if self.map[tuple(target_pos)] == 1:
            return True

        # check for collision with another agent
        if self.map[tuple(target_pos)] == 2 and not np.array_equal(action, np.array([0, 0])):
            return True

        # no collision -> move agent + update map
        self.map[tuple(agent.position)] = 0
        agent.move(action)
        self.map[tuple(agent.position)] = 2

        return False

    def _step_for_agent(self, agent, action):

        if self.output_shared_state_and_actions:
            self.shared_actions[agent.id] = action

        if not agent.done:
            collision = self._execute_agent_action(agent, action)

            if self.global_time >= self.time_limit - 1:
                reward = 0
                agent.done = True
                info = Timeout()
            elif collision:
                reward = self.collision_penalty
                agent.done = self.stop_on_collision
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
                # reward = -np.linalg.norm(agent.goal-agent.position)
                reward = self.time_step_penalty
                info = Nothing()

            return self._state_for_agent(agent), reward, agent.done, info
        else:
            return self._state_for_agent(agent), 0, True, Nothing()

    def _testing_set_agent(self, agent):
        self.agents.append(agent)
        self.map[tuple(agent.position)] = 2
        self.history = np.zeros((self.time_limit + 1, len(self.agents), 3))
