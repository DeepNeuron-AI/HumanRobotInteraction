import unittest
from Environment.MultiAgentCowdEnv import MultiAgentCrowdEnv
from Environment.Agent import Agent
from Environment.GymInfo import Nothing, Collision, ReachGoal, Discomfort, Timeout
import numpy as np


class MultiAgentCrowdEnvTests(unittest.TestCase):

    def test_agent_state_view_middle(self):
        # create Env
        env = self._get_empty_env()

        # create Agent
        agent = self._get_agent([10, 10])

        # get state view
        view = env._state_view(agent)

        # assert all empty spaces
        for row in view:
            self.assertTrue(len(row[row != 0]) == 0)

    def test_agent_state_view_top_left(self):
        # create Env
        env = self._get_empty_env()

        # create Agent
        agent = self._get_agent([0, 19])

        # get state view
        view = env._state_view(agent)

        # assert correct view
        X, Y = view.shape
        for x in range(X):
            for y in range(Y):
                val = view[x, y]
                if x < int(X / 2):
                    self.assertTrue(val == 1)
                elif y > int(Y / 2):
                    self.assertTrue(val == 1)
                else:
                    self.assertTrue(val == 0)

    def test_agent_state_view_bot_right(self):
        # create Env
        env = self._get_empty_env()

        # create Agent
        agent = self._get_agent([19, 0])

        # get state view
        view = env._state_view(agent)

        # assert correct view
        X, Y = view.shape
        for x in range(X):
            for y in range(Y):
                val = view[x, y]
                if x > int(X / 2):
                    self.assertTrue(val == 1)
                elif y < int(Y / 2):
                    self.assertTrue(val == 1)
                else:
                    self.assertTrue(val == 0)

    def test_agent_state_view_other_agents(self):
        # create Env
        env = self._get_empty_env()

        # create Agent
        agent = self._get_agent([10, 10])
        other_1 = self._get_agent([11, 11])
        other_2 = self._get_agent([9, 9])
        other_3 = self._get_agent([11, 9])
        other_4 = self._get_agent([9, 11])
        env._testing_set_agent(other_1)
        env._testing_set_agent(other_2)
        env._testing_set_agent(other_3)
        env._testing_set_agent(other_4)

        # get state view
        view = env._state_view(agent)

        # assert other agents visible
        X, Y = view.shape
        agent_pos = [int(X / 2), int(Y / 2)]
        other_pos = []
        for x in (1, -1):
            for y in (1, -1):
                other_pos.append([agent_pos[0] + x, agent_pos[1] + y])
        for x, y in other_pos:
            self.assertTrue(view[x, y] == 2)

    def test_agent_state_goal(self):
        # create Env
        env = self._get_empty_env()

        # create Agent
        agent = self._get_agent([10, 10], [0, 0])

        # get state view
        gaol_pos = env._state_goal_position(agent)

        self.assertTrue(np.array_equal(np.array([-10, -10]), gaol_pos))

    def test_step_outside_map(self):
        # create Env
        env = self._get_empty_env()

        # create Agent
        agent = self._get_agent([0, 10])
        env._testing_set_agent(agent)

        # get state view
        obs, rew, done, info = env.step({0: np.array([-1, 0])})

        self.assertTrue(done[0])
        self.assertEqual(str(info[0]), "Collision")
        self.assertEqual(rew[0], env.collision_penalty)

    def test_step_into_obstacle(self):
        # create Env
        env = self._get_empty_env()
        env.map[11, 10] = 1

        # create Agent
        agent = self._get_agent([10, 10])
        env._testing_set_agent(agent)

        # get state view
        obs, rew, done, info = env.step({0: np.array([1, 0])})

        self.assertTrue(done[0])
        self.assertEqual(str(info[0]), "Collision")
        self.assertEqual(rew[0], env.collision_penalty)

    def test_step_reach_goal(self):
        # create Env
        env = self._get_empty_env()

        # create Agent
        agent = self._get_agent([10, 10], [11, 10])
        env._testing_set_agent(agent)

        # get state view
        obs, rew, done, info = env.step({0: np.array([1, 0])})

        self.assertTrue(done[0])
        self.assertEqual(str(info[0]), "Reaching goal")
        self.assertEqual(rew[0], env.success_reward)

    def test_step_into_other_agent(self):
        # create Env
        env = self._get_empty_env()

        # create Agent
        agent = self._get_agent([10, 10])
        other = self._get_agent([11, 10])
        env._testing_set_agent(agent)
        env._testing_set_agent(other)

        # get state view
        obs, rew, done, info = env.step({0: np.array([1, 0])})

        self.assertTrue(done[0])
        self.assertEqual(str(info[0]), "Collision")
        self.assertEqual(rew[0], env.collision_penalty)

    def test_step_out_of_time(self):
        # create Env
        env = self._get_empty_env()
        env.time_limit = 2
        env.global_time = 2

        # create Agent
        agent = self._get_agent([10, 10])
        env._testing_set_agent(agent)

        # get state view
        obs, rew, done, info = env.step({0: np.array([1, 0])})

        self.assertTrue(done[0])
        self.assertEqual(str(info[0]), "Timeout")
        self.assertEqual(rew[0], 0)

    def test_step_into_valid_space(self):
        # create Env
        env = self._get_empty_env()

        # create Agent
        agent = self._get_agent([10, 10])
        env._testing_set_agent(agent)

        # get state view
        obs, rew, done, info = env.step({0: np.array([1, 0])})

        self.assertFalse(done[0])
        self.assertEqual(rew[0], env.time_step_penalty)

    def _render_preview(self):
        dims = (100, 100)
        config = {"num_agents": 5,
                  "state_radius": 4,
                  "map_dim": dims,
                  "time_limit": 50,
                  "time_step_penalty": -1,
                  "success_reward": 10,
                  "collision_penalty": -10}
        env = MultiAgentCrowdEnv(config)

        env.reset()
        map = np.zeros(dims)
        map[9, 5] = 1
        map[9, 14] = 1
        map[9, 15] = 1
        map[19, 5] = 1
        map[14, 5] = 1
        env.map = map
        for e in range(50):
            actions = {}
            for i, agent in enumerate(env.agents):
                a = env.action_space.sample()
                actions[i] = a
            obs, rew, done, info = env.step(actions)
            g = 9

        # env.render(output_file="test.mp4")
        env.render()

    def _get_empty_env(self):
        dims = (20, 20)
        config = {"num_agents": 0,
                  "state_radius": 4,
                  "map_dim": dims,
                  "time_limit": 500,
                  "time_step_penalty": -1,
                  "success_reward": 10,
                  "collision_penalty": -10}
        env = MultiAgentCrowdEnv(config)
        map = np.zeros(dims)
        env.map = map
        return env

    def _get_agent(self, pos, goal=None):
        agent = Agent(0)
        agent.position = np.array(pos)
        agent.starting_pos = np.array(pos)
        if goal is not None:
            agent.goal = np.array(goal)
        return agent
