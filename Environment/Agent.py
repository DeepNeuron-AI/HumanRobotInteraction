import numpy as np

class Agent:
    def __init__(self, id):
        self.position = np.array([0, 0])
        self.goal = np.array([0, 0])
        self.done = False
        self.id = id

    def preview_move(self, action):
        assert len(action) == 2
        return self.position + action

    def move(self, action):
        assert len(action) == 2
        self.position += action

    def reset(self, starting_pos, goal_position):
        self.done = False
        self.position = starting_pos
        self.goal = goal_position

    def is_done(self):
        return self.done

    def reached_goal(self):
        return np.array_equal(self.position, self.goal)
