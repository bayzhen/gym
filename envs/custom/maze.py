import math
import random
import sys
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

FPS = 1
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

VIEWPORT_W = 600
VIEWPORT_H = 400


class Edge:
    two_points = []

    def __init__(self, x0, y0, x1, y1):
        self.two_points = [[x0, y0], [x1, y1]]


class Maze(gym.Env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FPS}

    maze_size_x = 10
    maze_size_y = 10
    count = None

    wall_color = (0, 0, 1)
    path_color = (1, 1, 1)
    agent_color = (0, 1, 0)

    current_x = 1
    current_y = 1
    destination_x = maze_size_x
    destination_y = maze_size_y

    step_count = 0
    step_up_limit = 2000

    def get_edge(self, two_points):
        x0, y0, x1, y1 = two_points[0][0], two_points[0][1], two_points[1][0], two_points[1][1]
        data_x0 = 2 * x0 - 1
        data_y0 = 2 * y0 - 1
        data_x1 = 2 * x1 - 1
        data_y1 = 2 * y1 - 1
        data_x = int((data_x0 + data_x1) / 2)
        data_y = int((data_y0 + data_y1) / 2)
        return self.data[data_x, data_y]

    def break_edge(self, two_points):
        x0, y0, x1, y1 = two_points[0][0], two_points[0][1], two_points[1][0], two_points[1][1]
        data_x0 = 2 * x0 - 1
        data_y0 = 2 * y0 - 1
        data_x1 = 2 * x1 - 1
        data_y1 = 2 * y1 - 1
        data_x = int((data_x0 + data_x1) / 2)
        data_y = int((data_y0 + data_y1) / 2)
        self.data[data_x, data_y] = 0

    def get_point(self, point):
        x = point[0]
        y = point[1]
        return self.data[2 * x - 1][2 * y - 1]

    def set_point(self, point, value):
        x = point[0]
        y = point[1]
        self.data[2 * x - 1][2 * y - 1] = value

    def __init__(self):
        EzPickle.__init__(self)
        self.data_sizeH, self.data_sizeW = self.maze_size_x * 2 + 1, self.maze_size_y * 2 + 1
        self.data = np.zeros(shape=(self.data_sizeH, self.data_sizeW), dtype=int)
        self.seed()
        self.viewer = None
        self.prev_reward = None
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.data_sizeH, self.data_sizeW), dtype=int
        )
        self.action_space = spaces.Discrete(4)
        self.reset()

    def regenerate_maze_frame(self):
        self.data = np.zeros(shape=(self.data_sizeH, self.data_sizeW), dtype=int)
        i = 0
        while i <= self.maze_size_x:
            self.data[i * 2, :] = 1
            i = i + 1
        i = 0
        while i <= self.maze_size_y:
            self.data[:, i * 2] = 1
            i = i + 1
        self.data[0, :] = 2
        self.data[self.maze_size_x * 2, :] = 2
        self.data[:, 0] = 2
        self.data[:, self.maze_size_y * 2] = 2
        self.random_prime_create_maze()

    def random_prime_create_maze(self):
        point_list = [[1, 1]]
        two_points_list = [[[1, 1], [1, 2]], [[1, 1], [2, 1]]]
        while len(two_points_list):
            if self.count is not None:
                position = self.count % len(two_points_list)
                two_points = two_points_list[position]
                self.count = self.count + 12
            else:
                two_points = random.choice(two_points_list)
            if self.get_edge(two_points) == 2:
                two_points_list.remove(two_points)
            elif two_points[0] in point_list and two_points[1] in point_list:
                two_points_list.remove(two_points)
            else:
                self.break_edge(two_points)
                out_point = two_points[0] if two_points[0] not in point_list else two_points[1]
                point_list.append(out_point)
                up_point = [out_point[0] - 1, out_point[1]]
                down_point = [out_point[0] + 1, out_point[1]]
                left_point = [out_point[0], out_point[1] - 1]
                right_point = [out_point[0], out_point[1] + 1]
                two_points_list.append([out_point, up_point])
                two_points_list.append([out_point, down_point])
                two_points_list.append([out_point, left_point])
                two_points_list.append([out_point, right_point])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.regenerate_maze_frame()
        self.current_x = 1
        self.current_y = 1
        self.step_count = 0
        return self.step(0)[0]

    def step(self, action):
        state = np.copy(self.data)
        reward = 0
        done = False
        if action == 0:
            next_point = [self.current_x - 1, self.current_y]
        elif action == 1:
            next_point = [self.current_x + 1, self.current_y]
        elif action == 2:
            next_point = [self.current_x, self.current_y - 1]
        elif action == 3:
            next_point = [self.current_x, self.current_y + 1]
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (
                action,
                type(action),
            )
        two_points = [[self.current_x, self.current_y], next_point]
        self.step_count = self.step_count + 1
        if self.step_count >= self.step_up_limit:
            done = True
            state[self.current_x * 2 - 1, self.current_y * 2 - 1] = 3
            return np.array(state, dtype=int), reward, done, {}

        if self.get_edge(two_points) == 1 or self.get_edge(two_points) == 2:
            reward = -1
            state[self.current_x * 2 - 1, self.current_y * 2 - 1] = 3
            return np.array(state, dtype=int), reward, done, {}
        else:
            if next_point == [self.destination_x, self.destination_y]:
                reward = 1
                done = True
            else:
                reward = 0
                done = False
            state[next_point[0] * 2 - 1, next_point[1] * 2 - 1] = 3
            self.current_x, self.current_y = next_point[0], next_point[1]
            return np.array(state, dtype=int), reward, done, {}

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        box_w = VIEWPORT_W / (self.maze_size_x) / SCALE
        box_h = VIEWPORT_H / (self.maze_size_y) / SCALE
        line_w = 5
        for i in range(1, self.maze_size_x + 1):
            for j in range(1, self.maze_size_y + 1):
                p0 = ((i - 1) * box_w, (j - 1) * box_h)
                p1 = ((i) * box_w, (j - 1) * box_h)
                p2 = ((i) * box_w, (j) * box_h)
                p3 = ((i - 1) * box_w, (j) * box_h)
                p = [p0, p1, p2, p3]
                self.viewer.draw_polygon(p, color=self.path_color)
        p0 = (0, 0)
        p1 = (VIEWPORT_W / SCALE, 0)
        p2 = (VIEWPORT_W / SCALE, 0)
        p3 = (0, 0)
        p = [p0, p1, p2, p3]
        self.viewer.draw_polyline(p, color=self.wall_color, linewidth=line_w)

        p0 = (0, 0)
        p1 = (0, VIEWPORT_H / SCALE)
        p2 = (0, VIEWPORT_H / SCALE)
        p3 = (0, 0)
        p = [p0, p1, p2, p3]
        self.viewer.draw_polyline(p, color=self.wall_color, linewidth=line_w)

        for i in range(1, self.maze_size_x + 1):
            for j in range(1, self.maze_size_y + 1):
                p0 = (i * box_w, (j - 1) * box_h)
                p1 = (i * box_w, j * box_h)
                p = [p0, p1, p1, p0]
                if self.get_edge(((i, j), (i + 1, j))):
                    self.viewer.draw_polyline(p, color=self.wall_color, linewidth=line_w)
                p0 = ((i - 1) * box_w, j * box_h)
                p1 = (i * box_w, j * box_h)
                p = [p0, p1, p1, p0]
                if self.get_edge(((i, j), (i, j + 1))):
                    self.viewer.draw_polyline(p, color=self.wall_color, linewidth=line_w)

        p0 = [box_w * self.current_x, box_h * self.current_y]
        p1 = [box_w * (self.current_x - 1), box_h * self.current_y]
        p2 = [box_w * (self.current_x - 1), box_h * (self.current_y - 1)]
        p3 = [box_w * self.current_x, box_h * (self.current_y - 1)]
        p = [p0, p1, p2, p3]
        self.viewer.draw_polygon(p, color=self.agent_color)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class MazeSimple(Maze):
    count = 10


class MazeBigSimple(Maze):
    count = 10
    maze_size_x = 20
    maze_size_y = 20


def demo_heuristic_maze(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = random.choice([0, 1, 2, 3])
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False:
                break

        steps += 1
        if done:
            break
    if render:
        env.close()
    return total_reward


if __name__ == "__main__":
    demo_heuristic_maze(MazeBigSimple(), render=True)
