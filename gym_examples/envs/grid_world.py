import gym
from gym import spaces
import pygame
import torch
import numpy as np
import yaml

class GridWorldEnv():
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, size=10):
        with open("gridworld_params.yaml", "r") as cfg:
            try:
                self.cfg = yaml.safe_load(cfg)
            except yaml.YAMLError as exc:
                print(exc)
        print(self.cfg)

        self.device = self.cfg["env"]["device"]
        self.size = self.cfg["env"]["board_size"]
        self.num_envs = self.cfg["env"]["num_envs"]
        self.num_features = self.cfg["env"]["num_features"]
        self.num_apples = self.cfg["env"]["num_apples"]
        self.apple_prob = self.cfg["env"]["num_prob"]
        self.fire_prob = self.cfg["env"]["num_prob"]

        self.boards = torch.zeros(self.num_envs, self.num_features, self.size, self.size, device=self.device)
        self.player_layer = self.boards.view(self.num_envs, self.num_features, self.size, self.size)[:,0, ...]
        self.goal_layer = self.boards.view(self.num_envs, self.num_features, self.size, self.size)[:,1, ...]
        self.apple_layer = self.boards.view(self.num_envs, self.num_features, self.size, self.size)[:,2, ...]
        self.fire_layer = self.boards.view(self.num_envs, self.num_features, self.size, self.size)[:,3, ...]

        self.env_idx = torch.arange(0, self.num_envs, device=self.device)
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Box(0, self.num_features, shape=(self.size, self.size), dtype=int)
        # North, South, West, East, Stay
        self.action_space = spaces.Discrete(5)

        

        render_mode = "human"
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.view_env = 0

        self.reset(torch.arange(self.num_envs, device=self.device))

    def _get_obs(self, env_id):
        return self.boards[env_id, ...]


    def reset(self, env_ids, seed=None, options=None):
        new_player_loc = torch.randint(0, self.size, [len(env_ids), 2])
        self.player_layer[env_ids, ...] = torch.zeros_like(self.player_layer[env_ids, ...], device=self.device)
        self.player_layer[env_ids, new_player_loc[:,0], new_player_loc[:,1]] = 1

        new_goal_loc = torch.randint(0, self.size, [len(env_ids), 2])
        self.goal_layer[env_ids, ...] = torch.zeros_like(self.goal_layer[env_ids, ...], device=self.device)
        self.goal_layer[env_ids, new_goal_loc[:,0], new_goal_loc[:,1]] = 1

        self.apple_layer[env_ids, ...] = torch.where(torch.rand([self.size,self.size], device=self.device) > self.apple_prob, 0., 1.)
        self.fire_layer[env_ids, ...] = torch.where(torch.rand([self.size,self.size], device=self.device) > self.fire_prob, 0., 1.)
      
        observation = self._get_obs(env_ids)

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        
        # # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        # # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # reward = 1 if terminated else 0  # Binary sparse rewards
        # observation = self._get_obs()
        # info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        # return observation, reward, terminated, False, info

    def render(self):
        # if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels


        # Draw Apple
        _al_xt, _al_yt = torch.where(self.apple_layer[self.view_env, ...] == 1)
        for i in range(len(_al_xt)):
            _al_x = _al_xt[i].cpu().numpy()
            _al_y = _al_yt[i].cpu().numpy()
            _apple_location = np.array([_al_x, _al_y])
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (_apple_location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
        # Draw Fire
        _fl_xt, _fl_yt = torch.where(self.fire_layer[self.view_env, ...] == 1)
        for i in range(len(_fl_xt)):
            _fl_x = _fl_xt[i].cpu().numpy()
            _fl_y = _fl_yt[i].cpu().numpy()
            _fire_location = np.array([_fl_x, _fl_y])
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * _fire_location,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Draw Goal
        _gl_xt, _gl_yt = torch.where(self.goal_layer[self.view_env, ...] == 1)
        _gl_x = _gl_xt[0].cpu().numpy()
        _gl_y = _gl_yt[0].cpu().numpy()
        _goal_location = np.array([_gl_x, _gl_y])
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * _goal_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Draw Player
        _pl_xt, _pl_yt = torch.where(self.player_layer[self.view_env, ...] == 1)
        _pl_x = _pl_xt[0].cpu().numpy()
        _pl_y = _pl_yt[0].cpu().numpy()
        _player_location = np.array([_pl_x, _pl_y])
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (_player_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )



        # Add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
