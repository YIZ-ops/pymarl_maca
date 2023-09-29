from functools import partial

import numpy as np
import torch

from pymarl_src.components.episode_buffer import EpisodeBatch
# from pymarl_src.envs import REGISTRY as env_REGISTRY

import copy

from agent.fix_rule.agent import Agent
from interface import Environment

RENDER = True
MAP_PATH = 'maps/1000_1000_fighter10v10.map'
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM  # plus one action: no attack


# LEARN_INTERVAL = 100


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # create blue agent
        self.blue_agent = Agent()
        # get agent obs type
        red_agent_obs_ind = 'simple_copy'
        blue_agent_obs_ind = self.blue_agent.get_obs_ind() # raw
        # make env
        self.env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)

        # get map info
        size_x, size_y = self.env.get_map_size()
        red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = self.env.get_unit_num()
        # set map info to blue agent
        self.blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = 300  # Following the number used in MaCA
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):

        env_info = {
            "state_shape": 140,
            "obs_shape": 140,
            "n_actions": ACTION_NUM,
            "n_agents": FIGHTER_NUM,
            "episode_limit": self.episode_limit

        }
        return env_info

    def save_replay(self):
        # self.env.save_replay()
        return

    def close_env(self):
        # self.env.close()
        return

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def env_get_state(self):
        return

    def env_get_avail_actions(self):
        red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = self.env.get_unit_num()
        available_action = np.ones((red_fighter_num, ACTION_NUM))
        return available_action.tolist()

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        while not terminated:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = self.env.get_unit_num()
            red_obs_dict, blue_obs_dict = self.env.get_obs()
            red_state_dict, blue_state_dict = self.env.get_state()

            pre_transition_data = {
                "state": red_state_dict,
                "avail_actions": [self.env_get_avail_actions()],
                "obs": red_obs_dict
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # get action
            # get blue action
            blue_detector_action, blue_fighter_action = self.blue_agent.get_action(blue_obs_dict, step_cnt=self.t)

            # get red action
            red_detector_action = []
            red_fighter_action = []
            action_list = []
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions = actions[0]
            cpu_actions = actions.to("cpu").numpy()

            obs_got_ind = [False] * red_fighter_num
            for i in range(len(red_obs_dict)):
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                if red_obs_dict[i][0] != 0: # alive
                    obs_got_ind[i] = True
                    tmp_action = [actions[i]]
                    action_list.append(tmp_action)
                    # action formation
                    true_action[0] = int(360 / COURSE_NUM * int(tmp_action[0] / ATTACK_IND_NUM))
                    true_action[3] = int(tmp_action[0] % ATTACK_IND_NUM)
                else:
                    tmp_action = ACTION_NUM - 1  # if the fighter is dead, it cannot execute any action, so no attack
                    action_list.append(tmp_action)

                red_fighter_action.append(true_action)
            red_fighter_action = np.array(red_fighter_action)

            # step
            self.env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
            # get reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = self.env.get_reward()
            episode_return += red_fighter_reward

            terminated = self.env.get_done()
            if self.t == self.episode_limit:
                terminated = True
            
            post_transition_data = {
                "actions": [np.array([action]) for action in cpu_actions],
                "reward": [(sum(red_fighter_reward),)],
                "terminated": [(self.env.get_done(),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        if not test_mode:
            self.t_env += self.t

        return self.batch
