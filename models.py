#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:21:50 2024

@author: ahmadrezafrh
"""

import random
import contextlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import defaultdict
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

from tqdm.auto import tqdm
from game import Game, Player
from utils import load_dict, save_dict
from envs import QuixoEnv
import os



class ValueClass:
    def __init__(
        self,
        player1: Player,
        player2: Player
    ):
        """
        Initialize the ValueClass instance.

        Parameters:
        - player1 (Player): First player object.
        - player2 (Player): Second player object.
        """
        self.player1 = player1
        self.player2 = player2
        self.set_value_dictionary()

    def set_value_dictionary(self):
        """
        Initialize the value dictionary and set model existence flag to False.

        Parameters:
        - None

        Returns:
        - None
        """
        self.value_dictionary = defaultdict(float)
        self.model_exist = False

    def train(
        self,
        epsilon: float = 0.001,
        steps: int = 5_000_000,
        checkpoints: int = 50_000,
        resume: str = None
    ) -> None:
        """
        Train the value model.

        Parameters:
        - epsilon (float): Learning rate.
        - steps (int): Number of training steps.
        - checkpoints (int): Frequency of saving checkpoints.
        - resume (str): File path to resume training from a saved state.

        Returns:
        - None
        """
        if not self.model_exist:
            if resume:
                self.value_dictionary = defaultdict(float, load_dict(resume))
            for steps in tqdm(range(steps)):
                g = Game()
                self.player2.set_trajectory()
                final_reward = g.play(self.player1, self.player2)
                for state in self.player2.trajectory:
                    hashable_state = "".join(map(str, list(state.reshape(1, -1)[0])))
                    self.value_dictionary[hashable_state] = round(self.value_dictionary[
                        hashable_state
                    ] + epsilon * (final_reward - self.value_dictionary[hashable_state]), 5)

                if steps % checkpoints == 0 and steps != 0:
                    self.model_exist = True
                    self.save(file_name=f'cpt-{steps}.json', exist_ok=True)
            self.model_exist = True
        else:
            print('model exists, try to empty the value dictionary first')

    def save(
        self,
        models_path: str = './models',
        file_name: str = 'model.json',
        exist_ok: bool = False
    ) -> None:
        """
        Save the value model.

        Parameters:
        - models_path (str): Directory path to save the model.
        - file_name (str): Name of the saved model file.
        - exist_ok (bool): If True, overwrite existing model.

        Returns:
        - None
        """
        if self.model_exist:
            file_path = f'{models_path}/{file_name}'
            os.makedirs(models_path, exist_ok=True)
            if os.path.isfile(file_path):
                if exist_ok:
                    save_dict(self.value_dictionary, file_path)
                    print(f'Value dictionary has been rewritten in {file_name}')
                else:
                    print('Value dictionary exists')
            else:
                save_dict(self.value_dictionary, file_path)
                print(f'Value dictionary has been created in {file_name}')
        else:
            print('Model is not trained, train the model first')

    def load(
        self,
        file_path: str = './models/value_dictionary.json'
    ) -> None:
        """
        Load a pre-trained value model.

        Parameters:
        - file_path (str): File path of the saved model.

        Returns:
        - None
        """
        assert os.path.isfile(file_path)
        self.model_exist = True
        self.value_dictionary = load_dict(file_path)
        print('Value dictionary loaded')

    def top(
        self,
        n: int = 5,
        reverse: bool = True
    ):
        """
        Print the top N entries in the value dictionary.

        Parameters:
        - n (int): Number of top entries to display.
        - reverse (bool): If True, display in descending order.

        Returns:
        - None
        """
        print(sorted(self.value_dictionary.items(), key=lambda e: e[1], reverse=reverse)[0:n])




    
class PPOWrapper:
    def __init__(self, players: list[int] = [0, 1], models_dir: str = './models/ppo'):
        """
        Initialize the PPOWrapper instance.

        Parameters:
        - players (list[int]): List of player indices.
        - models_dir (str): Directory path for saving models.
        """
        self.players = [0, 1]  # Default player list is set to [1]
        self.models_dir = models_dir

    def train(
        self,
        net_arch: dict = dict(pi=[1024, 512, 256, 128], vf=[32, 32, 32, 32]),
        policy='MlpPolicy',
        verbose=1,
        progress_bar=True,
        ts: int = 5e5,
        callbacks: int = 50_000
    ):
        """
        Train PPO models for specified players.

        Parameters:
        - net_arch (dict): Neural network architecture configuration.
        - policy (str): Policy type.
        - verbose (int): Verbosity level.
        - progress_bar (bool): Display progress bar during training.
        - ts (int): Total training steps.
        - callbacks (int): Frequency of saving checkpoints.

        Returns:
        - None
        """
        for player in self.players:
            checkpoint_callback = CheckpointCallback(
                save_freq=callbacks,
                save_path=f"{self.models_dir}/player_{player}",
                name_prefix="quixo-ppo",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
            policy_kwargs = dict(net_arch=net_arch)
            env = QuixoEnv(player=player)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                model = PPO(policy, env, policy_kwargs=policy_kwargs,
                            verbose=verbose, tensorboard_log="./logs/ppo")
            model.learn(total_timesteps=ts, progress_bar=progress_bar,
                        callback=checkpoint_callback, tb_log_name=f"./quixo-ppo-{player}")
            del model

    def experiment(self, trials=100, steps=450000):
        """
        Run experiments with trained PPO models.

        Parameters:
        - trials (int): Number of trials for each player.

        Returns:
        - None
        """
        trials = trials
        for player in self.players:
            env = QuixoEnv(player=player)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                model = PPO.load(
                    f"{self.models_dir}/player_{player}/quixo-ppo_{steps}_steps", env=env)
            winners = 0
            done = False
            f = 0
            for i in range(trials):
                obs = env.reset()
                done = False
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    moves = env.get_moves(env.opposite)
                    if action not in moves:
                        action = random.choice(moves)
                        f += 1
                    obs, rewards, done, info = env.step(action)
                if player == 0 and info["winner"] == 0:
                    winners += 1
                elif player == 1 and info["winner"] == 1:
                    winners += 1
            print("\n")
            if player == 0:
                print("playing first")
            else:
                print("playing second")
            print(f"win percentage: {winners/trials}")
            print(f"total wrong actions: {f}\n")
            print("-----")






class DQNWrapper:
    def __init__(self, players: list[int] = [0, 1], models_dir: str = './models/dqn'):
        """
        Initialize the DQNWrapper instance.

        Parameters:
        - players (list[int]): List of player indices.
        - models_dir (str): Directory path for saving models.
        """
        self.players = [0, 1]  # Default player list is set to [0, 1]
        self.models_dir = models_dir

    def train(
        self,
        net_arch: list[int] = [1024, 512, 256, 128],
        policy='MlpPolicy',
        verbose=1,
        progress_bar=True,
        ts=int(2e6),
        callbacks=50_000
    ):
        """
        Train DQN models for specified players.

        Parameters:
        - net_arch (list[int]): Neural network architecture configuration.
        - policy (str): Policy type.
        - verbose (int): Verbosity level.
        - progress_bar (bool): Display progress bar during training.
        - ts (int): Total training steps.
        - callbacks (int): Frequency of saving checkpoints.

        Returns:
        - None
        """
        for player in self.players:
            checkpoint_callback = CheckpointCallback(
              save_freq=callbacks,
              save_path=f"{self.models_dir}/player_{player}",
              name_prefix="quixo-dqn",
              save_replay_buffer=False,
              save_vecnormalize=False,
            )
            policy_kwargs = dict(net_arch=net_arch)
            env = QuixoEnv(player=player)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=verbose, tensorboard_log="./logs/dqn")
            model.learn(total_timesteps=ts, progress_bar=progress_bar, callback=checkpoint_callback, tb_log_name=f"./quixo-dqn-{player}")
            del model

    def experiment(self, trials=100, steps=450000):
        """
        Run experiments with trained DQN models.

        Parameters:
        - trials (int): Number of trials for each player.

        Returns:
        - None
        """
        trials = trials
        for player in self.players:
            env = QuixoEnv(player=player)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                model = DQN.load(f"{self.models_dir}/player_{player}/quixo-dqn_{steps}_steps", env=env)
            winners = 0
            done = False
            f = 0
            for i in range(trials):
                obs = env.reset()
                done = False
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    moves = env.get_moves(env.opposite)
                    if action not in moves:
                        action = random.choice(moves)
                        f += 1
                    obs, rewards, done, info = env.step(action)
                if player == 0 and info["winner"] == 0:
                    winners += 1
                elif player == 1 and info["winner"] == 1:
                    winners += 1
            print("\n")
            if player == 0:
                print("playing first")
            else:
                print("playing second")
            print(f"win percentage: {winners/trials}")
            print(f"total wrong actions: {f}\n")
            print("-----")


class A2CWrapper:
    def __init__(self, players: list[int] = [0, 1], models_dir: str = './models/a2c'):
        """
        Initialize the A2CWrapper instance.

        Parameters:
        - players (list[int]): List of player indices.
        - models_dir (str): Directory path for saving models.
        """
        self.players = [0, 1]  # Default player list is set to [0, 1]
        self.models_dir = models_dir

    def train(
        self,
        net_arch: list[int] = [1024, 512, 256, 128],
        policy='MlpPolicy',
        verbose=1,
        progress_bar=True,
        ts=int(2e6),
        callbacks=50_000
    ):
        """
        Train A2C models for specified players.

        Parameters:
        - net_arch (list[int]): Neural network architecture configuration.
        - policy (str): Policy type.
        - verbose (int): Verbosity level.
        - progress_bar (bool): Display progress bar during training.
        - ts (int): Total training steps.
        - callbacks (int): Frequency of saving checkpoints.

        Returns:
        - None
        """
        for player in self.players:
            checkpoint_callback = CheckpointCallback(
              save_freq=callbacks,
              save_path=f"{self.models_dir}/player_{player}",
              name_prefix="quixo-a2c",
              save_replay_buffer=False,
              save_vecnormalize=False,
            )
            policy_kwargs = dict(net_arch=net_arch)
            env = QuixoEnv(player=player)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                model = A2C(policy, env, policy_kwargs=policy_kwargs, verbose=verbose, tensorboard_log="./logs/a2c")
            model.learn(total_timesteps=ts, progress_bar=progress_bar, callback=checkpoint_callback, tb_log_name=f"./quixo-a2c-{player}")
            del model

    def experiment(self, trials=100, steps=450000):
        """
        Run experiments with trained A2C models.

        Parameters:
        - trials (int): Number of trials for each player.

        Returns:
        - None
        """
        trials = trials
        for player in self.players:
            env = QuixoEnv(player=player)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                model = A2C.load(f"{self.models_dir}/player_{player}/quixo-a2c_{steps}_steps", env=env)
            winners = 0
            done = False
            f = 0
            for i in range(trials):
                obs = env.reset()
                done = False
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    moves = env.get_moves(env.opposite)
                    if action not in moves:
                        action = random.choice(moves)
                        f += 1
                    obs, rewards, done, info = env.step(action)
                if player == 0 and info["winner"] == 0:
                    winners += 1
                elif player == 1 and info["winner"] == 1:
                    winners += 1
            print("\n")
            if player == 0:
                print("playing first")
            else:
                print("playing second")
            print(f"win percentage: {winners/trials}")
            print(f"total wrong actions: {f}\n")
            print("-----")
