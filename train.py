#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:52:56 2024

@author: ahmadrezafrh
"""


from game import Game, Move, Player
from utils import play_game
from players import RandomPlayer, ValuePlayer, PPOPlayer
from models import ValueClass, PPOWrapper, DQNWrapper, A2CWrapper
from envs import QuixoEnv
from utils import evaluate
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'





if __name__ == "__main__":
    print("\nTRAINING PPO AGENT\n")
    ts=5e5
    callbacks=50000
    agent = PPOWrapper()
    agent.train(ts=ts, callbacks=callbacks, verbose=1)
    agent.experiment(steps=450000)
    print("\nTRAINING DQN AGENT\n")
    agent = DQNWrapper()
    agent.train(ts=ts, callbacks=callbacks, verbose=1)
    agent.experiment(steps=450000)
    print("\nTRAINING DQN AGENT\n")
    agent = A2CWrapper()
    agent.train(ts=ts, callbacks=callbacks, verbose=1)
    agent.experiment(steps=450000)