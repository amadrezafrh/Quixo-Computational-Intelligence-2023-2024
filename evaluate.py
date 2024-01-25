#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:52:56 2024

@author: ahmadrezafrh
"""


from game import Game, Move, Player
from utils import play_game
from players import RandomPlayer, PPOPlayer, DeterministicPlayer
from models import ValueClass, PPOWrapper, DQNWrapper, A2CWrapper
from envs import QuixoEnv
from utils import evaluate
import os
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




if __name__ == "__main__":
    
    print("\n\nEVALUATING 'DeterministicPlayer' AGENT AGAINST 'RandomPlayer'")
    opponent = RandomPlayer()
    eplayer = DeterministicPlayer()    
    evaluate(eplayer, opponent, enum=1_000)
    
    
    
    print("\n\nEVALUATING 'PPOPlayer' AGENT AGAINST 'RandomPlayer'")
    opponent = RandomPlayer()
    eplayer = PPOPlayer()    
    evaluate(eplayer, opponent, enum=1_000)
    
    
    
    print("\n\nEVALUATING 'PPOPlayer' AGENT AGAINST 'DeterministicPlayer'")
    opponent = DeterministicPlayer()
    eplayer = PPOPlayer()    
    evaluate(eplayer, opponent, enum=1_000)