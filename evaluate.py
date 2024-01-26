#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:52:56 2024

@author: ahmadrezafrh
"""


from players import RandomPlayer, PPOPlayer, DQNPlayer, A2CPlayer, DeterministicPlayer
from utils import evaluate
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




if __name__ == "__main__":
    enum=1000
    print("\n\nEVALUATING 'DeterministicPlayer' AGENT AGAINST 'RandomPlayer'")
    opponent = RandomPlayer()
    eplayer = DeterministicPlayer()    
    evaluate(eplayer, opponent, enum=enum)
    
    
    
    print("\n\nEVALUATING 'PPOPlayer' AGENT AGAINST 'RandomPlayer'")
    opponent = RandomPlayer()
    eplayer = PPOPlayer()    
    evaluate(eplayer, opponent, enum=enum)
    
    
    
    print("\n\nEVALUATING 'PPOPlayer' AGENT AGAINST 'DeterministicPlayer'")
    opponent = DeterministicPlayer()
    eplayer = PPOPlayer()    
    evaluate(eplayer, opponent, enum=enum)
    
    
    
    
    print("\n\nEVALUATING 'DQNPlayer' AGENT AGAINST 'RandomPlayer'")
    opponent = RandomPlayer()
    eplayer = DQNPlayer()    
    evaluate(eplayer, opponent, enum=enum)
    
    print("\n\nEVALUATING 'DQNPlayer' AGENT AGAINST 'DeterministicPlayer'")
    opponent = DeterministicPlayer()
    eplayer = DQNPlayer()    
    evaluate(eplayer, opponent, enum=enum)
    
    
    
    print("\n\nEVALUATING 'A2CPlayer' AGENT AGAINST 'RandomPlayer'")
    opponent = RandomPlayer()
    eplayer = A2CPlayer()    
    evaluate(eplayer, opponent, enum=enum)
    
    print("\n\nEVALUATING 'A2CPlayer' AGENT AGAINST 'DeterministicPlayer'")
    opponent = DeterministicPlayer()
    eplayer = A2CPlayer()    
    evaluate(eplayer, opponent, enum=enum)
    
    

    
    
    