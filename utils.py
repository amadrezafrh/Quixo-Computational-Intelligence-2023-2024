#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:23:26 2024

@author: ahmadrezafrh
"""

from game import Move, Game, Player
from tqdm import tqdm
import itertools
import json

def acceptable_slides(from_position: tuple[int, int]):
    """When taking a piece from {from_position} returns the possible moves (slides)"""
    acceptable_slides = [Move.BOTTOM, Move.TOP, Move.LEFT, Move.RIGHT]
    axis_0 = from_position[0]    # axis_0 = 0 means uppermost row
    axis_1 = from_position[1]    # axis_1 = 0 means leftmost column

    if axis_0 == 0:  # can't move upwards if in the top row...
        acceptable_slides.remove(Move.TOP)
    elif axis_0 == 4:
        acceptable_slides.remove(Move.BOTTOM)

    if axis_1 == 0:
        acceptable_slides.remove(Move.LEFT)
    elif axis_1 == 4:
        acceptable_slides.remove(Move.RIGHT)
    return acceptable_slides


def play_game(eplayer, rplayer):
    g = Game()
    winner = g.play(rplayer, eplayer)
    print(f"Winner: Player {winner}")
    
    
def get_available_actions():
    pos_ranges = [range(0,5),range(0,5)]
    all_pos = list(itertools.product(*pos_ranges))
    available_actions = []
    for pos in all_pos:
        row, col = pos
        from_border = row in (0, 4) or col in (0, 4)
        if from_border:
            available_moves = acceptable_slides((pos[1], pos[0]))
            for mov in available_moves:
                available_actions.append((pos, mov))
    return available_actions


def map_actions(discrete_value, available_actions):
    assert discrete_value<len(available_actions)
    assert discrete_value>=0
    assert type(discrete_value)==int
    return available_actions[discrete_value]

def save_dict(file: dict, path: str) -> dict:
    with open(path, 'w') as out:
        json.dump(file, out)
        
def load_dict(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def evaluate(
    eplayer: Player,
    opponent: Player,
    enum: int = 10,
):
    win1 = 0
    win2 = 0
    with tqdm(total=2*enum) as pbar:

        for i in range(enum):
            g = Game()
            winner = g.play(eplayer, opponent)
            if winner==0:
                win1+=1
            pbar.update(1)
            
        for i in range(enum):
            g = Game()
            winner = g.play(opponent, eplayer)
            if winner==1:
                win2+=1
            pbar.update(1)
            
    print(f"TOTAL # of trials:             {2*enum}")
    print(f"WIN RATIO -- PLAYING SECOND:   {round(win1/enum, 3)}")
    print(f"WIN RATIO -- PLAYING FIRST:    {round(win2/enum, 3)}")
    print(f"TOTAL WIN RATIO:               {round((win2+win1)/(2*enum), 3)}")