
import random
import itertools
import numpy as np
import os
import contextlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stable_baselines3 import DQN, PPO, SAC, TD3, A2C
from copy import deepcopy
from game import Player, Move, Game
from utils import acceptable_slides
from envs import QuixoEnv


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
    
    
    
    
class ValuePlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        
        self.pos_ranges = [range(0,5),range(0,5)]
        self.all_pos = list(itertools.product(*self.pos_ranges))
        self.available_pos = []
        for pos in self.all_pos:
            row, col = pos
            from_border = row in (0, 4) or col in (0, 4)
            if from_border:
                self.available_pos.append(pos)
                
        self.set_trajectory()
                
    def set_trajectory(self):
        self.trajectory = []
        
    def get_moves(self, game):
        available_actions = []
        for pos in self.available_pos:
            new_pos = deepcopy((pos[1], pos[0]))
            if game._board[new_pos] != 1 and game._board[new_pos] != -1:
                continue
            available_slides = acceptable_slides(new_pos)
            for slide in available_slides:
                available_actions.append((pos, slide))
        return available_actions
        
        
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        moves = self.get_moves(game)
        from_pos, move = random.choice(moves)
        self.trajectory.append(deepcopy(game._board))
        return from_pos, move




class PPOPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.env0 = QuixoEnv(player=0)
        self.env1 = QuixoEnv(player=1)
        self.env0.reset()
        self.env1.reset()
        
        self.models = []
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.models.append(PPO.load("./models/ppo/quixo-ppo-0", env=self.env0))
            self.models.append(PPO.load("./models/ppo/quixo-ppo-1", env=self.env1))
        

    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        if game.current_player_idx==0:
            action = self.get_action(game, self.env0, self.models[0])
        else:
            action = self.get_action(game, self.env1, self.models[1])
        return action
    
    def get_action(self, game, env, model):
        opposite = 0 if env.player==1 else 1
        env.update_board(game._board)
        obs = env.get_obs()
        action, _ = model.predict(obs, deterministic=True)
        env.update_board(game._board)
        moves = env.get_moves(opposite)
        if action not in moves:
            action = random.choice(moves)
        return env.actions[action]
    
    
    
class DeterministicPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.pos_ranges = [range(0,5),range(0,5)]
        self.all_pos = list(itertools.product(*self.pos_ranges))
        self.available_pos = []
        for pos in self.all_pos:
            row, col = pos
            from_border = row in (0, 4) or col in (0, 4)
            if from_border:
                self.available_pos.append(pos)
        
    def check_winner(self, board, player_id) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        player = player_id
        winner = -1
        for x in range(board.shape[0]):
            if board[x, 0] != -1 and all(board[x, :] == board[x, 0]):
                winner = board[x, 0]
        if winner > -1 and winner != player:
            return winner
        for y in range(board.shape[1]):
            if board[0, y] != -1 and all(board[:, y] == board[0, y]):
                winner = board[0, y]
        if winner > -1 and winner != player:
            return winner
        if board[0, 0] != -1 and all(
            [board[x, x]
                for x in range(board.shape[0])] == board[0, 0]
        ):
            winner = board[0, 0]
        if winner > -1 and winner != player:
            return winner
        if board[0, -1] != -1 and all(
            [board[x, -(x + 1)]
             for x in range(board.shape[0])] == board[0, -1]
        ):
            winner = board[0, -1]
        return winner
    
    
    def get_moves(self, game):
        available_actions = []
        for pos in self.available_pos:
            new_pos = deepcopy((pos[1], pos[0]))
            if game._board[new_pos] != game.current_player_idx and game._board[new_pos] != -1:
                continue
            available_slides = acceptable_slides(new_pos)
            for slide in available_slides:
                available_actions.append((pos, slide))
        return available_actions
    
    
    
    def get_action_results(self, game, available_actions):
        no_wins = []
        for action in available_actions:
            pos = action[0]
            mov = action[1]
            axis_0 = pos[1]
            axis_1 = pos[0]
            cp_board = deepcopy(game._board)
            cp_board[(pos[1], pos[0])] = game.current_player_idx

            if mov == Move.RIGHT:
                cp_board[axis_0] = np.roll(cp_board[axis_0], -1)
            elif mov == Move.LEFT:
                cp_board[axis_0] = np.roll(cp_board[axis_0], 1)
            elif mov == Move.BOTTOM:
                cp_board[:, axis_1] = np.roll(cp_board[:, axis_1], -1)
            elif mov == Move.TOP:
                cp_board[:, axis_1] = np.roll(cp_board[:, axis_1], 1)

            winner = self.check_winner(cp_board, game.current_player_idx)
            if winner==game.current_player_idx:
                return action
            if winner==-1:
                no_wins.append(action)
                
        
        return no_wins
                    
    
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        moves = self.get_moves(game)
        best_actions = self.get_action_results(game, moves)
        if type(best_actions)==list:
            action = random.choice(best_actions)
        else:
            action = best_actions
            
        return action