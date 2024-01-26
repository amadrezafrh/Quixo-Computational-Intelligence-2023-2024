#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:23:26 2024

@author: ahmadrezafrh
"""

import numpy as np
import random


from gym import Env
from gym.spaces import Discrete, Box
from game import Move
from utils import get_available_actions
from copy import deepcopy



class QuixoEnvV1(Env):
    def __init__(self):
        self.actions = get_available_actions()
        self.action_space = Discrete(len(self.actions))
        self.observation_space = Box(low=-30, high=30, shape=(len(self.actions)+5*5, ), dtype=np.int32)
        self.board = -np.ones((5,5), dtype=np.int8)
    
    
    def slide(self, pos, slide):
        axis_0 = pos[1]
        axis_1 = pos[0]
        
        if slide == Move.RIGHT:
            self.board[axis_0] = np.roll(self.board[axis_0], -1)
        elif slide == Move.LEFT:
            self.board[axis_0] = np.roll(self.board[axis_0], 1)
        elif slide == Move.BOTTOM:
            self.board[:, axis_1] = np.roll(self.board[:, axis_1], -1)
        elif slide == Move.TOP:
            self.board[:, axis_1] = np.roll(self.board[:, axis_1], 1)
        
    def get_moves(self, player):
        a_actions = []
        for i in range(len(self.actions)):
            if self.board[(self.actions[i][0][1], self.actions[i][0][0])]!=player:
                a_actions.append(i)
        
        return a_actions
        
        
    def get_action_results(self):
        rewards = []
        count=0
        neg_count=0
        for i in range(len(self.actions)):
            pos = self.actions[i][0]
            mov = self.actions[i][1]
            if self.board[(pos[1], pos[0])]==0:
                rewards.append(-3)
            else:
                axis_0 = pos[1]
                axis_1 = pos[0]
                cp_board = deepcopy(self.board)
                cp_board[(pos[1], pos[0])] = 0
                
                if mov == Move.RIGHT:
                    cp_board[axis_0] = np.roll(cp_board[axis_0], -1)
                elif mov == Move.LEFT:
                    cp_board[axis_0] = np.roll(cp_board[axis_0], 1)
                elif mov == Move.BOTTOM:
                    cp_board[:, axis_1] = np.roll(cp_board[:, axis_1], -1)
                elif mov == Move.TOP:
                    cp_board[:, axis_1] = np.roll(cp_board[:, axis_1], 1)
                
                winner = self.check_winner(cp_board, 1)
                if winner==0:
                    rewards.append(-1)
                    neg_count += 1
                elif winner==1:
                    rewards.append(5)
                    count +=1
                else:
                    rewards.append(1)
                    
        return rewards, count, neg_count
        
    def step(self, action):
        mapped_action = self.actions[action]
        pos = mapped_action[0]
        mov = mapped_action[1]
        self.ep_count+=1
        
        if self.board[(pos[1], pos[0])] == 0:
            self.score = -5
            self.action_results = len(self.actions) * [0]
            self.observation = np.append(self.board.flatten(), np.array(self.action_results, dtype=np.int32))
            info = {}
            winner=-1
            info['winner'] = winner
            info['detail'] = "selected position in board taken"
            return self.observation, self.score, self.done,info
        
        self.board[(pos[1], pos[0])] = 1
        self.slide(pos, mov)
    
        winner = self.check_winner(self.board, 1)
        if winner==1:
            if self.ep_count<10:
                self.score = 100
            elif self.ep_count<20:
                self.score = 80
            elif self.ep_count<30:
                self.score = 60
            elif self.ep_count<40:
                self.score = 40
            elif self.ep_count<60:
                self.score = 20  
            else:
                self.score = 10
            self.action_results = len(self.actions) * [0]
            self.done = True
            info = {}
            self.observation = np.append(self.board.flatten(), np.array(self.action_results, dtype=np.int32))
            info['winner'] = winner
            info['detail'] = "player 1 won with his own move"
            return self.observation, self.score, self.done, info
        
        elif winner==0:
            self.score = -10
            self.action_results = len(self.actions) * [0]
            self.done = True
            info = {}
            self.observation = np.append(self.board.flatten(), np.array(self.action_results, dtype=np.int32))
            info['winner'] = winner
            info['detail'] = "player 0 won with oponent move"
            return self.observation, self.score, self.done, info
            
            
            
        
        op_actions = self.get_moves(1)
        op_action = random.choice(op_actions)
        op_pos, op_mov = self.actions[op_action]
        self.board[(op_pos[1], op_pos[0])] = 0
        self.slide(op_pos, op_mov)
        winner = self.check_winner(self.board, 0)
        if winner==0:
            self.score = -5
            self.action_results = len(self.actions) * [-10]
            self.done = True
            info = {}
            self.observation = np.append(self.board.flatten(), np.array(self.action_results, dtype=np.int32))
            info['winner'] = winner
            info['detail'] = "player 0 won with his own move"
            return self.observation, self.score, self.done, info
        
        elif winner==1:
            self.score = 0
            self.action_results = len(self.actions) * [-2]
            self.done = True
            info = {}
            self.observation = np.append(self.board.flatten(), np.array(self.action_results, dtype=np.int32))
            info['winner'] = winner
            info['detail'] = "player 1 won with oponent move"
            return self.observation, self.score, self.done, info
        
        
        self.action_results, count, neg_count = self.get_action_results()
        if self.ep_count>=35:
            self.score = 1 + count + (35 - self.ep_count)
        else:
            self.score = 1 + count
        info = {}
        self.observation = np.append(self.board.flatten(), np.array(self.action_results, dtype=np.int32))
        winner=-1
        info['winner'] = winner
        info['detail'] = "no one won"
        return self.observation, self.score, self.done, info
        

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
    
    
    
    def reset(self):
        self.done = False
        self.ep_count = 0
        self.score = 0
        self.board = -np.ones((5,5), dtype=np.int32)
        self.action_results = len(self.actions)*[1]
        self.observation = np.append(self.board.flatten(), np.array(self.action_results, dtype=np.int32))
        return self.observation
    
    def update_board(self, board):
        self.board = deepcopy(board)
        
    def get_obs(self):
        action_results,_,_ = self.get_action_results()
        observation = np.append(self.board.flatten(), np.array(action_results, dtype=np.int32))
        return observation
    
    
    
    
    

class QuixoEnv(Env):
    def __init__(self, player: int):
        """
        Quixo Environment class.

        Parameters:
        - player (int): Player ID (0 or 1).

        Attributes:
        - actions (List[tuple]): List of available actions.
        - action_space (Discrete): Discrete action space.
        - observation_space (Box): Observation space.
        - board (np.ndarray): Game board.
        - player (int): Current player ID.
        - opposite (int): Opponent player ID.
        - ep_count (int): Episode count.
        - done (bool): Flag indicating if the episode is done.
        - score (int): Score of the current state.
        - observation (np.ndarray): Current observation state.
        """
        self.actions = get_available_actions()
        self.action_space = Discrete(len(self.actions))
        self.observation_space = Box(
            low=-30, high=30, shape=(len(self.actions) + 5 * 5,), dtype=np.int32)
        self.board = -np.ones((5, 5), dtype=np.int8)
        self.player = player
        self.opposite = 0 if self.player == 1 else 1
        if self.player == 1:
            result = self.op_move()
            assert result == None

    def slide(self, pos, slide):
        """
        Slide a piece on the game board.

        Parameters:
        - pos (tuple): Position of the piece.
        - slide (Move): Slide direction.

        Returns:
        - None
        """
        axis_0 = pos[1]
        axis_1 = pos[0]

        if slide == Move.RIGHT:
            self.board[axis_0] = np.roll(self.board[axis_0], -1)
        elif slide == Move.LEFT:
            self.board[axis_0] = np.roll(self.board[axis_0], 1)
        elif slide == Move.BOTTOM:
            self.board[:, axis_1] = np.roll(self.board[:, axis_1], -1)
        elif slide == Move.TOP:
            self.board[:, axis_1] = np.roll(self.board[:, axis_1], 1)

    def get_moves(self, player):
        """
        Get available moves for a player.

        Parameters:
        - player (int): Player ID.

        Returns:
        - List[int]: List of available move indices.
        """
        a_actions = []
        for i in range(len(self.actions)):
            if self.board[(self.actions[i][0][1], self.actions[i][0][0])] != player:
                a_actions.append(i)

        return a_actions

    def get_action_results(self):
        """
        Get results of available actions.

        Returns:
        - Tuple[List[int], int, int]: Tuple containing rewards, count, and neg_count.
        """
        rewards = []
        count = 0
        neg_count = 0
        for i in range(len(self.actions)):
            pos = self.actions[i][0]
            mov = self.actions[i][1]
            if self.board[(pos[1], pos[0])] == self.player:
                rewards.append(0)
            else:
                axis_0 = pos[1]
                axis_1 = pos[0]
                cp_board = deepcopy(self.board)
                cp_board[(pos[1], pos[0])] = self.player

                if mov == Move.RIGHT:
                    cp_board[axis_0] = np.roll(cp_board[axis_0], -1)
                elif mov == Move.LEFT:
                    cp_board[axis_0] = np.roll(cp_board[axis_0], 1)
                elif mov == Move.BOTTOM:
                    cp_board[:, axis_1] = np.roll(cp_board[:, axis_1], -1)
                elif mov == Move.TOP:
                    cp_board[:, axis_1] = np.roll(cp_board[:, axis_1], 1)

                winner = self.check_winner(cp_board, self.player)
                if winner == self.opposite:
                    rewards.append(-1)
                    neg_count += 1
                elif winner == self.player:
                    rewards.append(2)
                    count += 1
                else:
                    rewards.append(1)

        return rewards, count, neg_count

    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
        - action (int): Action index.

        Returns:
        - Tuple[np.ndarray, int, bool, dict]: Tuple containing the next observation, reward, done flag, and info.
        """
        self.ep_count += 1
        result = self.player_move(action)
        if result:
            return result
        result = self.op_move()
        if result:
            return result

        self.action_results, count, neg_count = self.get_action_results()

        if self.ep_count >= 35:
            self.score = 1 + count + (35 - self.ep_count)
        else:
            self.score = 1 + count

        info = {}
        self.observation = np.append(self.board.flatten(), np.array(
            self.action_results, dtype=np.int32))
        winner = -1
        info['winner'] = winner
        info['detail'] = "no one won"
        return self.observation, self.score, self.done, info

    def player_move(self, action):
        """
        Perform a move by the player.

        Parameters:
        - action (int): Action index.

        Returns:
        - Tuple[np.ndarray, int, bool, dict]: Tuple containing the next observation, reward, done flag, and info.
        """
        mapped_action = self.actions[action]
        pos = mapped_action[0]
        mov = mapped_action[1]

        if self.board[(pos[1], pos[0])] == self.opposite:
            self.score = -15
            self.action_results = len(self.actions) * [0]
            self.observation = np.append(self.board.flatten(), np.array(
                self.action_results, dtype=np.int32))
            info = {}
            winner = -1
            info['winner'] = winner
            info['detail'] = "selected position in board taken"
            return self.observation, self.score, self.done, info

        self.board[(pos[1], pos[0])] = self.player
        self.slide(pos, mov)

        winner = self.check_winner(self.board, self.player)
        if winner == self.player:
            if self.ep_count<10:
                self.score = 110
            if self.ep_count < 20:
                self.score = 80
            elif self.ep_count < 30:
                self.score = 60
            elif self.ep_count < 40:
                self.score = 40
            elif self.ep_count < 60:
                self.score = 20
            else:
                self.score = 10
            self.action_results = len(self.actions) * [20]
            self.done = True
            info = {}
            self.observation = np.append(self.board.flatten(), np.array(
                self.action_results, dtype=np.int32))
            info['winner'] = winner
            info['detail'] = f"player {self.player} won with his own move"
            return self.observation, self.score, self.done, info

        elif winner == self.opposite:
            self.score = -10
            self.action_results = len(self.actions) * [15]
            self.done = True
            info = {}
            self.observation = np.append(self.board.flatten(), np.array(
                self.action_results, dtype=np.int32))
            info['winner'] = winner
            info['detail'] = f"player {self.opposite} won with oponent move"
            return self.observation, self.score, self.done, info

        return None

    def op_move(self):
        """
        Perform a move by the opponent.

        Returns:
        - Tuple[np.ndarray, int, bool, dict]: Tuple containing the next observation, reward, done flag, and info.
        """
        op_actions = self.get_moves(self.player)
        op_action = random.choice(op_actions)
        op_pos, op_mov = self.actions[op_action]
        self.board[(op_pos[1], op_pos[0])] = self.opposite
        self.slide(op_pos, op_mov)
        winner = self.check_winner(self.board, self.opposite)
        if winner == self.opposite:
            self.score = -5
            self.action_results = len(self.actions) * [-10]
            self.done = True
            info = {}
            self.observation = np.append(self.board.flatten(), np.array(
                self.action_results, dtype=np.int32))
            info['winner'] = winner
            info['detail'] = f"player {self.opposite} won with his own move"
            return self.observation, self.score, self.done, info

        elif winner == self.player:
            self.score = 0
            self.action_results = len(self.actions) * [-5]
            self.done = True
            info = {}
            self.observation = np.append(self.board.flatten(), np.array(
                self.action_results, dtype=np.int32))
            info['winner'] = winner
            info['detail'] = f"player {self.player} won with oponent move"
            return self.observation, self.score, self.done, info

        return None

    def check_winner(self, board, player_id) -> int:
        '''
        Check the winner. Returns the player ID of the winner if any, otherwise returns -1.

        Parameters:
        - board (np.ndarray): Game board.
        - player_id (int): Player ID.

        Returns:
        - int: Player ID of the winner or -1 if no winner.
        '''
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

    def reset(self):
        """
        Reset the environment.

        Returns:
        - np.ndarray: Initial observation state.
        """
        self.done = False
        self.ep_count = 0
        self.score = 0
        self.board = -np.ones((5, 5), dtype=np.int32)
        self.action_results = len(self.actions) * [1]
        if self.player == 1:
            result = self.op_move()
            assert result == None
            self.action_results, _, _ = self.get_action_results()

        self.observation = np.append(self.board.flatten(), np.array(
            self.action_results, dtype=np.int32))
        return self.observation

    def update_board(self, board):
        """
        Update the game board.

        Parameters:
        - board (np.ndarray): New game board.

        Returns:
        - None
        """
        self.board = deepcopy(board)

    def get_obs(self):
        action_results, _, _ = self.get_action_results()
        observation = np.append(self.board.flatten(), np.array(
            action_results, dtype=np.int32))
        return observation

    def experiment(self):
        t = 0
        observation = self.reset()
        tot_reward = 0
        while True:
            t += 1
            action = self.action_space.sample()
            observation, reward, done, info = self.step(action)
            tot_reward += reward
            if done:
                break

        print(f"episode length: {t}")
        print(f"total reward: {tot_reward}")
        print(f"winner: player {info['winner']}")
