
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
        # Initialize the RandomPlayer, inheriting from the Player class
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Generate a random move for the player.

        Args:
            game ('Game'): The instance of the game.

        Returns:
            tuple[tuple[int, int], Move]: A tuple containing the starting position
            (randomly generated) and the chosen move (randomly selected from available options).
        """
        # Generate a random starting position within the bounds of the game grid
        from_pos = (random.randint(0, 4), random.randint(0, 4))

        # Choose a random move from the available options
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        # Return the selected move as a tuple containing the starting position and the move
        return from_pos, move
    
    
    
class ValuePlayer(Player):
    def __init__(self) -> None:
        # Initialize the ValuePlayer, inheriting from the Player class
        super().__init__()

        # Define position ranges and calculate all possible positions
        self.pos_ranges = [range(0, 5), range(0, 5)]
        self.all_pos = list(itertools.product(*self.pos_ranges))
        
        # Identify positions on the border and store them as available positions
        self.available_pos = []
        for pos in self.all_pos:
            row, col = pos
            from_border = row in (0, 4) or col in (0, 4)
            if from_border:
                self.available_pos.append(pos)
        
        # Initialize the trajectory list
        self.set_trajectory()

    def set_trajectory(self):
        # Reset the trajectory list
        self.trajectory = []

    def get_moves(self, game):
        """
        Get available moves for the player.

        Args:
            game ('Game'): The instance of the game.

        Returns:
            list: List of available actions, each represented as a tuple (position, move).
        """
        available_actions = []
        for pos in self.available_pos:
            new_pos = deepcopy((pos[1], pos[0]))
            if game._board[new_pos] != 1 and game._board[new_pos] != -1:
                continue
            available_slides = acceptable_slides(new_pos)  # Assuming acceptable_slides is defined somewhere
            for slide in available_slides:
                available_actions.append((pos, slide))
        return available_actions

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Make a move for the player.

        Args:
            game ('Game'): The instance of the game.

        Returns:
            tuple[tuple[int, int], Move]: A tuple containing the starting position
            and the chosen move (randomly selected from available options).
        """
        moves = self.get_moves(game)
        from_pos, move = random.choice(moves)
        
        # Record the current state of the game board in the trajectory
        self.trajectory.append(deepcopy(game._board))
        
        return from_pos, move



class PPOPlayer(Player):
    def __init__(self, models_path: str = "./models/ppo", steps=450000) -> None:
        """
        Initialize PPOPlayer.

        Args:
            models_path (str): The path to the PPO models.
        """
        super().__init__()

        # Initialize two Quixo environments for players 0 and 1
        self.env0 = QuixoEnv(player=0)
        self.env1 = QuixoEnv(player=1)
        self.env0.reset()
        self.env1.reset()

        # Load PPO models for both players
        self.models = []
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.models.append(PPO.load(f"{models_path}/player_0/quixo-ppo_{steps}_steps", env=self.env0))
            self.models.append(PPO.load(f"{models_path}/player_0/quixo-ppo_{steps}_steps", env=self.env1))

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Make a move for the player.

        Args:
            game ('Game'): The instance of the game.

        Returns:
            tuple[tuple[int, int], Move]: A tuple containing the starting position
            and the chosen move.
        """
        if game.current_player_idx == 0:
            action = self.get_action(game, self.env0, self.models[0])
        else:
            action = self.get_action(game, self.env1, self.models[1])
        return action

    def get_action(self, game, env, model):
        """
        Get the action to perform based on the current game state.

        Args:
            game ('Game'): The instance of the game.
            env ('QuixoEnv'): The Quixo environment for the player.
            model ('PPO'): The PPO model for the player.

        Returns:
            tuple[tuple[int, int], Move]: A tuple containing the starting position
            and the chosen move.
        """
        opposite = 0 if env.player == 1 else 1

        # Update the Quixo environment with the current game board
        env.update_board(game._board)
        obs = env.get_obs()

        # Predict an action using the PPO model
        action, _ = model.predict(obs, deterministic=True)

        # Update the Quixo environment with the current game board again
        env.update_board(game._board)

        # Get available moves for the opposite player
        moves = env.get_moves(opposite)

        # If the predicted action is not valid, choose a random move
        if action not in moves:
            action = random.choice(moves)

        return env.actions[action]




class DQNPlayer(Player):
    def __init__(self, models_path: str = "./models/dqn", steps=450000) -> None:
        """
        Initialize DQNPlayer.

        Args:
            models_path (str): The path to the DQN models.
        """
        super().__init__()

        # Initialize two Quixo environments for players 0 and 1
        self.env0 = QuixoEnv(player=0)
        self.env1 = QuixoEnv(player=1)
        self.env0.reset()
        self.env1.reset()

        # Load DQN models for both players
        self.models = []
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.models.append(DQN.load(f"{models_path}/player_0/quixo-dqn_{steps}_steps", env=self.env0))
            self.models.append(DQN.load(f"{models_path}/player_1/quixo-dqn_{steps}_steps", env=self.env1))

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Make a move for the player.

        Args:
            game ('Game'): The instance of the game.

        Returns:
            tuple[tuple[int, int], Move]: A tuple containing the starting position
            and the chosen move.
        """
        if game.current_player_idx == 0:
            action = self.get_action(game, self.env0, self.models[0])
        else:
            action = self.get_action(game, self.env1, self.models[1])
        return action

    def get_action(self, game, env, model):
        """
        Get the action to perform based on the current game state.

        Args:
            game ('Game'): The instance of the game.
            env ('QuixoEnv'): The Quixo environment for the player.
            model ('DQN'): The DQN model for the player.

        Returns:
            tuple[tuple[int, int], Move]: A tuple containing the starting position
            and the chosen move.
        """
        opposite = 0 if env.player == 1 else 1

        # Update the Quixo environment with the current game board
        env.update_board(game._board)
        obs = env.get_obs()

        # Predict an action using the DQN model
        action, _ = model.predict(obs, deterministic=True)

        # Update the Quixo environment with the current game board again
        env.update_board(game._board)

        # Get available moves for the opposite player
        moves = env.get_moves(opposite)

        # If the predicted action is not valid, choose a random move
        if action not in moves:
            action = random.choice(moves)

        return env.actions[action]


class A2CPlayer(Player):
    def __init__(self, models_path: str = "./models/a2c", steps=450000) -> None:
        """
        Initialize A2CPlayer.

        Args:
            models_path (str): The path to the A2C models.
        """
        super().__init__()

        # Initialize two Quixo environments for players 0 and 1
        self.env0 = QuixoEnv(player=0)
        self.env1 = QuixoEnv(player=1)
        self.env0.reset()
        self.env1.reset()

        # Load A2C models for both players
        self.models = []
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.models.append(A2C.load(f"{models_path}/player_0/quixo-a2c_{steps}_steps", env=self.env0))
            self.models.append(A2C.load(f"{models_path}/player_0/quixo-a2c_{steps}_steps", env=self.env1))

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Make a move for the player.

        Args:
            game ('Game'): The instance of the game.

        Returns:
            tuple[tuple[int, int], Move]: A tuple containing the starting position
            and the chosen move.
        """
        if game.current_player_idx == 0:
            action = self.get_action(game, self.env0, self.models[0])
        else:
            action = self.get_action(game, self.env1, self.models[1])
        return action

    def get_action(self, game, env, model):
        """
        Get the action to perform based on the current game state.

        Args:
            game ('Game'): The instance of the game.
            env ('QuixoEnv'): The Quixo environment for the player.
            model ('A2C'): The A2C model for the player.

        Returns:
            tuple[tuple[int, int], Move]: A tuple containing the starting position
            and the chosen move.
        """
        opposite = 0 if env.player == 1 else 1

        # Update the Quixo environment with the current game board
        env.update_board(game._board)
        obs = env.get_obs()

        # Predict an action using the A2C model
        action, _ = model.predict(obs, deterministic=True)

        # Update the Quixo environment with the current game board again
        env.update_board(game._board)

        # Get available moves for the opposite player
        moves = env.get_moves(opposite)

        # If the predicted action is not valid, choose a random move
        if action not in moves:
            action = random.choice(moves)

        return env.actions[action]




class DeterministicPlayer(Player):
    def __init__(self) -> None:
        """
        Initialize DeterministicPlayer.
        """
        super().__init__()

        # Define position ranges and calculate all possible positions
        self.pos_ranges = [range(0, 5), range(0, 5)]
        self.all_pos = list(itertools.product(*self.pos_ranges))

        # Identify positions on the border and store them as available positions
        self.available_pos = []
        for pos in self.all_pos:
            row, col = pos
            from_border = row in (0, 4) or col in (0, 4)
            if from_border:
                self.available_pos.append(pos)

    def check_winner(self, board, player_id) -> int:
        """
        Check the winner of the game.

        Args:
            board (numpy.ndarray): The game board.
            player_id (int): The player ID.

        Returns:
            int: The player ID of the winner if any, otherwise returns -1.
        """
        player = player_id
        winner = -1

        # Check rows for a winner
        for x in range(board.shape[0]):
            if board[x, 0] != -1 and all(board[x, :] == board[x, 0]):
                winner = board[x, 0]

        if winner > -1 and winner != player:
            return winner

        # Check columns for a winner
        for y in range(board.shape[1]):
            if board[0, y] != -1 and all(board[:, y] == board[0, y]):
                winner = board[0, y]

        if winner > -1 and winner != player:
            return winner

        # Check diagonals for a winner (top-left to bottom-right)
        if board[0, 0] != -1 and all(
            [board[x, x] for x in range(board.shape[0])] == board[0, 0]
        ):
            winner = board[0, 0]

        if winner > -1 and winner != player:
            return winner

        # Check diagonals for a winner (top-right to bottom-left)
        if board[0, -1] != -1 and all(
            [board[x, -(x + 1)] for x in range(board.shape[0])] == board[0, -1]
        ):
            winner = board[0, -1]

        return winner

    def get_moves(self, game):
        """
        Get available moves for the player.

        Args:
            game ('Game'): The instance of the game.

        Returns:
            list: List of available actions, each represented as a tuple (position, move).
        """
        available_actions = []
        for pos in self.available_pos:
            new_pos = deepcopy((pos[1], pos[0]))
            if (
                game._board[new_pos] != game.current_player_idx
                and game._board[new_pos] != -1
            ):
                continue
            available_slides = acceptable_slides(new_pos)  # Assuming acceptable_slides is defined somewhere
            for slide in available_slides:
                available_actions.append((pos, slide))
        return available_actions

    def get_action_results(self, game, available_actions):
        """
        Get the result of each available action.

        Args:
            game ('Game'): The instance of the game.
            available_actions (list): List of available actions.

        Returns:
            list: List of actions that lead to a win for the current player.
        """
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
            if winner == game.current_player_idx:
                return action
            if winner == -1:
                no_wins.append(action)

        return no_wins

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """
        Make a move for the player.

        Args:
            game ('Game'): The instance of the game.

        Returns:
            tuple[tuple[int, int], Move]: A tuple containing the starting position
            and the chosen move.
        """
        moves = self.get_moves(game)
        best_actions = self.get_action_results(game, moves)
        if type(best_actions) == list:
            action = random.choice(best_actions)
        else:
            action = best_actions

        return action
