import itertools
from game import Game, Player, Move
from tqdm.auto import tqdm
import json


def acceptable_slides(from_position: tuple[int, int]):
    """
    Returns the possible moves (slides) when taking a piece from a given position.

    Parameters:
    - from_position (tuple): The position from which the piece is taken.

    Returns:
    - List[Move]: List of acceptable moves (slides).
    """
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
    """
    Play a game between two players.

    Parameters:
    - eplayer: The player to evaluate.
    - rplayer: The opponent player.

    Returns:
    - None
    """
    g = Game()
    winner = g.play(rplayer, eplayer)
    print(f"Winner: Player {winner}")


def get_available_actions():
    """
    Get all available actions on the game board.

    Returns:
    - List[tuple]: List of available actions as tuples (position, move).
    """
    pos_ranges = [range(0, 5), range(0, 5)]
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
    """
    Map a discrete action value to an actual action.

    Parameters:
    - discrete_value (int): Discrete action value.
    - available_actions (List[tuple]): List of available actions.

    Returns:
    - tuple: The mapped action as a tuple (position, move).
    """
    assert discrete_value < len(available_actions)
    assert discrete_value >= 0
    assert type(discrete_value) == int
    return available_actions[discrete_value]


def save_dict(file: dict, path: str) -> None:
    """
    Save a dictionary to a file.

    Parameters:
    - file (dict): Dictionary to be saved.
    - path (str): File path to save the dictionary.

    Returns:
    - None
    """
    with open(path, 'w') as out:
        json.dump(file, out)


def load_dict(path: str) -> dict:
    """
    Load a dictionary from a file.

    Parameters:
    - path (str): File path to load the dictionary from.

    Returns:
    - dict: Loaded dictionary.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def evaluate(
    eplayer: Player,
    opponent: Player,
    enum: int = 10,
):
    """
    Evaluate the performance of a player against an opponent.

    Parameters:
    - eplayer (Player): Player to evaluate.
    - opponent (Player): Opponent player.
    - enum (int): Number of evaluation trials.

    Returns:
    - None
    """
    win1 = 0
    win2 = 0
    with tqdm(total=2 * enum) as pbar:

        for i in range(enum):
            g = Game()
            winner = g.play(eplayer, opponent)
            if winner == 0:
                win1 += 1
            pbar.update(1)

        for i in range(enum):
            g = Game()
            winner = g.play(opponent, eplayer)
            if winner == 1:
                win2 += 1
            pbar.update(1)

    print(f"TOTAL # of trials:             {2 * enum}")
    print(f"WIN RATIO -- PLAYING FIRST:    {round(win1 / enum, 3)}")
    print(f"WIN RATIO -- PLAYING SECOND:   {round(win2 / enum, 3)}")
    print(f"TOTAL WIN RATIO:               {round((win2 + win1) / (2 * enum), 3)}")
