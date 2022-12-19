import random
from typing import List, Union
import numpy as np
import Gobblet_Gobblers_Env as gge
from abc import ABC, abstractmethod
from collections import defaultdict


NOT_ON_BOARD = np.array([-1, -1])

# Piece weights & identifiers:
# ----------------------------
B_PAWN = 5
M_PAWN = 3
S_PAWN = 1

PAWN_MAP = {
    'B': B_PAWN,
    'M': M_PAWN,
    'S': S_PAWN
}


# Utilities:
# ----------
def get_board(state: gge.State, agent_id: int) -> np.ndarray:
    board = np.zeros((3, 3, 3))  # Create a 3x3 board that can hold up to 3 pawns in each slot
    next_index = defaultdict(int)  # For each cell save the next index to 

    pawns_on_board = [pawn for pawn in sorted(
        list({f'{key}_0': val[0] for key, val in state.player1_pawns.items()}.items()) 
        + list({f'{key}_1': val[0] for key, val in state.player2_pawns.items()}.items())
    ) if np.all(pawn[1] != NOT_ON_BOARD)]  # Calculate all the pawns that are currently on the board

    for pawn in pawns_on_board:
        player = int(pawn[0][-1])  # The last letter in the pawn name is the player ID
        multiplier = 1 if player == agent_id else -1  # The score is negative if this is the enemy's pawn
        pawn_location = (pawn[1][0], pawn[1][1])
        board[pawn_location][next_index[pawn_location]] = multiplier * PAWN_MAP[pawn[0][0]]
        next_index[pawn_location] += 1
    return board


# Heuristics:
# -----------
class Heuristic(ABC):
    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        return

class WeightedPawnPosLayerDecay(Heuristic):
    """
    A heuristic for Gobblet Gobblers.

    Weighs pawns by a predetermined weight, it's location on the board,
    and decays each pawn according to the layer in which it resides.
    """
    def __init__(self, layer_decay: float=0.2):
        self._layer_decay_mask: np.ndarray = np.ones((3, 3, 3))
        for i in range(1, 3):
            self._layer_decay_mask[:, :, i:] *= layer_decay

        self._position_mask: np.ndarray = np.array([
            [[3, 3, 3], [1, 1, 1], [3, 3, 3]],
            [[1, 1, 1], [5, 5, 5], [1, 1, 1]],
            [[3, 3, 3], [1, 1, 1], [3, 3, 3]]
        ])

        # Estimate a win as the maximal (im)possible value.
        # Play with this value to get to a good balance (for Expectimax, for example)
        self._win_score = 600
    
    def evaluate(self, state: gge.State, agent_id: int):
        is_final = gge.is_final_state(state)
        if is_final is not None and is_final != 0:
            is_final = int(is_final)
            # If it is a final state and not a tie, someone won.
            if is_final - 1 == agent_id:
                return self._win_score
            return -self._win_score

        board = get_board(state, agent_id)
        return np.sum(board * self._layer_decay_mask * self._position_mask)

# agent_id is which player I am, 0 - for the first player , 1 - if second player
def dumb_heuristic1(state, agent_id):
    is_final = gge.is_final_state(state)
    # this means it is not a final state
    if is_final is None:
        return 0
    # this means it's a tie
    if is_final == 0:
        return -1
    # now convert to our numbers the win
    winner = int(is_final) - 1
    # now winner is 0 if first player won and 1 if second player won
    # and remember that agent_id is 0 if we are first player  and 1 if we are second player won
    if winner == agent_id:
        # if we won
        return 1
    else:
        # if other player won
        return -1


# checks if a pawn is under another pawn
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False


# count the numbers of pawns that i have that aren't hidden
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], NOT_ON_BOARD) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], NOT_ON_BOARD) and not is_hidden(state, agent_id, key):
                sum_pawns += 1

    return sum_pawns


def smart_heuristic(state: gge.State, agent_id: int):
    heuristic: WeightedPawnPosLayerDecay = WeightedPawnPosLayerDecay()
    return heuristic(state, agent_id)


# IMPLEMENTED FOR YOU - NO NEED TO CHANGE
def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


# agent_id is which agent you are - first player or second player
def random_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    rnd = random.randint(0, neighbor_list.__len__() - 1)
    return neighbor_list[rnd][0]


# TODO - instead of action to return check how to raise not_implemented
def greedy(curr_state: gge.State, agent_id: int, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = dumb_heuristic2(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


# TODO - add your code here
def greedy_improved(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = -np.inf
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    raise NotImplementedError()


def alpha_beta(curr_state, agent_id, time_limit):
    raise NotImplementedError()


def expectimax(curr_state, agent_id, time_limit):
    raise NotImplementedError()


# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    raise NotImplementedError()
