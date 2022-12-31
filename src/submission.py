import random
import threading as th
from typing import Callable, List, Tuple
import numpy as np
import Gobblet_Gobblers_Env as gge
from abc import ABC, abstractmethod
from collections import defaultdict


NOT_ON_BOARD = np.array([-1, -1])

# Piece weights & identifiers:
# ----------------------------
B_PAWN = 6
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


def is_action_eating_enemy_pawn(curr_state: gge.State, action: Tuple[str, int]) -> bool:
    board = get_board(curr_state, curr_state.turn)
    action_coords: Tuple[int, int] = tuple(gge.action_to_direction[action[1]])
    action_pawn: str = action[0][0]

    if not np.any(board[action_coords] != 0) or not board[action_coords][0] < 0:
        return False

    return PAWN_MAP[action_pawn] > np.abs(board[action_coords][0])


def is_action_involving_pawn_s(action: Tuple[str, int]) -> bool:
    return action[0][0] == 'S'


def get_event_value(curr_state: gge.State, action: Tuple[str, int]) -> float:
    BASE_VALUE = 1

    if is_action_eating_enemy_pawn(curr_state, action) or is_action_involving_pawn_s(action):
        BASE_VALUE *= 2

    return BASE_VALUE


# Heuristics:
# -----------
class MAHeuristic(ABC):
    def __call__(self, state: gge.State, agent_id: int, *args, **kwargs):
        return self.evaluate(state, agent_id, *args, **kwargs)

    @abstractmethod
    def evaluate(self, state: gge.State, agent_id: int, *args, **kwargs):
        return

class WeightedPawnPosLayerDecay(MAHeuristic):
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
        win_bonus = 0

        if is_final is not None and is_final != 0:
            is_final = int(is_final)
            # If it is a final state and not a tie, someone won.
            if is_final - 1 == agent_id:
                win_bonus = self._win_score
            else:
                win_bonus = -self._win_score

        board = get_board(state, agent_id)
        return np.sum(board * self._layer_decay_mask * self._position_mask) + win_bonus

HEURISTIC_OBJ = WeightedPawnPosLayerDecay()


# Agents:
# -------
class AnytimeAlgorithm(th.Thread):
    """
    Thread mixin class with a stop() method.
    The thread itself has to check regularly for the is_stopped() condition.
    """
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = th.Event()

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()


class Agent:
    """
    Agent mixin class.
    Provides common functionalities that agents commonly need.
    """
    expanded: int
    agent_id: int
    starting_state: gge.State

    def __init__(self, starting_state: gge.State, agent_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expanded = 0
        self.agent_id = agent_id
        self.starting_state = starting_state
    
    def get_starting_state(self) -> gge.State:
        return self.starting_state
    
    def set_starting_state(self, state: gge.State) -> None:
        self.starting_state = state

    def reset_agent(self):
        self.expanded = 0

    def expand(self, state: gge.State) -> List[Tuple[Tuple[str, int], gge.State]]:
        """
        Return:
        -------
        A list of neighbours, where every neighbour is a tuple of (action, next_state).
        An action is a tuple of the form (pawn, board_cell_index)
        """
        self.expanded += 1
        return state.get_neighbors()

    @classmethod
    def is_terminal_state(cls, state: gge.State) -> bool:
        """
        Returns whether the state is a terminal state or not.
        """
        return (gge.is_final_state(state) is not None)
    
    @classmethod
    def is_player_winner(cls, state: gge.State, player_id: int) -> bool:
        """
        Returns whether the player_id is the winner in the input state.
        """
        winner_id = gge.is_final_state(state)
        return (winner_id is not None and (int(winner_id) - 1) == player_id)
    
    @classmethod
    def get_neighbour_action(cls, neighbour: Tuple[Tuple[str, int], gge.State]) -> Tuple[str, int]:
        """
        Extracts an action from a neighbour in the list received from expand().

        Parameters:
        -----------
        neighbour : Tuple[Tuple[str, int], gge.State]
            A neighbour from the list of expand().

        Return:
        -------
        pawn: str, board_cell_index: int
            Returns the neighbour's action as a tuple of the form (pawn, board_cell_index).
            board_cell_index is the index of the board cell to which the pawn is to be moved onto.
        """
        return neighbour[0]
    
    @classmethod
    def get_neighbour_state(cls, neighbour: Tuple[Tuple[str, int], gge.State]) -> gge.State:
        """
        Extracts the state from a neighbour in the list received from expand().

        Parameters:
        -----------
        neighbour : Tuple[Tuple[str, int], gge.State]
            A neighbour from the list of expand().

        Return:
        -------
        pawn: str, board_cell_index: int
            Returns the neighbour's action as a tuple of the form (pawn, board_cell_index).
        """
        return neighbour[1]


class Minimax(Agent):
    _heuristic: MAHeuristic
    __action_to_curr_state: Tuple[str, int]

    def __init__(self, heuristic: MAHeuristic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heuristic = heuristic
        self.__action_to_curr_state = tuple()

    def should_abort(self) -> bool:
        return False
    
    def search_for_immediate_win(self, state: gge.State):
        neighbours = sorted(self.expand(state), 
                            key=lambda neighbour: self._heuristic(type(self).get_neighbour_state(neighbour), self.agent_id)
        )

        found_winning_action = False
        max_value = -np.inf
        best_action = None

        for neighbour in neighbours:
            neighbour_action = type(self).get_neighbour_action(neighbour)
            neighbour_state = type(self).get_neighbour_state(neighbour)

            if type(self).is_player_winner(neighbour_state, self.agent_id):
                found_winning_action = True
                value = self._heuristic(neighbour_state, self.agent_id)
                if value > max_value:
                    max_value = value
                    best_action = neighbour_action
        
        return found_winning_action, best_action
    
    def minimax(self, state: gge.State, depth_limit: int=np.inf, prune: bool=False) -> Tuple[Tuple[str, int], float]:
        """
        Calculates the next best action from the input state under a depth limit.

        Parameters:
        -----------
        state : gge.State
            The starting state to calculate the next best action from.
        
        depth_limit : int = np.inf
            The maximum depth to search in, should be initially at least 1.

        Return:
        -------
        action: (pawn, board_cell_index), value: float
            Returns the best action to take as a tuple of the form (pawn, board_cell_index), and 
            the value that action is guaranteed to achieve heuristically.
        """
        self.__action_to_curr_state = tuple()  # Reset the action take to the current state
        
        return self._minimax_aux(state, depth_limit, prune)
        
    
    def _minimax_aux(self, 
                     state: gge.State, 
                     depth_limit: int=np.inf, 
                     prune: bool=False,
                     alpha: float=-np.inf,
                     beta: float=np.inf) -> Tuple[Tuple[str, int], float]:
        assert depth_limit >= 0
        if self.should_abort():
            # Was decided to abort the search
            return self.__action_to_curr_state, 0
        
        if type(self).is_terminal_state(state) or depth_limit == 0:
            return self.__action_to_curr_state, self._heuristic(state, self.agent_id)
        
        self.__action_to_curr_state = tuple()  # Reset the action take to the current state
        
        if state.turn == self.agent_id:
            # This is the agent's turn so it wants to maximize it's value.
            neighbours = sorted(self.expand(state), 
                            key=lambda neighbour: self._heuristic(type(self).get_neighbour_state(neighbour), self.agent_id),
                            reverse=True
            )
            max_value = -np.inf
            best_action = random.choice(neighbours) if len(neighbours) else None
            found_winning_action = False

            for neighbour in neighbours:
                neighbour_action = type(self).get_neighbour_action(neighbour)
                neighbour_state = type(self).get_neighbour_state(neighbour)
                
                self.__action_to_curr_state = neighbour_action
                if self.should_abort():
                    # Was decided to abort the search
                    return self.__action_to_curr_state, 0

                if type(self).is_player_winner(neighbour_state, state.turn):
                    # If this player is going to win by doing this action, no need to check further.
                    # Will attempt to find the optimal winning action.
                    found_winning_action = True
                    value = self._heuristic(neighbour_state, self.agent_id)
                    if value > max_value:
                        max_value = value
                        best_action = neighbour_action
                    continue
                
                if not found_winning_action:
                    _, value = self._minimax_aux(state=neighbour_state,
                                                 depth_limit=depth_limit - 1,
                                                 prune=prune,
                                                 alpha=alpha,
                                                 beta=beta)
                    if value > max_value:
                        max_value = value
                        best_action = neighbour_action
                    
                    if prune:
                        alpha = max(max_value, alpha)
                        if max_value >= beta:
                            return best_action, np.inf
            
            return best_action, max_value
        
        else:
            # This isn't the agent's turn so it wants to minimize the agent's value.
            neighbours = sorted(self.expand(state), 
                            key=lambda neighbour: self._heuristic(type(self).get_neighbour_state(neighbour), self.agent_id)
            )

            min_value = np.inf
            best_action = random.choice(neighbours) if len(neighbours) else None

            for neighbour in neighbours:
                neighbour_action = type(self).get_neighbour_action(neighbour)
                neighbour_state = type(self).get_neighbour_state(neighbour)

                self.__action_to_curr_state = neighbour_action
                if self.should_abort():
                    # Was decided to abort the search
                    return self.__action_to_curr_state, 0

                _, value = self._minimax_aux(state=neighbour_state,
                                                depth_limit=depth_limit - 1,
                                                prune=prune,
                                                alpha=alpha,
                                                beta=beta)
                if value < min_value:
                    min_value = value
                    best_action = neighbour_action
                
                if prune:
                    beta = min(min_value, beta)
                    if min_value <= alpha:
                        return best_action, -np.inf
            
            return best_action, min_value


class RBMinimax(Minimax, AnytimeAlgorithm):
    best_action: Tuple[str, int]
    update_lock: th.Lock
    num_iterations: int

    def __init__(self, prune: bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune = prune
        self.best_action = None  # A tuple of the form (pawn, location)
        self.best_value = -np.inf
        self.update_lock = th.Lock()  # A lock for updating the best action.
        self.num_iterations = 0
    
    def reset(self) -> None:
        self.num_iterations = 0
        self.best_action = None
        self.reset_agent()

    def should_abort(self) -> bool:
        return self.is_stopped()

    def run(self) -> None:
        self.reset()
        curr_depth_limit = 1

        while not self.is_stopped():
            found_winning_action, best_action = self.search_for_immediate_win(self.starting_state)
            if found_winning_action:
                with self.update_lock:
                    if not self.is_stopped():
                        self.best_action = best_action
                        return

            best_action, best_value = self.minimax(state=self.starting_state, 
                                                   depth_limit=curr_depth_limit,
                                                   prune=self.prune)
            with self.update_lock:
                if self.is_stopped():
                    return self.best_action
                
                self.best_action = best_action
                self.best_value = best_value
                self.num_iterations += 1
                curr_depth_limit += 1
                print(f'The best action after {self.num_iterations} iterations is {self.best_action} with the estimated heuristical value of {self.best_value}.')
        
        return


class Expectimax(Agent):
    _heuristic: MAHeuristic
    __action_to_curr_state: Tuple[str, int]
    _is_probablistic: Callable[[gge.State], bool]
    _event_value: Callable[[gge.State, Tuple[str, int]], float]

    def __init__(self,
                 heuristic: MAHeuristic,
                 is_probablistic_fn: Callable[[gge.State], bool],
                 event_value_fn: Callable[[gge.State, Tuple[str, int]], float],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._heuristic = heuristic
        self._is_probablistic = is_probablistic_fn
        self._event_value = event_value_fn
        self.__action_to_curr_state = tuple()

    def should_abort(self) -> bool:
        return False
    
    def search_for_immediate_win(self, state: gge.State):
        neighbours = sorted(self.expand(state), 
                            key=lambda neighbour: self._heuristic(type(self).get_neighbour_state(neighbour), self.agent_id)
        )

        found_winning_action = False
        max_value = -np.inf
        best_action = None

        for neighbour in neighbours:
            neighbour_action = type(self).get_neighbour_action(neighbour)
            neighbour_state = type(self).get_neighbour_state(neighbour)

            if type(self).is_player_winner(neighbour_state, self.agent_id):
                found_winning_action = True
                value = self._heuristic(neighbour_state, self.agent_id)
                if value > max_value:
                    max_value = value
                    best_action = neighbour_action
        
        return found_winning_action, best_action
    
    def total_event_values(self, state: gge.State, neighbours: Tuple[Tuple[str, int], gge.State]) -> float:
        return np.sum([self._event_value(state, action) for action, _ in neighbours])

    def expectimax(self, state: gge.State, depth_limit: int=np.inf) -> Tuple[Tuple[str, int], float]:
        """
        Calculates the next best expected action starting from the input state subjected to a depth limit.

        Parameters:
        -----------
        state : gge.State
            The starting state to calculate the next best action from.
        
        depth_limit : int = np.inf
            The maximum depth to search in, should be initially at least 1.

        Return:
        -------
        action: (pawn, board_cell_index), value: float
            Returns the best action to take as a tuple of the form (pawn, board_cell_index), and 
            the value that action is guaranteed to achieve heuristically.
        """
        self.__action_to_curr_state = tuple()  # Reset the action take to the current state
        
        return self._expectimax_aux(state, depth_limit)
        
    
    def _expectimax_aux(self,
                        state: gge.State, 
                        depth_limit: int=np.inf) -> Tuple[Tuple[str, int], float]:
        assert depth_limit >= 0
        if self.should_abort():
            # Was decided to abort the search
            return self.__action_to_curr_state, 0
        
        if type(self).is_terminal_state(state) or depth_limit == 0:
            return self.__action_to_curr_state, self._heuristic(state, self.agent_id)

        if self._is_probablistic(state):
            # Calculate expected value
            neighbours = self.expand(state)
            return self.__action_to_curr_state, np.sum(
                np.array([
                    self._event_value(state, action) * self._expectimax_aux(child_state, depth_limit=depth_limit - 1)[1]
                    for action, child_state in neighbours
                ])
            ) / self.total_event_values(state, neighbours)
        
        self.__action_to_curr_state = tuple()  # Reset the action take to the current state
        
        if state.turn == self.agent_id:
            # This is the agent's turn so it wants to maximize it's value.
            neighbours = sorted(self.expand(state), 
                            key=lambda neighbour: self._heuristic(type(self).get_neighbour_state(neighbour), self.agent_id),
                            reverse=True
            )
            max_value = -np.inf
            best_action = random.choice(neighbours) if len(neighbours) else None
            found_winning_action = False

            for neighbour in neighbours:
                neighbour_action = type(self).get_neighbour_action(neighbour)
                neighbour_state = type(self).get_neighbour_state(neighbour)
                
                self.__action_to_curr_state = neighbour_action
                if self.should_abort():
                    # Was decided to abort the search
                    return self.__action_to_curr_state, 0

                if type(self).is_player_winner(neighbour_state, state.turn):
                    # If this player is going to win by doing this action, no need to check further.
                    # Will attempt to find the optimal winning action.
                    found_winning_action = True
                    value = self._heuristic(neighbour_state, self.agent_id)
                    if value > max_value:
                        max_value = value
                        best_action = neighbour_action
                    continue
                
                if not found_winning_action:
                    _, value = self._expectimax_aux(state=neighbour_state,
                                                    depth_limit=depth_limit - 1)
                    if value > max_value:
                        max_value = value
                        best_action = neighbour_action
            
            return best_action, max_value
        
        else:
            # This isn't the agent's turn so it wants to minimize the agent's value.
            neighbours = sorted(self.expand(state), 
                            key=lambda neighbour: self._heuristic(type(self).get_neighbour_state(neighbour), self.agent_id)
            )

            min_value = np.inf
            best_action = random.choice(neighbours) if len(neighbours) else None

            for neighbour in neighbours:
                neighbour_action = type(self).get_neighbour_action(neighbour)
                neighbour_state = type(self).get_neighbour_state(neighbour)

                self.__action_to_curr_state = neighbour_action
                if self.should_abort():
                    # Was decided to abort the search
                    return self.__action_to_curr_state, 0

                _, value = self._expectimax_aux(state=neighbour_state,
                                                depth_limit=depth_limit - 1)
                if value < min_value:
                    min_value = value
                    best_action = neighbour_action
            
            return best_action, min_value


class RBExpectimax(Expectimax, AnytimeAlgorithm):
    best_action: Tuple[str, int]
    update_lock: th.Lock
    num_iterations: int

    def __init__(self, prune: bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_action = None  # A tuple of the form (pawn, location)
        self.best_value = -np.inf
        self.update_lock = th.Lock()  # A lock for updating the best action.
        self.num_iterations = 0
    
    def reset(self) -> None:
        self.num_iterations = 0
        self.best_action = None
        self.reset_agent()

    def should_abort(self) -> bool:
        return self.is_stopped()

    def run(self) -> None:
        self.reset()
        curr_depth_limit = 1

        while not self.is_stopped():
            found_winning_action, best_action = self.search_for_immediate_win(self.starting_state)
            if found_winning_action:
                with self.update_lock:
                    if not self.is_stopped():
                        self.best_action = best_action
                        return

            best_action, best_value = self.expectimax(state=self.starting_state, 
                                                   depth_limit=curr_depth_limit)
            with self.update_lock:
                if self.is_stopped():
                    return self.best_action
                
                self.best_action = best_action
                self.best_value = best_value
                self.num_iterations += 1
                curr_depth_limit += 1
                print(f'The best action after {self.num_iterations} iterations is {self.best_action} with the estimated heuristical value of {self.best_value}.')
        
        return


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


def rb_heuristic_min_max(curr_state, agent_id, time_limit) -> Tuple[str, int]:
    global HEURISTIC_OBJ

    rb_minimax = RBMinimax(starting_state=curr_state,
                           agent_id=agent_id,
                           heuristic=HEURISTIC_OBJ)
    
    # Start the minimax search and abort it just before the timer expires
    rb_minimax.start()
    rb_minimax.join(timeout=time_limit - 0.2)
    with rb_minimax.update_lock:
        rb_minimax.stop()

    print(f'RB-Minimax has completed {rb_minimax.num_iterations} full iterations.')
    return rb_minimax.best_action


def alpha_beta(curr_state, agent_id, time_limit):
    global HEURISTIC_OBJ

    rb_minimax = RBMinimax(starting_state=curr_state,
                           agent_id=agent_id,
                           prune=True,
                           heuristic=HEURISTIC_OBJ)
    
    # Start the minimax search and abort it just before the timer expires
    rb_minimax.start()
    rb_minimax.join(timeout=time_limit - 0.2)
    with rb_minimax.update_lock:
        rb_minimax.stop()

    print(f'RB-Minimax has completed {rb_minimax.num_iterations} full iterations.')
    return rb_minimax.best_action


def expectimax(curr_state, agent_id, time_limit):
    global HEURISTIC_OBJ

    rb_expectimax = RBExpectimax(starting_state=curr_state,
                                 agent_id=agent_id,
                                 heuristic=HEURISTIC_OBJ,
                                 is_probablistic_fn=lambda state: state.turn != agent_id,
                                 event_value_fn=get_event_value)
    
    # Start the Expectimax search and abort it just before the timer expires
    rb_expectimax.start()
    rb_expectimax.join(timeout=time_limit - 0.2)
    with rb_expectimax.update_lock:
        rb_expectimax.stop()

    print(f'RB-Expectimax has completed {rb_expectimax.num_iterations} full iterations.')
    return rb_expectimax.best_action


# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    raise NotImplementedError()
