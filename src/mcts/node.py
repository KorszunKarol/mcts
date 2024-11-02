from dataclasses import dataclass, field
import chess
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree representing a chess position.

    This class maintains the state and statistics of a chess position in the MCTS tree,
    including visit counts, value estimates, and relationships to other nodes.

    Attributes:
        board (chess.Board): The chess position represented by this node
        parent (Optional[MCTSNode]): The parent node in the MCTS tree
        children (List[MCTSNode]): Child nodes representing possible next positions
        visits (int): Number of times this node has been visited during search
        value (float): Accumulated value from evaluations and backpropagation
        prior (float): Prior probability assigned to this node (for future policy network integration)
        move (Optional[chess.Move]): The chess move that led to this position
        exploration_factor (float): Default exploration factor
        uncertainty (float): Store uncertainty from evaluation
    """
    board: chess.Board
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    prior: float = 0.0
    move: Optional[chess.Move] = None
    exploration_factor: float = 1.4  # Default exploration factor
    uncertainty: float = 0.0  # Store uncertainty from evaluation

    def __post_init__(self):
        """Initialize move attribute if not provided but parent exists."""
        if self.parent is not None and self.move is None:
            self.move = self.board.move_stack[-1] if self.board.move_stack else None
            logger.debug(f"Node initialized with move: {self.move}")

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded in the search tree."""
        return len(self.children) > 0

    @property
    def q_value(self) -> float:
        """
        Calculate the Q-value (mean value) of this node.

        Returns:
            float: The average value of this node from all visits
        """
        return self.value / (self.visits + 1e-8)

    def update_value(self, value: float):
        """
        Update node statistics with a new value.

        Args:
            value (float): The value to add to this node's accumulated value
        """
        self.visits += 1
        self.value += value
        logger.debug(f"Node {self.move} updated: visits={self.visits}, value={self.value:.3f}")

    def __repr__(self):
        """String representation of the node showing key statistics."""
        return f"MCTSNode(move={self.move}, visits={self.visits}, value={self.value:.3f}, q_value={self.q_value:.3f})"
