from dataclasses import dataclass, field
import chess
import numpy as np
from typing import List, Optional, Tuple, Dict
import logging
from src.mcts.node import MCTSNode
from src.evaluator import SingleEvaluator
import time
from src.mcts.transposition_table import TranspositionTable

logger = logging.getLogger(__name__)

@dataclass
class MCTS:
    """
    Monte Carlo Tree Search implementation optimized for neural network evaluation.

    Attributes:
        evaluator: Neural network evaluator
        simulations: Number of simulations to run
        batch_size: Number of positions to evaluate at once
        exploration_constant: Controls exploration vs exploitation
        temperature: Controls move selection randomness
        position_uncertainty_threshold: Threshold for position uncertainty
    """
    evaluator: SingleEvaluator
    simulations: int = 800
    batch_size: int = 32
    exploration_constant: float = 1.4
    temperature: float = 1.0
    position_uncertainty_threshold: float = 0.3

    def __post_init__(self):
        """Initialize statistics tracking."""
        self.stats = {
            'total_nodes': 0,
            'evaluation_time': 0.0,
            'selection_time': 0.0,
            'expansion_time': 0.0,
            'backprop_time': 0.0,
            'total_time': 0.0,
            'nodes_per_second': 0.0
        }
        logger.debug(f"MCTS initialized with simulations={self.simulations}, batch_size={self.batch_size}, "
                     f"exploration_constant={self.exploration_constant}, temperature={self.temperature}")
        self.tt = TranspositionTable()  # Add transposition table

    def search(self, root_board: chess.Board) -> Tuple[chess.Move, Dict]:
        """
        Perform MCTS search from the given position.

        Args:
            root_board: Starting chess position

        Returns:
            Tuple of (best_move, statistics)
        """
        logger.info(f"\nStarting MCTS search with {self.simulations} simulations")
        logger.info(f"Batch size: {self.batch_size}, Temperature: {self.temperature}")

        self.start_time = time.time()
        root_node = MCTSNode(root_board)

        # Initialize root children
        logger.info("Evaluating initial moves...")
        legal_moves = list(root_board.legal_moves)
        logger.info(f"Found {len(legal_moves)} legal moves")

        for move in legal_moves:
            board_copy = root_board.copy()
            board_copy.push(move)
            child = MCTSNode(board_copy, parent=root_node, move=move)
            root_node.children.append(child)
            logger.debug(f"Initialized child node for move {move}")

        # Initial evaluation of root children
        logger.info("Evaluating initial positions...")
        initial_evals = self._batch_evaluate_nodes(root_node.children)

        # Log initial evaluations
        logger.info("\nInitial move evaluations:")
        for child, (value, uncertainty) in zip(root_node.children, initial_evals):
            logger.info(f"Move {child.move}: value={value:.3f}, uncertainty={uncertainty:.3f}")
            child.update_value(value)
            logger.debug(f"Updated node {child.move} with value={value:.3f}, uncertainty={uncertainty:.3f}")

        # Fix timing measurements
        search_start = time.time()
        eval_time = 0.0
        select_time = 0.0
        backprop_time = 0.0

        # Main MCTS loop
        total_batches = (self.simulations + self.batch_size - 1) // self.batch_size
        logger.info(f"Starting main MCTS loop with {total_batches} batches")
        for batch_idx in range(total_batches):
            batch_start = time.time()
            batch_nodes = []
            batch_paths = []

            # Selection and expansion phase
            sel_start = time.time()
            current_batch_size = min(self.batch_size, self.simulations - batch_idx * self.batch_size)
            logger.debug(f"\nBatch {batch_idx + 1}/{total_batches}: Processing {current_batch_size} simulations")

            for i in range(current_batch_size):
                path = [root_node]
                node = root_node
                depth = 0

                # Selection
                while node.children and not node.board.is_game_over():
                    prev_node = node
                    node = self._select_child(node)
                    path.append(node)
                    depth += 1
                    logger.debug(f"Batch {batch_idx + 1}, Simulation {i + 1}: "
                                 f"Selected move {node.move} at depth {depth} "
                                 f"(Q={node.q_value:.3f}, Visits={node.visits})")

                # Expansion
                if not node.board.is_game_over() and not node.children:
                    legal_moves = list(node.board.legal_moves)
                    if legal_moves:
                        move = np.random.choice(legal_moves)
                        board_copy = node.board.copy()
                        board_copy.push(move)
                        child = MCTSNode(board_copy, parent=node, move=move)
                        node.children.append(child)
                        batch_nodes.append(child)
                        batch_paths.append(path + [child])
                        logger.debug(f"Batch {batch_idx + 1}, Simulation {i + 1}: "
                                     f"Expanded move {move} at depth {depth + 1}")
                    else:
                        logger.debug(f"Batch {batch_idx + 1}, Simulation {i + 1}: No legal moves to expand")

            select_time += time.time() - sel_start

            # Evaluation and backpropagation
            if batch_nodes:
                logger.debug(f"\nBatch {batch_idx + 1}: Evaluating {len(batch_nodes)} new nodes")
                eval_start = time.time()
                evaluations = self._batch_evaluate_nodes(batch_nodes)
                eval_duration = time.time() - eval_start
                eval_time += eval_duration
                logger.debug(f"Batch {batch_idx + 1}: Evaluation took {eval_duration:.3f}s")

                back_start = time.time()
                for node, path, (value, uncertainty) in zip(batch_nodes, batch_paths, evaluations):
                    self._backpropagate(path, value)
                    logger.debug(f"Batch {batch_idx + 1}: Backpropagated value={value:.3f} to path length={len(path)}")
                backprop_time += time.time() - back_start

            # Log detailed statistics about the current state
            elapsed = time.time() - search_start
            nodes_per_sec = (self.stats['total_nodes'] / elapsed) if elapsed > 0 else 0
            logger.info(f"\nBatch {batch_idx + 1}/{total_batches} completed:")
            logger.info(f"Time spent: eval={eval_time:.2f}s, select={select_time:.2f}s, backprop={backprop_time:.2f}s")
            logger.info(f"Total nodes evaluated: {self.stats['total_nodes']}")
            logger.info(f"Nodes/second: {nodes_per_sec:.1f}")

            # Log top moves with detailed statistics
            top_moves = sorted(root_node.children,
                               key=lambda c: (c.visits, c.q_value),
                               reverse=True)[:5]
            logger.info("\nCurrent top moves:")
            for child in top_moves:
                ucb_score = self._calculate_ucb_score(child)
                logger.info(f"Move {child.move}: visits={child.visits}, Q={child.q_value:.3f}, "
                            f"raw_value={child.value:.3f}, UCB score={ucb_score:.3f}")
                logger.debug(f"Move {child.move} details: visits={child.visits}, Q={child.q_value}, "
                             f"raw_value={child.value}, UCB={ucb_score:.3f}")

        # Update final statistics
        total_time = time.time() - search_start
        self.stats.update({
            'total_time': total_time,
            'evaluation_time': eval_time,
            'selection_time': select_time,
            'backprop_time': backprop_time,
            'nodes_per_second': self.stats['total_nodes'] / total_time if total_time > 0 else 0.0
        })

        # Select best move based on visits and Q-value
        best_child = max(root_node.children, key=lambda c: (c.visits, c.q_value))

        logger.info("\nSearch complete!")
        logger.info(f"Total nodes evaluated: {self.stats['total_nodes']}")
        logger.info(f"Time breakdown: eval={self.stats['evaluation_time']:.2f}s ({self.stats['evaluation_time']/total_time*100:.1f}%), "
                    f"select={self.stats['selection_time']:.2f}s ({self.stats['selection_time']/total_time*100:.1f}%), "
                    f"backprop={self.stats['backprop_time']:.2f}s ({self.stats['backprop_time']/total_time*100:.1f}%)")
        logger.info(f"Nodes per second: {self.stats['nodes_per_second']:.1f}")

        logger.info("\nFinal move statistics:")
        for child in sorted(root_node.children, key=lambda x: x.visits, reverse=True)[:5]:
            logger.info(f"Move {child.move}: visits={child.visits}, Q={child.q_value:.3f}, raw_value={child.value:.3f}")

        logger.info(f"\nSelected move: {best_child.move}")
        logger.info(f"Visits: {best_child.visits}")
        logger.info(f"Q-value: {best_child.q_value:.3f}")
        logger.info(f"Raw value: {best_child.value:.3f}")

        # Modified move selection at the end
        candidates = sorted(root_node.children,
                          key=lambda c: (c.visits, c.q_value),
                          reverse=True)[:3]

        # Verify top candidates
        for candidate in candidates:
            eval_change = abs(candidate.q_value - root_node.q_value)
            has_captures = any(candidate.board.is_capture(move)
                             for move in candidate.board.legal_moves)

            if eval_change > 0.5 and not has_captures:
                logger.info(f"Suspicious evaluation change for move {candidate.move}: "
                           f"Δ={eval_change:.2f}, no captures")
                continue

            return candidate.move

        # If all candidates are suspicious, return most visited
        logger.warning("All top moves showed suspicious evaluation changes")
        return candidates[0].move

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child using PUCT formula with transposition table lookup."""
        # Check transposition table first
        tt_entry = self.tt.lookup(node.board)
        if tt_entry and tt_entry.best_move:
            # Try to use cached best move
            for child in node.children:
                if child.move == tt_entry.best_move:
                    logger.debug(f"Using cached best move {tt_entry.best_move}")
                    return child

        # Regular selection if no cache hit
        total_visits = sum(child.visits for child in node.children)

        ucb_scores = []
        for child in node.children:
            # Basic PUCT formula
            q_value = child.q_value
            exploration = self.exploration_constant * \
                np.sqrt(np.log(total_visits + 1) / (child.visits + 1))

            # Adjust score based on uncertainty
            uncertainty_penalty = -0.1 * getattr(child, 'uncertainty', 0)
            ucb_score = q_value + exploration + uncertainty_penalty

            ucb_scores.append(ucb_score)

        best_idx = np.argmax(ucb_scores)
        selected_child = node.children[best_idx]

        return selected_child

    def _normalize_value(self, value: float) -> float:
        """
        Normalize evaluation values from centipawns (±2000) to [-1, 1] range.
        """
        return np.tanh(value / 1000.0)  # Smooth scaling with tanh

    def _batch_evaluate_nodes(self, nodes: List[MCTSNode]) -> List[Tuple[float, float]]:
        """Evaluate nodes using neural network with transposition table."""
        to_evaluate = []
        cached_results = []

        # Check cache first
        for node in nodes:
            tt_entry = self.tt.lookup(node.board)
            if tt_entry:
                cached_results.append((tt_entry.value, tt_entry.uncertainty))
            else:
                to_evaluate.append(node)

        # Evaluate only nodes not in cache
        if to_evaluate:
            # Get the boards for evaluation
            boards = [node.board for node in to_evaluate]

            # Use evaluator to get values
            evaluations = self.evaluator.evaluate(boards)

            # Store new evaluations in cache
            for node, (value, uncertainty) in zip(to_evaluate, evaluations):
                self.tt.store(
                    board=node.board,
                    value=value,
                    visits=node.visits,
                    depth=len(node.board.move_stack),
                    uncertainty=uncertainty
                )

            # Combine cached and new evaluations
            results = []
            eval_idx = 0
            for node in nodes:
                if node in to_evaluate:
                    results.append(evaluations[eval_idx])
                    eval_idx += 1
                else:
                    results.append(cached_results[len(results)])

            return results

        return cached_results

    def _backpropagate(self, path: List[MCTSNode], value: float):
        """
        Backpropagate the evaluation up the tree.
        Value starts from the perspective of the last node in the path.
        """
        current_value = value
        logger.debug("\nStarting backpropagation")
        for node in reversed(path):
            prev_q = node.q_value
            prev_visits = node.visits
            node.update_value(current_value)
            logger.debug(f"Backpropagating to move {node.move}:")
            logger.debug(f"    Previous Q: {prev_q:.3f}, Previous Visits: {prev_visits}")
            logger.debug(f"    New Q: {node.q_value:.3f}, New Visits: {node.visits}")
            logger.debug(f"    Propagating value: {current_value:.3f}")
            # Flip value for opponent's perspective
            current_value = -current_value

    def _calculate_ucb_score(self, node: MCTSNode) -> float:
        """Calculate UCB score for debugging purposes."""
        if not node.parent:
            return 0.0
        total_visits = sum(c.visits for c in node.parent.children)
        exploration = self.exploration_constant * \
            np.sqrt(np.log(total_visits) / (node.visits + 1))
        return node.q_value + exploration