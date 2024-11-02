import chess
import os
import tensorflow as tf
from src.evaluator import SingleEvaluator
from src.mcts.mcts import MCTS
import logging
import time
from typing import Tuple

def configure_logging():
    """
    Configure logging to output to both file and console with timestamps.
    Logs are saved to chess_engine_detailed.log and displayed in the console.
    """
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture all detailed logs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("chess_engine_detailed.log"),
            logging.StreamHandler()
        ]
    )

def configure_gpu():
    """
    Configure GPU memory growth to prevent TensorFlow from allocating all memory.
    """
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except Exception as e:
        logging.error(f"GPU configuration failed: {str(e)}")

def find_best_move(fen: str, simulations: int = 800, batch_size: int = 32) -> str:
    """
    Find the best move for a given chess position using MCTS with neural network evaluation.

    Args:
        fen (str): FEN string representing the chess position
        simulations (int): Number of MCTS simulations to perform
        batch_size (int): Number of positions to evaluate simultaneously

    Returns:
        str: UCI format string representing the best move found

    Raises:
        FileNotFoundError: If the neural network weights file is not found
        Exception: If evaluator initialization fails
    """
    logger = logging.getLogger('ChessEngine')
    start_time = time.time()

    weights_path = 'src/model_2.0.h5'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    try:
        evaluator = SingleEvaluator(
            weights_path=weights_path,
            num_uncertainty_samples=10
        )
        logger.info("Evaluator initialized successfully")
    except Exception as e:
        raise Exception(f"Failed to initialize evaluator: {str(e)}")

    mcts = MCTS(
        evaluator=evaluator,
        simulations=simulations,
        exploration_constant=1.4,
        temperature=0.1,
        batch_size=batch_size
    )

    logger.info(f"Analyzing position: {fen}")
    logger.info(f"Running batched MCTS with {simulations} simulations...")

    board = chess.Board(fen)
    logger.info(f"Current position:\n{board}")

    best_move, stats = mcts.search(board)

    logger.info("\nSearch Statistics:")
    logger.info(f"Total time: {stats['total_time']:.2f} seconds")
    logger.info(f"Nodes evaluated: {stats['total_nodes']}")
    logger.info(f"Nodes per second: {stats['nodes_per_second']:.1f}")
    logger.info(f"Evaluation time: {stats['evaluation_time']:.2f}s ({stats['evaluation_time']/stats['total_time']*100:.1f}%)")
    logger.info(f"Selection time: {stats['selection_time']:.2f}s ({stats['selection_time']/stats['total_time']*100:.1f}%)")
    logger.info(f"Backpropagation time: {stats['backprop_time']:.2f}s ({stats['backprop_time']/stats['total_time']*100:.1f}%)")

    return best_move.uci()

def test_engine(fen = "r2qkb1r/pppb1pp1/2np1n1B/1B2p2p/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 0 7"):
    """
    Test the chess engine on a set of predetermined positions.
    Includes starting position, middle game, and endgame positions.
    """
    configure_logging()
    configure_gpu()
    move = find_best_move(fen, simulations=5000, batch_size=64)
    logging.info(f"Best move found: {move}")

def evaluate_position(fen: str) -> Tuple[float, float]:
    """
    Directly evaluate a chess position using the neural network.

    Args:
        fen: FEN string of the position to evaluate

    Returns:
        Tuple[float, float]: (evaluation, uncertainty)
        Evaluation is from white's perspective:
        - Positive values indicate white is better
        - Negative values indicate black is better
        Range is typically ±2000 centipawns
    """
    evaluator = SingleEvaluator(
        weights_path='src/model_2.0.h5',
        num_uncertainty_samples=1
    )

    board = chess.Board(fen)
    evaluation, uncertainty = evaluator.evaluate(board)

    print(f"\nPosition evaluation:")
    print(f"FEN: {fen}")
    print(f"Board:\n{board}")
    print(f"Evaluation: {evaluation:.1f} cp (±{uncertainty:.1f})")

    return evaluation, uncertainty

if __name__ == "__main__":
    evaluate_position("r2qkb1r/pppb1pp1/2np1n1B/1B2p2p/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 0 7")