from dataclasses import dataclass
import chess
import numpy as np
import tensorflow as tf
from typing import List, Union, Tuple
from src.encoder import Encoder
import logging

logger = logging.getLogger(__name__)

def create_model():
    inputs = tf.keras.Input(shape=(8, 8, 35))

    # Initial convolution block
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Residual blocks
    for i in range(2):
        residual = x
        filters = 128 * (2 ** min(i, 3))

        for _ in range(2):
            x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        if i > 0:
            residual = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(residual)

        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        # Squeeze-and-Excitation block
        se = tf.keras.layers.GlobalAveragePooling2D()(x)
        se = tf.keras.layers.Dense(filters // 4, activation="relu")(se)
        se = tf.keras.layers.Dense(filters, activation="sigmoid")(se)
        x = tf.keras.layers.Multiply()([x, tf.keras.layers.Reshape((1, 1, filters))(se)])

        x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Add Monte Carlo Dropout for uncertainty estimation
    x = tf.keras.layers.Dropout(0.2)(x, training=True)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

@dataclass
class SingleEvaluator:
    """
    Neural network evaluator with uncertainty estimation using Monte Carlo Dropout.
    """
    weights_path: str
    num_uncertainty_samples: int = 10  # Number of forward passes for uncertainty estimation

    def __post_init__(self):
        logger.debug(f"Loading model weights from {self.weights_path}")
        self.model = create_model()
        if self.weights_path:
            self.model.load_weights(self.weights_path)
            logger.debug("Model weights loaded successfully")
        self.encoder = Encoder()

    def evaluate(self, boards: Union[chess.Board, List[chess.Board]]) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """
        Evaluates position(s) with uncertainty estimation.
        """
        # Handle single board case
        if isinstance(boards, chess.Board):
            logger.debug(f"Evaluating single board: {boards.fen()}")
            position_tensor = self.encoder.encode(boards)
            value, uncertainty = self._evaluate_single_position(position_tensor)

            logger.debug(f"Single evaluation result: value={value:.3f}, uncertainty={uncertainty:.3f}")
            return value, uncertainty

        # Handle list of boards
        logger.debug(f"Evaluating batch of {len(boards)} boards")
        position_tensors = [self.encoder.encode(board) for board in boards]
        batch_tensor = np.array(position_tensors)
        results = self._evaluate_batch(batch_tensor)

        # Flip values for positions where it's black's turn
        flipped_results = []
        for board, (value, uncertainty) in zip(boards, results):
            if not board.turn:
                flipped_results.append((-value, uncertainty))
                logger.debug(f"Flipped evaluation for black: value={-value:.3f}, uncertainty={uncertainty:.3f}")
            else:
                flipped_results.append((value, uncertainty))
                logger.debug(f"White evaluation: value={value:.3f}, uncertainty={uncertainty:.3f}")

        return flipped_results

    def _evaluate_single_position(self, position_tensor: np.ndarray) -> Tuple[float, float]:
        """Evaluate single position with uncertainty using multiple forward passes."""
        predictions = []
        tensor = np.array([position_tensor])
        pred = float(self.model.predict(tensor, verbose=1)[0][0])
        # Debug logging

        return pred, 0.0
    def _evaluate_batch(self, batch_tensor: np.ndarray) -> List[Tuple[float, float]]:
        """Evaluate batch of positions with uncertainty."""
        all_predictions = []

        # Multiple forward passes with dropout
        for _ in range(self.num_uncertainty_samples):
            preds = self.model.predict(batch_tensor, verbose=0)
            # Scale each prediction in the batch
            scaled_preds = [self._scale_prediction(float(p[0])) for p in preds]
            all_predictions.append(scaled_preds)

        # Calculate mean and uncertainty for each position
        results = []
        all_predictions = np.array(all_predictions)

        for i in range(len(batch_tensor)):
            position_preds = all_predictions[:, i]
            mean_value = np.mean(position_preds)
            uncertainty = np.std(position_preds)
            results.append((mean_value, uncertainty))

        return results

    def _scale_prediction(self, raw_value: float) -> float:
        """
        Scale raw model output (centipawns) to a more useful range.
        Raw values are typically between -2000 and 2000 centipawns.
        Returns values roughly between -1 and 1.
        """
        # Simple tanh scaling to preserve more resolution
        return np.tanh(raw_value / 400.0)  # 400 centipawns ≈ 0.76 after tanh

    def save_model(self, path: str):
        """Save the full model architecture and weights."""
        self.model.save(path)

    @classmethod
    def load_full_model(cls, model_path: str, num_uncertainty_samples: int = 10):
        """Load a full model (architecture + weights)."""
        evaluator = cls(weights_path="", num_uncertainty_samples=num_uncertainty_samples)
        evaluator.model = tf.keras.models.load_model(model_path)
        evaluator.encoder = Encoder()
        return evaluator
