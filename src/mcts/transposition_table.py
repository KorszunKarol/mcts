from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import chess
import numpy as np
import logging
import zlib  # We'll use this for hashing instead of polyglot

logger = logging.getLogger(__name__)

@dataclass
class TranspositionEntry:
    """
    Entry in the transposition table.

    Attributes:
        value: Evaluation value
        visits: Number of visits to this position
        depth: Depth at which this position was evaluated
        best_move: Best move found from this position
        uncertainty: Uncertainty of the evaluation
    """
    value: float
    visits: int
    depth: int
    best_move: Optional[chess.Move]
    uncertainty: float

    def should_replace(self, new_depth: int) -> bool:
        """Determine if this entry should be replaced by a new one."""
        return new_depth >= self.depth

class TranspositionTable:
    """
    Transposition table for caching chess position evaluations.

    Uses a custom hashing function for position keys and implements a replacement strategy
    based on depth and visit counts.
    """
    def __init__(self, max_size: int = 1_000_000):
        self.max_size = max_size
        self.table: Dict[int, TranspositionEntry] = {}
        self.hits = 0
        self.misses = 0
        logger.info(f"Initialized transposition table with max size {max_size}")

    def store(self, board: chess.Board, value: float, visits: int,
             depth: int, best_move: Optional[chess.Move] = None,
             uncertainty: float = 0.0) -> None:
        """
        Store a position evaluation in the table.

        Args:
            board: Chess position
            value: Evaluation value
            visits: Number of visits
            depth: Search depth
            best_move: Best move found (optional)
            uncertainty: Evaluation uncertainty
        """
        key = self._get_key(board)

        # Check if we need to replace existing entry
        if key in self.table:
            existing = self.table[key]
            if not existing.should_replace(depth):
                return

        # Manage table size
        if len(self.table) >= self.max_size:
            self._cleanup()

        self.table[key] = TranspositionEntry(
            value=value,
            visits=visits,
            depth=depth,
            best_move=best_move,
            uncertainty=uncertainty
        )

    def lookup(self, board: chess.Board) -> Optional[TranspositionEntry]:
        """
        Look up a position in the table.

        Args:
            board: Chess position to look up

        Returns:
            TranspositionEntry if found, None otherwise
        """
        key = self._get_key(board)
        entry = self.table.get(key)

        if entry is not None:
            self.hits += 1
            logger.debug(f"Cache hit: value={entry.value:.3f}, "
                        f"visits={entry.visits}, depth={entry.depth}")
        else:
            self.misses += 1
            logger.debug("Cache miss")

        return entry

    def _get_key(self, board: chess.Board) -> int:
        """Generate a unique key for a chess position."""
        # Create a string representation of the important position features
        key_parts = [
            board.fen(),  # Basic position
            str(board.turn),  # Who's turn
            "".join(str(int(x)) for x in [  # Castling rights
                board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK)
            ]),
            str(board.ep_square if board.ep_square else "None")  # En passant
        ]

        # Join all parts and create a hash
        key_str = "_".join(key_parts)
        return zlib.crc32(key_str.encode())

    def _cleanup(self) -> None:
        """Remove least valuable entries when table is full."""
        # Remove 10% of entries with lowest visits
        entries = sorted(
            self.table.items(),
            key=lambda x: x[1].visits
        )
        to_remove = len(self.table) // 10

        for key, _ in entries[:to_remove]:
            del self.table[key]

        logger.debug(f"Cleaned up {to_remove} entries from transposition table")

    def get_stats(self) -> Dict:
        """Get statistics about table usage."""
        total_lookups = self.hits + self.misses
        hit_rate = self.hits / total_lookups if total_lookups > 0 else 0

        return {
            'size': len(self.table),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }