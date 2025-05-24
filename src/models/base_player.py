from abc import ABC, abstractmethod
from models.board import Board

class BasePlayer(ABC):
    @abstractmethod
    def get_next_shot(self, board: Board) -> tuple[int, int]:
        """Get the next shot coordinates (x, y)"""
        pass
        
    @abstractmethod
    def record_shot(self, x: int, y: int, hit: bool) -> None:
        """Record the result of a shot"""
        pass
        
    def place_ships(self, board: Board) -> None:
        """Place ships on the board (default implementation)"""
        from models.ai_player import AIPlayer
        ai = AIPlayer()
        ai.place_ships(board) 