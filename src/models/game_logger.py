import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class GameLogger:
    def __init__(self, data_dir: str = "models/battleship/data"):
        """Initialize the game logger with a directory for saving game data."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.current_game = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "moves": [],
            "initial_board_player": None,
            "initial_board_ai": None,
            "winner": None,
            "total_moves": 0,
            "hits": 0,
            "misses": 0,
            "player_type": None,  # Will be set explicitly when game starts
            "player_ai_type": None,  # Will be set for AI players
            "enemy_ai_type": None,  # Will be set for enemy AI
            "player_board": None,
            "enemy_board": None,
            "start_time": None,
            "end_time": None
        }
    
    def log_initial_board(self, board: List[List[int]], is_player: bool = True):
        """Log the initial board state."""
        if is_player:
            self.current_game["initial_board_player"] = [row[:] for row in board]
        else:
            self.current_game["initial_board_ai"] = [row[:] for row in board]
    
    def log_move(self, x: int, y: int, hit: bool, player_type: str):
        """Log a single move with its result."""
        move_data = {
            "x": x,
            "y": y,
            "hit": hit,
            "player_type": player_type,
            "timestamp": datetime.now().isoformat()
        }
        self.current_game["moves"].append(move_data)
        self.current_game["total_moves"] += 1
        if hit:
            self.current_game["hits"] += 1
        else:
            self.current_game["misses"] += 1
    
    def log_game_end(self, winner: str):
        """Log the end of the game with the winner."""
        self.current_game["winner"] = winner
        self.current_game["end_time"] = datetime.now().isoformat()
        self._save_game()
    
    def set_player_type(self, player_type: str, ai_type: str = None):
        """Set the type of player (human, transformer, or algorithmic) and AI type if applicable."""
        if player_type not in ["human", "transformer", "algorithmic"]:
            raise ValueError("Player type must be either 'human', 'transformer', or 'algorithmic'")
        
        # For human players, no AI type needed
        if player_type == "human":
            self.current_game["player_type"] = "human"
            self.current_game["player_ai_type"] = None
            return
            
        # For AI players, validate and set the AI type
        if player_type in ["transformer", "algorithmic"]:
            self.current_game["player_type"] = "ai"
            self.current_game["player_ai_type"] = player_type
        elif ai_type:
            if ai_type not in ["transformer", "algorithmic"]:
                raise ValueError("AI type must be either 'transformer' or 'algorithmic'")
            self.current_game["player_type"] = "ai"
            self.current_game["player_ai_type"] = ai_type
    
    def set_enemy_ai_type(self, ai_type: str):
        """Set the type of enemy AI (only transformer or algorithmic allowed)."""
        if ai_type not in ["transformer", "algorithmic"]:
            raise ValueError("Enemy AI type must be either 'transformer' or 'algorithmic'")
        self.current_game["enemy_ai_type"] = ai_type
    
    def _save_game(self):
        """Save the current game data to a JSON file."""
        if self.current_game["player_type"] is None:
            raise ValueError("Player type must be set before saving game")
            
        try:
            # Validate game data before saving
            if not self.current_game["moves"]:
                print("Warning: No moves recorded in game, skipping save")
                return
                
            if not self.current_game["initial_board_player"] or not self.current_game["initial_board_ai"]:
                print("Warning: Missing initial board states, skipping save")
                return
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            # Create backup directory if it doesn't exist
            backup_dir = os.path.join(self.data_dir, "old")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Move any existing file with same name to backup
            if os.path.exists(filepath):
                backup_path = os.path.join(backup_dir, filename)
                os.rename(filepath, backup_path)
            
            # Save the game data
            with open(filepath, 'w') as f:
                json.dump(self.current_game, f, indent=2, cls=NumpyEncoder)
                
            # Reset the current game
            self.current_game = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "moves": [],
                "initial_board_player": None,
                "initial_board_ai": None,
                "winner": None,
                "total_moves": 0,
                "hits": 0,
                "misses": 0,
                "player_type": None,
                "player_ai_type": None,
                "enemy_ai_type": None,
                "player_board": None,
                "enemy_board": None,
                "start_time": None,
                "end_time": None
            }
        except Exception as e:
            print(f"Error saving game log: {e}")
            # Reset current game even if save fails
            self.current_game = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "moves": [],
                "initial_board_player": None,
                "initial_board_ai": None,
                "winner": None,
                "total_moves": 0,
                "hits": 0,
                "misses": 0,
                "player_type": None,
                "player_ai_type": None,
                "enemy_ai_type": None,
                "player_board": None,
                "enemy_board": None,
                "start_time": None,
                "end_time": None
            }
    
    def get_game_stats(self) -> Dict:
        """Get statistics about the current game."""
        return {
            "total_moves": self.current_game["total_moves"],
            "hits": self.current_game["hits"],
            "misses": self.current_game["misses"],
            "hit_rate": self.current_game["hits"] / max(1, self.current_game["total_moves"])
        } 