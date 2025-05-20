import torch
from typing import List, Dict, Tuple
import numpy as np
import json
import os
from datetime import datetime

class TrainingData:
    def __init__(self, board_size: int = 10, data_dir: str = "models/battleship/data"):
        self.board_size = board_size
        self.data_dir = data_dir
        self.games = []
        self.current_game = {
            "board_states": [],
            "moves": [],
            "results": [],
            "ship_positions": []
        }
        os.makedirs(data_dir, exist_ok=True)
    
    def load_from_logs(self, min_games: int = 100) -> bool:
        """Load training data from game logs"""
        if not os.path.exists(self.data_dir):
            return False
            
        # Get all log files
        log_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        if len(log_files) < min_games:
            return False
            
        # Load each game
        for log_file in log_files:
            with open(os.path.join(self.data_dir, log_file), 'r') as f:
                game_data = json.load(f)
                
                # Convert game data to training format
                game = {
                    "board_states": [],
                    "moves": [],
                    "results": game_data["winner"],
                    "ship_positions": []
                }
                
                # Add initial board state
                if game_data["initial_board_player"]:
                    game["board_states"].append(game_data["initial_board_player"])
                
                # Add moves
                current_board = [row[:] for row in game_data["initial_board_player"]] if game_data["initial_board_player"] else [[0] * self.board_size for _ in range(self.board_size)]
                
                for move in game_data["moves"]:
                    # Update board state
                    x, y = move["x"], move["y"]
                    current_board[y][x] = 2 if move["hit"] else 3  # 2 for hit, 3 for miss
                    game["board_states"].append([row[:] for row in current_board])
                    game["moves"].append((x, y, move["hit"]))
                
                self.games.append(game)
        
        return len(self.games) >= min_games

    def add_move(self, board_state: List[List[int]], move: Tuple[int, int], hit: bool):
        """Add a move to the current game"""
        self.current_game["board_states"].append(board_state)
        self.current_game["moves"].append((move[0], move[1], hit))
    
    def add_ship_positions(self, positions: List[List[Tuple[int, int]]]):
        """Add ship positions for the current game"""
        self.current_game["ship_positions"] = positions
    
    def end_game(self, winner: str):
        """End the current game and store it"""
        self.current_game["results"] = winner
        self.games.append(self.current_game)
        self.current_game = {
            "board_states": [],
            "moves": [],
            "results": [],
            "ship_positions": []
        }
    
    def get_training_batch(self, batch_size: int) -> Dict:
        """Get a batch of training data"""
        if not self.games:
            return None
            
        # Randomly select games for the batch
        selected_games = np.random.choice(self.games, min(batch_size, len(self.games)), replace=False)
        
        # Prepare batch data
        batch_states = []
        batch_moves = []
        batch_targets = []
        
        for game in selected_games:
            for i in range(len(game["moves"]) - 1):  # -1 because we need next move as target
                # Current state
                state = torch.tensor(game["board_states"][i], dtype=torch.long)
                batch_states.append(state)
                
                # Current move
                move = game["moves"][i]
                batch_moves.append(move)
                
                # Target (next move)
                next_move = game["moves"][i + 1]
                target = torch.zeros((self.board_size, self.board_size))
                target[next_move[1], next_move[0]] = 1
                batch_targets.append(target)
        
        return {
            "states": torch.stack(batch_states),
            "moves": torch.tensor(batch_moves),
            "targets": torch.stack(batch_targets)
        }
    
    def get_validation_data(self, validation_split: float = 0.2) -> Tuple[Dict, Dict]:
        """Split data into training and validation sets"""
        if not self.games:
            return None, None
            
        # Shuffle games
        np.random.shuffle(self.games)
        
        # Split into train and validation
        split_idx = int(len(self.games) * (1 - validation_split))
        train_games = self.games[:split_idx]
        val_games = self.games[split_idx:]
        
        # Create training data object for validation set
        val_data = TrainingData(self.board_size)
        val_data.games = val_games
        
        return self, val_data
    
    def clear(self):
        """Clear all stored games"""
        self.games = []
        self.current_game = {
            "board_states": [],
            "moves": [],
            "results": [],
            "ship_positions": []
        }

    def save_to_file(self, filename: str = None):
        """Save collected training data to file"""
        if not filename:
            filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "board_size": self.board_size,
            "games": self.games
        }
        
        with open(os.path.join(self.data_dir, filename), 'w') as f:
            json.dump(data, f)

    def load_from_file(self, filename: str) -> bool:
        """Load training data from file"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return False
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.board_size = data["board_size"]
            self.games = data["games"]
        return True 