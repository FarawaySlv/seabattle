import torch
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
import os
from datetime import datetime
import random

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
        self.current_batch_index = 0
        os.makedirs(data_dir, exist_ok=True)
    
    def load_from_logs(self, min_games: int = 100) -> bool:
        """Load training data from game logs"""
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return False
            
        # Load all game logs
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.data_dir, filename), 'r') as f:
                    try:
                        game_data = json.load(f)
                        if self._validate_game_data(game_data):
                            self.games.append(game_data)
                    except json.JSONDecodeError:
                        print(f"Error reading log file: {filename}")
                        continue
        
        print(f"Loaded {len(self.games)} games from {self.data_dir}")
        return len(self.games) >= min_games
    
    def _validate_game_data(self, game_data: Dict) -> bool:
        """Validate that game data has required fields"""
        required_fields = ["timestamp", "moves"]
        if not all(field in game_data for field in required_fields):
            return False
            
        # Validate moves format
        for move in game_data["moves"]:
            if not all(field in move for field in ["x", "y", "hit"]):
                return False
                
        return True
    
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
    
    def get_validation_data(self, validation_split: float = 0.2) -> Tuple[Optional['TrainingData'], Optional['TrainingData']]:
        """Split data into training and validation sets"""
        if not self.games:
            return None, None
            
        # Shuffle games
        random.shuffle(self.games)
        
        # Calculate split index
        split_idx = int(len(self.games) * (1 - validation_split))
        
        # Create new instances for training and validation
        train_data = TrainingData(self.board_size)
        val_data = TrainingData(self.board_size)
        
        # Split the games
        train_data.games = self.games[:split_idx]
        val_data.games = self.games[split_idx:]
        
        return train_data, val_data
    
    def get_training_batch(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get a batch of training data"""
        if not self.games:
            return None
            
        # Initialize batch tensors
        states = []
        targets = []
        
        # Collect batch_size samples
        for _ in range(batch_size):
            if self.current_batch_index >= len(self.games):
                self.current_batch_index = 0
                random.shuffle(self.games)
                
            game = self.games[self.current_batch_index]
            self.current_batch_index += 1
            
            # Process each move in the game
            for i in range(len(game["moves"]) - 1):
                current_state = self._create_state_tensor(game["moves"][:i+1])
                next_move = game["moves"][i+1]
                target = self._create_target_tensor(next_move)
                
                states.append(current_state)
                targets.append(target)
                
                if len(states) >= batch_size:
                    break
            
            if len(states) >= batch_size:
                break
        
        if not states:
            return None
            
        # Convert to tensors
        states_tensor = torch.stack(states)
        targets_tensor = torch.stack(targets)
        
        return {
            "states": states_tensor,
            "targets": targets_tensor
        }
    
    def _create_state_tensor(self, moves: List[Dict]) -> torch.Tensor:
        """Create input tensor from game state"""
        state = torch.zeros((self.board_size, self.board_size), dtype=torch.long)
        
        for move in moves:
            x, y = move["x"], move["y"]
            if move["hit"]:
                state[y, x] = 1  # Hit
            else:
                state[y, x] = 2  # Miss
                
        return state
    
    def _create_target_tensor(self, move: Dict) -> torch.Tensor:
        """Create target tensor for next move"""
        target = torch.zeros((self.board_size, self.board_size), dtype=torch.float)
        x, y = move["x"], move["y"]
        target[y, x] = 1.0
        return target
    
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