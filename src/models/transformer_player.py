import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import torch
import torch.nn as nn
import json
import random
from typing import List, Tuple, Dict
import numpy as np
from models.board import Board
from models.ship import Ship
from utils.constants import SHIPS, GRID_SIZE
from transformers import DistilBertModel, DistilBertConfig

class TransformerPlayer:
    """A Battleship player that uses a trained transformer model to make decisions."""
    def __init__(self, model_path: str = "models/battleship/checkpoints", config_path: str = "models/battleship/configs/model_config.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config(config_path)
        self.model = self._create_model()
        self.model.to(self.device)
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        
        # Game state tracking
        self.shot_history = set()  # Keep track of all shots
        self.hit_history = set()   # Keep track of hits
        self.last_hit = None       # Last successful hit
        self.consecutive_hits = []  # Track consecutive hits to determine ship direction
        self.ship_direction = None  # Current ship direction (dx, dy)
        
        # Transformer-specific state
        self.board_size = self.config["input_format"]["board_size"]
        self.move_history = []
        self.miss_history = []
        
        # Load the best model if available
        self.load_model("best")

    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _create_model(self) -> nn.Module:
        """Create DistilBERT-based model"""
        class BattleshipDistilBERT(nn.Module):
            def __init__(self, full_config):
                super().__init__()
                self.board_size = full_config["input_format"]["board_size"]
                config = full_config["architecture"]
                
                # Initialize DistilBERT with custom configuration
                bert_config = DistilBertConfig(
                    vocab_size=config["vocab_size"],
                    hidden_size=config["hidden_dim"],
                    num_hidden_layers=config["num_layers"],
                    num_attention_heads=config["num_heads"],
                    hidden_dropout_prob=config["dropout"],
                    attention_probs_dropout_prob=config["dropout"],
                    max_position_embeddings=self.board_size * self.board_size + 1  # +1 for [CLS] token
                )
                
                self.bert = DistilBertModel(bert_config)
                
                # Additional layers for board prediction
                self.fc = nn.Sequential(
                    nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                    nn.ReLU(),
                    nn.Dropout(config["dropout"]),
                    nn.Linear(config["hidden_dim"], self.board_size * self.board_size)
                )
            
            def forward(self, x, attention_mask=None):
                # x shape: (batch_size, board_size, board_size)
                batch_size = x.shape[0]
                
                # Flatten the board and add [CLS] token
                x = x.view(batch_size, -1)  # (batch_size, board_size * board_size)
                cls_token = torch.zeros((batch_size, 1), dtype=torch.long, device=x.device)
                x = torch.cat([cls_token, x], dim=1)  # Add [CLS] token
                
                # Create attention mask
                if attention_mask is None:
                    attention_mask = torch.ones_like(x)
                
                # Get BERT output
                outputs = self.bert(
                    input_ids=x,
                    attention_mask=attention_mask
                )
                
                # Use [CLS] token output for prediction
                cls_output = outputs.last_hidden_state[:, 0]
                
                # Project to board size
                x = self.fc(cls_output)
                x = x.view(batch_size, self.board_size, self.board_size)
                
                return x
        
        return BattleshipDistilBERT(self.config)

    def place_ships(self, board: Board):
        """Place ships randomly on the board"""
        for size, count in SHIPS.items():
            for _ in range(count):
                while True:
                    # Try random position and orientation
                    x = random.randint(0, GRID_SIZE - 1)
                    y = random.randint(0, GRID_SIZE - 1)
                    horizontal = random.choice([True, False])
                    
                    ship = Ship(size)
                    if board.place_ship(ship, x, y, horizontal):
                        break

    def get_next_shot(self, board: Board) -> Tuple[int, int]:
        """Get next shot coordinates based on current board state"""
        # Create input tensor
        input_tensor = self._create_input_tensor()
        
        # Get model prediction
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
            
            # Reshape output to board size
            output = output.view(self.board_size, self.board_size)
            
            # Apply strategic modifications to the output probabilities
            
            # 1. Strongly penalize already shot positions
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if board.get_cell_state(x, y) in [2, 3, 4]:  # 2 for hit, 3 for miss, 4 for partially hit
                        output[y, x] = float('-inf')
            
            # 2. Add bonus for positions adjacent to hits (if any)
            for x, y in self.hit_history:
                # First, try to find ship direction from consecutive hits
                if len(self.consecutive_hits) >= 2:
                    last_hit = self.consecutive_hits[-1]
                    prev_hit = self.consecutive_hits[-2]
                    dx = last_hit[0] - prev_hit[0]
                    dy = last_hit[1] - prev_hit[1]
                    
                    # If we have a direction, prioritize continuing in that direction
                    if dx != 0 or dy != 0:
                        nx, ny = last_hit[0] + dx, last_hit[1] + dy
                        if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                            if board.get_cell_state(nx, ny) not in [2, 3, 4]:
                                output[ny, nx] += 5.0  # Higher bonus for continuing in ship direction
                
                # Also check adjacent cells in all directions
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if board.get_cell_state(nx, ny) not in [2, 3, 4]:
                            output[ny, nx] += 3.0  # Bonus for adjacent cells
            
            # 3. Add small random noise for exploration
            noise = torch.randn_like(output) * 0.005
            output = output + noise
            
            # 4. Apply temperature scaling to make probabilities more distinct
            temperature = 0.2
            output = output / temperature
            
            # 5. Find the best probability and similar moves
            flat_output = output.view(-1)
            best_value = torch.max(flat_output).item()
            
            # Print top 5 results
            top_k_values, top_k_indices = torch.topk(flat_output, min(5, flat_output.numel()))
            print("\nTop 5 predicted moves:")
            for i, (value, idx) in enumerate(zip(top_k_values, top_k_indices)):
                x = idx.item() % self.board_size
                y = idx.item() // self.board_size
                cell_state = board.get_cell_state(x, y)
                state_str = "valid" if cell_state not in [2, 3, 4] else "invalid"
                print(f"{i+1}. Position ({x}, {y}) - Probability: {value.item():.4f} - State: {state_str}")
            
            # Find all moves with similar probability (within 5% of the best)
            similar_moves = []
            threshold = best_value * 0.95  # 5% threshold
            
            for i in range(len(flat_output)):
                if flat_output[i].item() >= threshold:
                    x = i % self.board_size
                    y = i // self.board_size
                    if board.get_cell_state(x, y) not in [2, 3, 4]:  # Only add if not already shot
                        similar_moves.append((x, y))
            
            # If we have similar moves, choose randomly among them
            if similar_moves:
                x, y = random.choice(similar_moves)
                print(f"Selected from {len(similar_moves)} similar moves: ({x}, {y})")
            else:
                # If no similar moves, find all valid moves
                valid_moves = []
                for i in range(self.board_size):
                    for j in range(self.board_size):
                        if board.get_cell_state(i, j) not in [2, 3, 4]:
                            valid_moves.append((j, i))
                
                if valid_moves:
                    x, y = random.choice(valid_moves)
                    print(f"No similar moves found, selected from {len(valid_moves)} valid moves: ({x}, {y})")
                else:
                    # If no valid moves (shouldn't happen), pick any random position
                    x = random.randint(0, self.board_size - 1)
                    y = random.randint(0, self.board_size - 1)
                    print(f"No valid moves found, selected random position: ({x}, {y})")
        
        return x, y

    def record_shot(self, x: int, y: int, hit: bool):
        """Record the result of a shot"""
        self.shot_history.add((x, y))
        if hit:
            self.hit_history.add((x, y))
            self.last_hit = (x, y)
            self.consecutive_hits.append((x, y))
            # If we have two or more hits, update ship direction
            if len(self.consecutive_hits) >= 2:
                x1, y1 = self.consecutive_hits[-2]
                x2, y2 = self.consecutive_hits[-1]
                self.ship_direction = (x2 - x1, y2 - y1)
        else:
            # Reset consecutive hits and ship direction on miss
            self.consecutive_hits = []
            self.ship_direction = None
            self.miss_history.append((x, y))

    def _get_current_board_state(self) -> List[List[int]]:
        """Get current board state as a 2D list"""
        board_state = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        # Mark hits
        for x, y in self.hit_history:
            board_state[y][x] = 1
        
        # Mark misses
        for x, y in self.miss_history:
            board_state[y][x] = 2
        
        return board_state

    def _create_input_tensor(self) -> torch.Tensor:
        """Create input tensor from current game state"""
        # Create 2D board tensor
        input_tensor = torch.zeros((1, self.board_size, self.board_size), dtype=torch.long, device=self.device)
        
        # Add hit history (1)
        for x, y in self.hit_history:
            input_tensor[0, y, x] = 1
        
        # Add miss history (2)
        for x, y in self.miss_history:
            input_tensor[0, y, x] = 2
            
        # Add move history context (only if we have recent moves)
        if len(self.move_history) > 0:
            # Get the last 3 moves for context
            recent_moves = self.move_history[-3:]
            for x, y, hit in recent_moves:
                # Only mark if not already marked as hit/miss
                if input_tensor[0, y, x] == 0:
                    input_tensor[0, y, x] = 3 if hit else 4
        
        return input_tensor

    def load_model(self, name: str = "best"):
        """Load model state"""
        path = os.path.join(self.model_path, f"{name}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return True
        return False
    
    def reset(self):
        """Reset game state tracking"""
        self.shot_history = set()
        self.hit_history = set()
        self.last_hit = None
        self.consecutive_hits = []
        self.ship_direction = None
        self.move_history = []
        self.miss_history = [] 