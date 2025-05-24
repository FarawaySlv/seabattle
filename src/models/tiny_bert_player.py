import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from models.base_player import BasePlayer
from utils.constants import BOARD_SIZE
import os
import json
import time

class TinyBertPlayer(BasePlayer):
    def __init__(self, use_trained_model=True):
        super().__init__()
        self.model_name = "prajjwal1/bert-tiny"  # Using TinyBERT model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Add classification head
        self.classifier = nn.Linear(self.model.config.hidden_size, BOARD_SIZE * BOARD_SIZE)
        
        # Load best model if available and requested
        if use_trained_model:
            self._load_best_model()
        else:
            self._init_weights()
        
        self.model.eval()  # Set to evaluation mode
        
    def _init_weights(self):
        """Initialize weights for untrained model"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def _load_best_model(self):
        """Load the best model weights"""
        model_path = os.path.join("models", "battleship", "best_tiny_bert_model.pt")
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                print("Loaded best TinyBERT model")
            except Exception as e:
                print(f"Error loading TinyBERT model: {e}")
                self._init_weights()
        else:
            print("No saved TinyBERT model found, using untrained model")
            self._init_weights()
            
    def _board_to_text(self, board):
        """Convert board state to text representation"""
        text = []
        for y in range(BOARD_SIZE):
            row = []
            for x in range(BOARD_SIZE):
                cell = board[y][x]
                if cell == 0:  # Empty
                    row.append(".")
                elif cell == 1:  # Ship
                    row.append("S")
                elif cell == 2:  # Hit
                    row.append("H")
                elif cell == 3:  # Miss
                    row.append("M")
            text.append(" ".join(row))
        return "\n".join(text)
        
    def get_next_shot(self, board):
        """Get the next shot based on the current board state"""
        start_time = time.time()
        try:
            # Convert board to text
            board_text = self._board_to_text(board.grid)
            
            # Tokenize input
            inputs = self.tokenizer(
                board_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get model predictions with timeout
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = self.classifier(outputs.last_hidden_state[:, 0, :])
                probabilities = torch.softmax(logits, dim=1)
                
            # Convert probabilities to numpy array
            probs = probabilities.numpy().flatten()
            
            # Get top 5 moves for analysis
            top_5_indices = np.argsort(probs)[-5:][::-1]
            print("\nTop 5 predicted moves:")
            for idx in top_5_indices:
                x, y = idx % BOARD_SIZE, idx // BOARD_SIZE
                print(f"Position ({x}, {y}): {probs[idx]:.4f}")
                
            # Find the best move (highest probability)
            best_move_idx = np.argmax(probs)
            x, y = best_move_idx % BOARD_SIZE, best_move_idx // BOARD_SIZE
            
            # Check if the move is valid (not already hit or missed)
            if board.grid[y][x] in [2, 3]:  # If position is already hit or missed
                # Find the next best move that's valid
                for idx in top_5_indices:
                    x, y = idx % BOARD_SIZE, idx // BOARD_SIZE
                    if board.grid[y][x] not in [2, 3]:
                        break
            
            # Timeout check
            if time.time() - start_time > 1.0:  # 1 second timeout
                print("TinyBERT prediction took too long, using random move")
                # Use random move as fallback
                valid_moves = [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) 
                             if board.grid[y][x] not in [2, 3]]
                if valid_moves:
                    x, y = valid_moves[np.random.randint(len(valid_moves))]
            
            return x, y
            
        except Exception as e:
            print(f"Error in TinyBERT prediction: {e}")
            # Fallback to random move
            valid_moves = [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) 
                         if board.grid[y][x] not in [2, 3]]
            if valid_moves:
                x, y = valid_moves[np.random.randint(len(valid_moves))]
                return x, y
            return 0, 0  # Last resort fallback
        
    def record_shot(self, x, y, hit):
        """Record the result of a shot"""
        # TinyBERT doesn't need to record shots as it makes decisions based on the current board state
        pass 