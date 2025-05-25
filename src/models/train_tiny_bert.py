import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import numpy as np
import os
import json
from tqdm import tqdm
from models.tiny_bert_player import TinyBertPlayer
from utils.constants import GRID_SIZE

class BattleshipDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        
        # Load and process game logs
        for filename in os.listdir(data_path):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(data_path, filename), 'r') as f:
                        try:
                            game_data = json.load(f)
                            self._process_game(game_data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Corrupted JSON file {filename}, skipping")
                            continue
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
                    continue
    
    def _process_game(self, game_data):
        try:
            # Validate game data
            if not game_data.get('moves'):
                print(f"Warning: Game has no moves, skipping")
                return
                
            if not game_data.get('initial_board_player') or not game_data.get('initial_board_ai'):
                print(f"Warning: Game missing initial board states, skipping")
                return
            
            # Process each move in the game
            board_state = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
            for move in game_data['moves']:
                if not all(k in move for k in ['x', 'y', 'hit']):
                    print(f"Warning: Invalid move data, skipping move")
                    continue
                    
                x, y = move['x'], move['y']
                hit = move['hit']
                
                # Create target (one-hot encoding of the move)
                target = np.zeros(GRID_SIZE * GRID_SIZE)
                target[y * GRID_SIZE + x] = 1
                
                # Convert board to text
                board_text = self._board_to_text(board_state)
                
                # Update board state
                board_state[y][x] = 2 if hit else 3
                
                # Add to dataset
                self.data.append({
                    'text': board_text,
                    'target': target
                })
        except Exception as e:
            print(f"Error processing game: {e}")
            return
    
    def _board_to_text(self, board):
        text = []
        for y in range(GRID_SIZE):
            row = []
            for x in range(GRID_SIZE):
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'target': torch.tensor(item['target'], dtype=torch.float)
        }

def train_model():
    # Check if we have enough games
    data_path = os.path.join("models", "battleship", "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    game_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    if len(game_files) < 100:
        print("Need at least 100 games to train. Current games:", len(game_files))
        return
    
    # Initialize model and tokenizer
    model = TinyBertPlayer(use_trained_model=False)
    model.model.train()
    
    # Create dataset and dataloader
    dataset = BattleshipDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Training setup
    optimizer = AdamW(model.model.parameters(), lr=5e-5)
    num_epochs = 10
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            
            outputs = model.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            logits = model.classifier(outputs.last_hidden_state[:, 0, :])
            loss = criterion(logits, batch['target'])
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                logits = model.classifier(outputs.last_hidden_state[:, 0, :])
                loss = criterion(logits, batch['target'])
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.model.state_dict(), os.path.join("models", "battleship", "best_tiny_bert_model.pt"))
            print("Saved new best model!")

if __name__ == "__main__":
    train_model() 