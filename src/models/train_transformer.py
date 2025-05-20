import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, get_linear_schedule_with_warmup
import json
from src.models.training_data import TrainingData

class BattleshipDistilBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.board_size = config["input_format"]["board_size"]
        arch_config = config["architecture"]
        
        # Initialize DistilBERT with custom configuration
        bert_config = DistilBertConfig(
            vocab_size=arch_config["vocab_size"],
            hidden_size=arch_config["hidden_dim"],
            num_hidden_layers=arch_config["num_layers"],
            num_attention_heads=arch_config["num_heads"],
            hidden_dropout_prob=arch_config["dropout"],
            attention_probs_dropout_prob=arch_config["dropout"],
            max_position_embeddings=self.board_size * self.board_size + 1  # +1 for [CLS] token
        )
        
        self.bert = DistilBertModel(bert_config)
        
        # Additional layers for board prediction
        self.fc = nn.Sequential(
            nn.Linear(arch_config["hidden_dim"], arch_config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(arch_config["dropout"]),
            nn.Linear(arch_config["hidden_dim"], self.board_size * self.board_size)
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

def train_model(
    model_path: str = "models/battleship/checkpoints",
    config_path: str = "models/battleship/configs/model_config.json",
    min_games: int = 100
):
    """Train the DistilBERT model using collected game data"""
    print("\n=== Starting Training Process ===")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BattleshipDistilBERT(config)
    model.to(device)
    
    # Initialize training data
    training_data = TrainingData(board_size=config["input_format"]["board_size"])
    
    # Load and prepare training data
    print("Loading game logs...")
    success = training_data.load_from_logs(min_games)
    if not success:
        print(f"Failed to load minimum required games ({min_games})")
        return False
        
    print(f"Successfully loaded {len(training_data.games)} games")
    
    # Get training and validation data
    print("Splitting data into training and validation sets...")
    train_data, val_data = training_data.get_validation_data(
        config["training"]["validation_split"]
    )
    
    if not train_data or not val_data:
        print("Failed to split data into training and validation sets!")
        return False
        
    print(f"Training set size: {len(train_data.games)} games")
    print(f"Validation set size: {len(val_data.games)} games")
    
    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=0.01
    )
    
    # Calculate total training steps for scheduler
    batch_size = min(config["training"]["batch_size"], 32)
    total_train_batches = len(train_data.games) // batch_size
    total_steps = total_train_batches * config["training"]["max_epochs"]
    
    # Initialize learning rate scheduler with cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Training loop
    print("\n=== Training Loop ===")
    best_val_loss = float('inf')
    patience_counter = 0
    no_improvement_epochs = 0
    min_delta = 0.0001  # Minimum change in loss to be considered an improvement
    
    print(f"Total training batches per epoch: {total_train_batches}")
    print(f"Total validation batches per epoch: {len(val_data.games) // batch_size}")
    print(f"Using batch size: {batch_size}")
    
    try:
        for epoch in range(config["training"]["max_epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['training']['max_epochs']}")
            
            # Training phase
            model.train()
            total_train_loss = 0
            batches = 0
            
            print("Training phase...")
            while batches < total_train_batches:
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                batch = train_data.get_training_batch(batch_size)
                if not batch:
                    break
                
                # Forward pass
                states = batch["states"].to(device)
                targets = batch["targets"].to(device)
                
                optimizer.zero_grad()
                outputs = model(states)
                loss = nn.MSELoss()(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                batches += 1
                
                if batches % 5 == 0:
                    avg_loss = total_train_loss / batches
                    progress = (batches / total_train_batches) * 100
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Training Progress: {progress:.1f}% - Batch {batches}/{total_train_batches}")
                    print(f"Current Loss: {loss.item():.4f}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
                    
                    # Save checkpoint every 25 batches
                    if batches % 25 == 0:
                        checkpoint_path = os.path.join(model_path, f"checkpoint_epoch_{epoch + 1}_batch_{batches}.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                        }, checkpoint_path)
            
            avg_train_loss = total_train_loss / batches if batches > 0 else float('inf')
            print(f"Training completed - Average Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            print("\nValidation phase...")
            model.eval()
            total_val_loss = 0
            batches = 0
            
            with torch.no_grad():
                while batches < len(val_data.games) // batch_size:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    batch = val_data.get_training_batch(batch_size)
                    if not batch:
                        break
                    
                    # Forward pass
                    states = batch["states"].to(device)
                    targets = batch["targets"].to(device)
                    
                    outputs = model(states)
                    loss = nn.MSELoss()(outputs, targets)
                    total_val_loss += loss.item()
                    batches += 1
                    
                    if batches % 5 == 0:
                        avg_loss = total_val_loss / batches
                        progress = (batches / (len(val_data.games) // batch_size)) * 100
                        print(f"Validation Progress: {progress:.1f}% - Batch {batches}/{len(val_data.games) // batch_size}")
                        print(f"Current Loss: {loss.item():.4f}, Average Loss: {avg_loss:.4f}")
            
            avg_val_loss = total_val_loss / batches if batches > 0 else float('inf')
            print(f"Validation completed - Average Loss: {avg_val_loss:.4f}")
            
            # Save epoch checkpoint
            checkpoint_path = os.path.join(model_path, f"epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            
            # Early stopping check with minimum delta
            if avg_val_loss < (best_val_loss - min_delta):
                best_val_loss = avg_val_loss
                patience_counter = 0
                no_improvement_epochs = 0
                print("New best model found! Saving...")
                best_model_path = os.path.join(model_path, "best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, best_model_path)
            else:
                patience_counter += 1
                no_improvement_epochs += 1
                print(f"No improvement. Patience: {patience_counter}/{config['training']['early_stopping_patience']}")
                
                # Early stopping if no improvement for too long
                if patience_counter >= config["training"]["early_stopping_patience"]:
                    print("Early stopping triggered!")
                    break
                
                # If no improvement for 5 epochs, reduce learning rate
                if no_improvement_epochs >= 5:
                    print("Reducing learning rate...")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    no_improvement_epochs = 0
            
            # Force garbage collection after each epoch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("\n=== Training Completed ===")
        return True
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        error_model_path = os.path.join(model_path, "error_recovery.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, error_model_path)
        return False

if __name__ == "__main__":
    # You can run this script directly from the console
    train_model() 