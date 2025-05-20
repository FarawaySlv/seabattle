import torch
import torch.nn as nn
import json
import os
from models.transformer_player import TransformerPlayer
from models.training_data import TrainingData

def train_model(
    model_path: str = "models/battleship/checkpoints",
    config_path: str = "models/battleship/configs/model_config.json",
    min_games: int = 100
):
    """Train the transformer model using collected game data"""
    print("\n=== Starting Training Process ===")
    
    # Initialize model and training data
    model = TransformerPlayer(model_path, config_path)
    
    # Load and prepare training data
    print("Loading game logs...")
    success = model.training_data.load_from_logs(min_games)
    if not success:
        print(f"Failed to load minimum required games ({min_games})")
        return False
        
    print(f"Successfully loaded {len(model.training_data.games)} games")
    
    # Get training and validation data
    print("Splitting data into training and validation sets...")
    train_data, val_data = model.training_data.get_validation_data(
        model.config["training"]["validation_split"]
    )
    
    if not train_data or not val_data:
        print("Failed to split data into training and validation sets!")
        return False
        
    print(f"Training set size: {len(train_data.games)} games")
    print(f"Validation set size: {len(val_data.games)} games")
    
    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=model.config["training"]["learning_rate"]
    )
    
    # Training loop
    print("\n=== Training Loop ===")
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Calculate total batches for progress tracking
    batch_size = min(model.config["training"]["batch_size"], 32)
    total_train_batches = len(train_data.games) // batch_size
    total_val_batches = len(val_data.games) // batch_size
    print(f"Total training batches per epoch: {total_train_batches}")
    print(f"Total validation batches per epoch: {total_val_batches}")
    print(f"Using batch size: {batch_size}")
    
    try:
        for epoch in range(model.config["training"]["max_epochs"]):
            print(f"\nEpoch {epoch + 1}/{model.config['training']['max_epochs']}")
            
            # Training phase
            model.model.train()
            total_train_loss = 0
            batches = 0
            
            print("Training phase...")
            while True:
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                batch = train_data.get_training_batch(batch_size)
                if not batch:
                    break
                    
                optimizer.zero_grad()
                loss = model.train_step(batch)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                batches += 1
                
                if batches % 5 == 0:
                    avg_loss = total_train_loss / batches
                    progress = (batches / total_train_batches) * 100
                    print(f"Training Progress: {progress:.1f}% - Batch {batches}/{total_train_batches}")
                    print(f"Current Loss: {loss.item():.4f}, Average Loss: {avg_loss:.4f}")
                    
                    # Save checkpoint every 25 batches
                    if batches % 25 == 0:
                        model.save_model(f"checkpoint_epoch_{epoch + 1}_batch_{batches}")
            
            avg_train_loss = total_train_loss / batches if batches > 0 else float('inf')
            print(f"Training completed - Average Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            print("\nValidation phase...")
            model.model.eval()
            total_val_loss = 0
            batches = 0
            
            with torch.no_grad():
                while True:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    batch = val_data.get_training_batch(batch_size)
                    if not batch:
                        break
                        
                    loss = model.train_step(batch)
                    total_val_loss += loss.item()
                    batches += 1
                    
                    if batches % 5 == 0:
                        avg_loss = total_val_loss / batches
                        progress = (batches / total_val_batches) * 100
                        print(f"Validation Progress: {progress:.1f}% - Batch {batches}/{total_val_batches}")
                        print(f"Current Loss: {loss.item():.4f}, Average Loss: {avg_loss:.4f}")
            
            avg_val_loss = total_val_loss / batches if batches > 0 else float('inf')
            print(f"Validation completed - Average Loss: {avg_val_loss:.4f}")
            
            # Save epoch checkpoint
            model.save_model(f"epoch_{epoch + 1}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print("New best model found! Saving...")
                model.save_model("best")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{model.config['training']['early_stopping_patience']}")
                if patience_counter >= model.config["training"]["early_stopping_patience"]:
                    print("Early stopping triggered!")
                    break
            
            # Force garbage collection after each epoch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("\n=== Training Completed ===")
        return True
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        model.save_model("error_recovery")
        return False

if __name__ == "__main__":
    # You can run this script directly from the console
    train_model() 