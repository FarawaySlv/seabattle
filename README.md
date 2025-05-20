# Sea Battle Game with AI

A Python implementation of the classic Sea Battle (Battleship) game with an AI opponent powered by a transformer model.

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/seabattle.git
cd seabattle
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
seabattle/
├── src/
│   ├── game/           # Core game logic
│   ├── models/         # AI model implementation
│   │   ├── train_transformer.py  # Model training script
│   │   └── transformer_player.py # Transformer model implementation
│   └── ui/            # User interface components
├── models/
│   └── battleship/    # Model checkpoints and training data
├── requirements.txt
└── README.md
```

## Running the Game

To start a new game against the AI:

```bash
python src/main.py
```

The game will launch with a graphical interface where you can:

- Place your ships on the board
- Make shots at the AI's board
- See the AI's moves on your board

## Training the Model

To train a new model:

1. Navigate to the project directory:

```bash
cd seabattle
```

2. Run the training script:

```bash
python src/models/train_transformer.py
```

Training parameters can be configured in the training script. The model checkpoints will be saved in `models/battleship/checkpoints/`.

### Training Configuration

You can modify the following parameters in `src/models/train_transformer.py`:

- Learning rate
- Number of epochs
- Batch size
- Model architecture parameters
- Training data settings

## Using a Trained Model

The game automatically uses the latest trained model from the checkpoints directory. To use a specific model:

1. Place the model checkpoint file in `models/battleship/checkpoints/`
2. Update the model path in the training script if needed

## Game Rules

1. Each player has a 10x10 grid
2. Players place their ships (different sizes) on their grid
3. Players take turns shooting at coordinates
4. A ship is sunk when all its cells are hit
5. The first player to sink all opponent's ships wins

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Add your license information here]
