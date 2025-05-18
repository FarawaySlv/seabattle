import pygame
import sys
from game.game import Game
from utils.constants import WINDOW_TITLE

def main():
    # Initialize Pygame
    pygame.init()
    
    # Parse command line arguments for AI difficulty
    ai_difficulty = "medium"  # default difficulty
    if "--easy" in sys.argv:
        ai_difficulty = "easy"
    elif "--hard" in sys.argv:
        ai_difficulty = "hard"
    
    # Create and run the game
    game = Game(ai_difficulty=ai_difficulty)
    game.run()

if __name__ == "__main__":
    main() 