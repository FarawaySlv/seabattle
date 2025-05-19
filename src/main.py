import pygame
import sys
from game.game import Game
from utils.constants import WINDOW_TITLE

def main():
    # Initialize Pygame
    pygame.init()
    
    # Create and run the game
    game = Game()
    game.run()

if __name__ == "__main__":
    main() 