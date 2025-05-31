#pragma once

#include <SFML/Graphics.hpp>

namespace GUI {

// Colors (translated from Python pygame colors)
sf::Color const BACKGROUND_COLOR(60, 60, 100);        // NAVYBLUE
sf::Color const CELL_COLOR(255, 255, 255);            // WHITE
sf::Color const CORNER_COLOR(100, 100, 100);          // GRAY
sf::Color const WALL_SHADOW_COLOR(180, 180, 180);     // LIGHTGRAY
sf::Color const FORBIDDEN_WALL_COLOR(255, 128, 128);  // LIGHTRED
sf::Color const PLAYER1_COLOR(255, 0, 0);             // RED
sf::Color const PLAYER2_COLOR(0, 0, 255);             // BLUE
sf::Color const HIGHLIGHT_COLOR(255, 0, 255);         // PURPLE
sf::Color const BLACK(0, 0, 0);

// Color increments for visual effects
int const REACHABLE_CELLS_1ACTION_COLOR_INCREMENT = 80;
int const REACHABLE_CELLS_2ACTIONS_COLOR_INCREMENT = 160;
int const WALL_COLOR_INCREMENT = -90;
int const GOAL_COLOR_INCREMENT = 120;

// Default board dimensions (flexible - can be overridden)
int const DEFAULT_BOARD_COLS = 5;  // Using C++ default instead of Python's hardcoded 12
int const DEFAULT_BOARD_ROWS = 5;  // Using C++ default instead of Python's hardcoded 10

// Default window dimensions
int const DEFAULT_WINDOW_WIDTH = 800;
int const DEFAULT_WINDOW_HEIGHT = 700;  // Increased from 600 to accommodate text below board

// Font settings
int const FONT_SIZE = 20;
int const FPS = 60;

}  // namespace GUI
