#pragma once

#include <SFML/Graphics.hpp>

namespace GUI {

// Colors (translated from Python pygame colors)
const sf::Color BACKGROUND_COLOR(60, 60, 100);          // NAVYBLUE
const sf::Color CELL_COLOR(255, 255, 255);              // WHITE
const sf::Color CORNER_COLOR(100, 100, 100);            // GRAY
const sf::Color WALL_SHADOW_COLOR(180, 180, 180);       // LIGHTGRAY
const sf::Color FORBIDDEN_WALL_COLOR(255, 128, 128);    // LIGHTRED
const sf::Color PLAYER1_COLOR(255, 0, 0);               // RED
const sf::Color PLAYER2_COLOR(0, 0, 255);               // BLUE
const sf::Color HIGHLIGHT_COLOR(255, 0, 255);           // PURPLE
const sf::Color BLACK(0, 0, 0);

// Color increments for visual effects
const int REACHABLE_CELLS_1ACTION_COLOR_INCREMENT = 80;
const int REACHABLE_CELLS_2ACTIONS_COLOR_INCREMENT = 160;
const int WALL_COLOR_INCREMENT = -90;
const int GOAL_COLOR_INCREMENT = 120;

// Default board dimensions (flexible - can be overridden)
const int DEFAULT_BOARD_COLS = 5;  // Using C++ default instead of Python's hardcoded 12
const int DEFAULT_BOARD_ROWS = 5;  // Using C++ default instead of Python's hardcoded 10

// Default window dimensions
const int DEFAULT_WINDOW_WIDTH = 800;
const int DEFAULT_WINDOW_HEIGHT = 700;  // Increased from 600 to accommodate text below board

// Font settings
const int FONT_SIZE = 20;
const int FPS = 60;

} // namespace GUI 
