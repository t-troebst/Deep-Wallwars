#pragma once

#include <SFML/Graphics.hpp>

namespace GUI {

// Layout calculation functions (will be defined based on actual board size)
struct LayoutDimensions {
    int window_width;
    int window_height;
    int board_cols;
    int board_rows;
    
    // Calculated dimensions
    int margin_width;
    int margin_height;
    int perimeter_width;
    int perimeter_height;
    int cell_plus_wall_width;
    int cell_plus_wall_height;
    int cell_width;
    int cell_height;
    int wall_width;
    int wall_height;
    int wall_highlight_width;
    int wall_highlight_height;
    int cell_highlight_width;
    int cell_highlight_height;
    int board_width;
    int board_height;
    int shared_cell_offset_width;
    int shared_cell_offset_height;
    
    LayoutDimensions(int window_w, int window_h, int cols, int rows);
    
    // Coordinate conversion methods
    sf::Vector2f cellPosToCoords(int row, int col) const;
    sf::Vector2f cornerPosToCoords(int row, int col) const;
    sf::Vector2f hWallPosToCoords(int row, int col) const;
    sf::Vector2f vWallPosToCoords(int row, int col) const;
};

// Helper function to increment color with bounds checking
sf::Color incrementColor(const sf::Color& color, int increment);

// Element types for mouse interaction
enum class ElementType {
    NONE,
    CELL,
    VWALL,
    HWALL,
    CORNER
};

} // namespace GUI 
