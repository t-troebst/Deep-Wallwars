#pragma once

#include <SFML/Graphics.hpp>
#include <optional>
#include "../gamestate.hpp"
#include "gui_constants.hpp"
#include "gui_utils.hpp"

namespace GUI {

class InputHandler {
public:
    struct MouseAction {
        enum Type { NONE, MOVE_TO_CELL, PLACE_WALL };
        Type type = NONE;
        Cell target_cell{0, 0};
        Wall target_wall{};
    };
    
    InputHandler(const LayoutDimensions& layout);
    
    // Handle mouse click and return the appropriate action
    MouseAction handleMouseClick(sf::Vector2i mouse_pos, const Board& board, Player current_player) const;
    
    // Handle keyboard input for movement
    std::optional<Direction> handleKeyPress(sf::Keyboard::Key key) const;
    
    // Get the highlighted element under mouse cursor
    std::tuple<ElementType, int, int> getElementAtPosition(sf::Vector2i mouse_pos) const;
    
    // Check if a cell is reachable by the current player
    bool isCellReachableIn1Action(const Board& board, Player player, Cell target) const;
    bool isCellReachableIn2Actions(const Board& board, Player player, Cell target) const;
    
private:
    const LayoutDimensions& m_layout;
};

} // namespace GUI 
