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
        enum Type {
            NONE,
            MOVE_TO_CELL,
            PLACE_WALL
        };
        Type type = NONE;
        Cell target_cell{0, 0};
        Wall target_wall{};
    };

    InputHandler(LayoutDimensions const& layout);

    // Handle mouse click and return the appropriate action
    MouseAction handle_mouse_click(sf::Vector2i mouse_pos, Board const& board,
                                   Player current_player) const;

    // Handle keyboard input for movement
    std::optional<Direction> handle_key_press(sf::Keyboard::Key key) const;

    // Get the highlighted element under mouse cursor
    std::tuple<ElementType, int, int> get_element_at_position(sf::Vector2i mouse_pos) const;

    // Check if a cell is reachable by the current player
    bool is_cell_reachable_in_1_action(Board const& board, Player player, Cell target) const;
    bool is_cell_reachable_in_2_actions(Board const& board, Player player, Cell target) const;

private:
    LayoutDimensions const& m_layout;
};

}  // namespace GUI
