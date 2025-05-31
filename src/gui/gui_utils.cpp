#include "gui_utils.hpp"

#include <algorithm>

#include "gui_constants.hpp"

namespace GUI {

LayoutDimensions::LayoutDimensions(int window_w, int window_h, int cols, int rows)
    : window_width(window_w), window_height(window_h), board_cols(cols), board_rows(rows) {
    // Calculate layout dimensions (direct port from Python initSizes function)
    perimeter_width = std::max(window_width / 100, window_height / 100);
    perimeter_height = perimeter_width;
    margin_width = window_width / 10 - perimeter_width;
    margin_height = window_height / 10 - perimeter_height;

    cell_plus_wall_width = (8 * window_width / 10) / board_cols;
    cell_plus_wall_height = (8 * window_height / 10) / board_rows;

    cell_width = 4 * cell_plus_wall_width / 5;
    cell_height = 4 * cell_plus_wall_height / 5;
    wall_width = cell_plus_wall_width - cell_width;
    wall_height = cell_plus_wall_height - cell_height;

    wall_highlight_width = wall_width / 4;
    wall_highlight_height = wall_height / 4;
    cell_highlight_width = cell_width / 8;
    cell_highlight_height = cell_height / 8;

    board_width = cell_plus_wall_width * board_cols - wall_width;
    board_height = cell_plus_wall_height * board_rows - wall_height;

    shared_cell_offset_width = cell_width / 7;
    shared_cell_offset_height = cell_height / 7;
}

sf::Vector2f LayoutDimensions::cell_pos_to_coords(int row, int col) const {
    float left = margin_width + perimeter_width + cell_plus_wall_width * col;
    float top = margin_height + perimeter_height + cell_plus_wall_height * row;
    return sf::Vector2f(left, top);
}

sf::Vector2f LayoutDimensions::corner_pos_to_coords(int row, int col) const {
    auto cell_pos = cell_pos_to_coords(row, col);
    return sf::Vector2f(cell_pos.x + cell_width, cell_pos.y + cell_height);
}

sf::Vector2f LayoutDimensions::hwall_pos_to_coords(int row, int col) const {
    auto cell_pos = cell_pos_to_coords(row, col);
    return sf::Vector2f(cell_pos.x, cell_pos.y + cell_height);
}

sf::Vector2f LayoutDimensions::vwall_pos_to_coords(int row, int col) const {
    auto cell_pos = cell_pos_to_coords(row, col);
    return sf::Vector2f(cell_pos.x + cell_width, cell_pos.y);
}

sf::Color increment_color(sf::Color const& color, int increment) {
    int r = std::max(0, std::min(255, static_cast<int>(color.r) + increment));
    int g = std::max(0, std::min(255, static_cast<int>(color.g) + increment));
    int b = std::max(0, std::min(255, static_cast<int>(color.b) + increment));
    return sf::Color(r, g, b, color.a);
}

}  // namespace GUI
