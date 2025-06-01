#pragma once

#include <SFML/Graphics.hpp>

#include "../gamestate.hpp"
#include "gui_constants.hpp"
#include "gui_utils.hpp"

namespace GUI {

class BoardRenderer {
public:
    BoardRenderer(sf::RenderWindow& window, LayoutDimensions const& layout);

    // Main rendering function
    void render(Board const& board, Player current_player, int actions_left,
                ElementType highlight_type = ElementType::NONE, int highlight_row = -1,
                int highlight_col = -1);

    void draw_ai_thinking_indicator(int samples_done, int samples_total);

    // Set highlighting for interactive feedback
    void set_highlight(ElementType type, int row, int col);
    void clear_highlight();

private:
    sf::RenderWindow& m_window;
    LayoutDimensions const& m_layout;
    sf::Font m_font;

    // Highlight state
    ElementType m_highlight_type = ElementType::NONE;
    int m_highlight_row = -1;
    int m_highlight_col = -1;

    // Drawing functions (ports from Python code)
    void draw_board();
    void draw_board_perimeter();
    void draw_cells();
    void draw_corners();
    void draw_wall_shadows();

    void draw_walls(Board const& board);
    void draw_players(Board const& board);
    void draw_goals(Board const& board);
    void draw_game_info(Board const& board, Player current_player, int actions_left);
    void draw_reachable_cells(Board const& board, Player current_player, int actions_left);

    // Individual drawing primitives
    void draw_cell(int row, int col, sf::Color const& color);
    void draw_hwall(int row, int col, sf::Color const& color);
    void draw_vwall(int row, int col, sf::Color const& color);
    void draw_player(Cell const& position, sf::Color const& color, int offset_x = 0,
                     int offset_y = 0);
    void draw_empty_rect(sf::Color const& color, float left, float top, float right, float bottom,
                         float width, float height);

    // Highlighting functions
    void highlight_cell(int row, int col);
    void highlight_vwall(int row, int col);
    void highlight_hwall(int row, int col);

    // Text rendering
    void write_text(std::string const& text, float left, float top,
                    sf::Color const& color = GUI::BLACK);

    // Helper functions
    std::vector<Cell> get_cells_at_distance1(Board const& board, Player player);
    std::vector<Cell> get_cells_at_distance2(Board const& board, Player player);
};

}  // namespace GUI
