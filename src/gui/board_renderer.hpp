#pragma once

#include <SFML/Graphics.hpp>
#include "../gamestate.hpp"
#include "gui_constants.hpp"
#include "gui_utils.hpp"

namespace GUI {

class BoardRenderer {
public:
    BoardRenderer(sf::RenderWindow& window, const LayoutDimensions& layout);
    
    // Main rendering function
    void render(const Board& board, Player current_player, int actions_left, 
                ElementType highlight_type = ElementType::NONE, int highlight_row = -1, int highlight_col = -1);
    
    // Render with AI thinking indicator
    void render_with_ai_thinking(const Board& board, Player current_player, int actions_left, 
                              ElementType highlight_type = ElementType::NONE, int highlight_row = -1, int highlight_col = -1);
    
    // Set highlighting for interactive feedback
    void set_highlight(ElementType type, int row, int col);
    void clear_highlight();
    
private:
    sf::RenderWindow& m_window;
    const LayoutDimensions& m_layout;
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
    
    void draw_walls(const Board& board);
    void draw_players(const Board& board);
    void draw_goals(const Board& board);
    void draw_game_info(const Board& board, Player current_player, int actions_left);
    void draw_reachable_cells(const Board& board, Player current_player, int actions_left);
    void draw_ai_thinking_indicator();
    
    // Individual drawing primitives
    void draw_cell(int row, int col, const sf::Color& color);
    void draw_hwall(int row, int col, const sf::Color& color);
    void draw_vwall(int row, int col, const sf::Color& color);
    void draw_player(const Cell& position, const sf::Color& color, int offset_x = 0, int offset_y = 0);
    void draw_empty_rect(const sf::Color& color, float left, float top, float right, float bottom, 
                       float width, float height);
    
    // Highlighting functions
    void highlight_cell(int row, int col);
    void highlight_vwall(int row, int col);
    void highlight_hwall(int row, int col);
    
    // Text rendering
    void write_text(const std::string& text, float left, float top, const sf::Color& color = GUI::BLACK);
    
    // Helper functions
    std::vector<Cell> get_cells_at_distance1(const Board& board, Player player);
    std::vector<Cell> get_cells_at_distance2(const Board& board, Player player);
    
    // Font loading helper
    bool load_bundled_font();
};

} // namespace GUI 
