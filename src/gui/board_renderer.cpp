#include "board_renderer.hpp"
#include <iostream>

namespace GUI {

BoardRenderer::BoardRenderer(sf::RenderWindow& window, const LayoutDimensions& layout)
    : m_window(window), m_layout(layout) {
    
    if (!load_bundled_font()) {
        std::cerr << "Warning: GUI will run without text display due to font loading failure.\n";
    }
}

bool BoardRenderer::load_bundled_font() {
    // Try different paths since executable might run from build/ directory
    std::vector<std::string> font_paths = {
        "../assets/gui/fonts/DejaVuSans.ttf",   // From build/ directory
        "assets/gui/fonts/DejaVuSans.ttf"       // From project root
    };
    
    for (const auto& path : font_paths) {
        if (m_font.loadFromFile(path)) {
            std::cout << "Loaded bundled font: " << path << std::endl;
            return true;
        }
    }
    
    std::cerr << "Error: Could not load bundled font from any of these paths:" << std::endl;
    for (const auto& path : font_paths) {
        std::cerr << "  - " << path << std::endl;
    }
    std::cerr << "Text will not be displayed in the GUI." << std::endl;
    return false;
}

void BoardRenderer::render(const Board& board, Player current_player, int actions_left,
                          ElementType highlight_type, int highlight_row, int highlight_col) {
    // Clear window with background color
    m_window.clear(GUI::BACKGROUND_COLOR);
    
    // Store highlight state
    m_highlight_type = highlight_type;
    m_highlight_row = highlight_row;
    m_highlight_col = highlight_col;
    
    // Render in order (background to foreground)
    draw_board();
    draw_reachable_cells(board, current_player, actions_left);
    draw_walls(board);
    draw_goals(board);
    draw_players(board);
    
    // Draw highlighting
    if (m_highlight_type != ElementType::NONE) {
        switch (m_highlight_type) {
            case ElementType::CELL:
                highlight_cell(m_highlight_row, m_highlight_col);
                break;
            case ElementType::VWALL:
                highlight_vwall(m_highlight_row, m_highlight_col);
                break;
            case ElementType::HWALL:
                highlight_hwall(m_highlight_row, m_highlight_col);
                break;
            default:
                break;
        }
    }
    
    draw_game_info(board, current_player, actions_left);
}

void BoardRenderer::set_highlight(ElementType type, int row, int col) {
    m_highlight_type = type;
    m_highlight_row = row;
    m_highlight_col = col;
}

void BoardRenderer::clear_highlight() {
    m_highlight_type = ElementType::NONE;
    m_highlight_row = -1;
    m_highlight_col = -1;
}

void BoardRenderer::draw_board() {
    draw_board_perimeter();
    draw_cells();
    draw_corners();
    draw_wall_shadows();
}

void BoardRenderer::draw_board_perimeter() {
    float left = m_layout.margin_width;
    float top = m_layout.margin_height;
    float right = m_layout.margin_width + 2 * m_layout.perimeter_width + m_layout.board_width;
    float bottom = m_layout.margin_height + 2 * m_layout.perimeter_height + m_layout.board_height;
    
    draw_empty_rect(GUI::BLACK, left, top, right, bottom, 
                  m_layout.perimeter_width, m_layout.perimeter_height);
}

void BoardRenderer::draw_cells() {
    for (int row = 0; row < m_layout.board_rows; ++row) {
        for (int col = 0; col < m_layout.board_cols; ++col) {
            draw_cell(row, col, GUI::CELL_COLOR);
        }
    }
}

void BoardRenderer::draw_corners() {
    for (int row = 0; row < m_layout.board_rows - 1; ++row) {
        for (int col = 0; col < m_layout.board_cols - 1; ++col) {
            auto pos = m_layout.corner_pos_to_coords(row, col);
            sf::RectangleShape corner(sf::Vector2f(m_layout.wall_width, m_layout.wall_height));
            corner.setPosition(pos);
            corner.setFillColor(GUI::CORNER_COLOR);
            m_window.draw(corner);
        }
    }
}

void BoardRenderer::draw_wall_shadows() {
    for (int row = 0; row < m_layout.board_rows; ++row) {
        for (int col = 0; col < m_layout.board_cols; ++col) {
            if (row < m_layout.board_rows - 1) {
                draw_hwall(row, col, GUI::WALL_SHADOW_COLOR);
            }
            if (col < m_layout.board_cols - 1) {
                draw_vwall(row, col, GUI::WALL_SHADOW_COLOR);
            }
        }
    }
}

void BoardRenderer::draw_cell(int row, int col, const sf::Color& color) {
    auto pos = m_layout.cell_pos_to_coords(row, col);
    sf::RectangleShape cell(sf::Vector2f(m_layout.cell_width, m_layout.cell_height));
    cell.setPosition(pos);
    cell.setFillColor(color);
    m_window.draw(cell);
}

void BoardRenderer::draw_hwall(int row, int col, const sf::Color& color) {
    auto pos = m_layout.hwall_pos_to_coords(row, col);
    sf::RectangleShape wall(sf::Vector2f(m_layout.cell_width, m_layout.wall_height));
    wall.setPosition(pos);
    wall.setFillColor(color);
    m_window.draw(wall);
}

void BoardRenderer::draw_vwall(int row, int col, const sf::Color& color) {
    auto pos = m_layout.vwall_pos_to_coords(row, col);
    sf::RectangleShape wall(sf::Vector2f(m_layout.wall_width, m_layout.cell_height));
    wall.setPosition(pos);
    wall.setFillColor(color);
    m_window.draw(wall);
}

void BoardRenderer::draw_walls(const Board& board) {
    // Draw actual placed walls
    for (int row = 0; row < m_layout.board_rows; ++row) {
        for (int col = 0; col < m_layout.board_cols; ++col) {
            Cell cell{col, row};
            
            // Check horizontal walls (going down from this cell)
            if (row < m_layout.board_rows - 1) {
                Wall hwall{cell, Wall::Down};
                if (board.is_blocked(hwall)) {
                    // Determine wall color based on who placed it
                    auto owner = board.wall_owner(hwall);
                    sf::Color wall_color;
                    if (owner == Player::Red) {
                        wall_color = increment_color(GUI::PLAYER1_COLOR, GUI::WALL_COLOR_INCREMENT);
                    } else if (owner == Player::Blue) {
                        wall_color = increment_color(GUI::PLAYER2_COLOR, GUI::WALL_COLOR_INCREMENT);
                    } else {
                        // Boundary wall or unknown - use neutral color
                        wall_color = increment_color(GUI::CORNER_COLOR, GUI::WALL_COLOR_INCREMENT);
                    }
                    draw_hwall(row, col, wall_color);
                }
            }
            
            // Check vertical walls (going right from this cell)
            if (col < m_layout.board_cols - 1) {
                Wall vwall{cell, Wall::Right};
                if (board.is_blocked(vwall)) {
                    // Determine wall color based on who placed it
                    auto owner = board.wall_owner(vwall);
                    sf::Color wall_color;
                    if (owner == Player::Red) {
                        wall_color = increment_color(GUI::PLAYER1_COLOR, GUI::WALL_COLOR_INCREMENT);
                    } else if (owner == Player::Blue) {
                        wall_color = increment_color(GUI::PLAYER2_COLOR, GUI::WALL_COLOR_INCREMENT);
                    } else {
                        // Boundary wall or unknown - use neutral color
                        wall_color = increment_color(GUI::CORNER_COLOR, GUI::WALL_COLOR_INCREMENT);
                    }
                    draw_vwall(row, col, wall_color);
                }
            }
        }
    }
}

void BoardRenderer::draw_player(const Cell& position, const sf::Color& color, int offset_x, int offset_y) {
    auto cell_pos = m_layout.cell_pos_to_coords(position.row, position.column);
    
    float ellipse_width = m_layout.cell_width * 0.6f;
    float ellipse_height = m_layout.cell_height * 0.6f;
    float ellipse_left = cell_pos.x + (m_layout.cell_width - ellipse_width) / 2 + offset_x;
    float ellipse_top = cell_pos.y + (m_layout.cell_height - ellipse_height) / 2 + offset_y;
    
    // Use ellipse that matches cell proportions instead of fixed circle
    sf::CircleShape player(ellipse_height / 2);  // Use height as radius
    player.setScale(ellipse_width / ellipse_height, 1.0f);  // Scale to make it elliptical
    player.setPosition(ellipse_left, ellipse_top);
    player.setFillColor(color);
    player.setOutlineColor(GUI::BLACK);
    player.setOutlineThickness(1);
    m_window.draw(player);
}

void BoardRenderer::draw_players(const Board& board) {
    Cell red_pos = board.position(Player::Red);
    Cell blue_pos = board.position(Player::Blue);
    
    if (red_pos == blue_pos) {
        // Draw players with offset when on same cell
        draw_player(blue_pos, GUI::PLAYER2_COLOR, 
                  m_layout.shared_cell_offset_width, m_layout.shared_cell_offset_height);
        draw_player(red_pos, GUI::PLAYER1_COLOR);
    } else {
        draw_player(red_pos, GUI::PLAYER1_COLOR);
        draw_player(blue_pos, GUI::PLAYER2_COLOR);
    }
}

void BoardRenderer::draw_goals(const Board& board) {
    Cell red_goal = board.goal(Player::Red);
    Cell blue_goal = board.goal(Player::Blue);
    
    auto red_goal_color = increment_color(GUI::PLAYER1_COLOR, GUI::GOAL_COLOR_INCREMENT);
    auto blue_goal_color = increment_color(GUI::PLAYER2_COLOR, GUI::GOAL_COLOR_INCREMENT);
    
    draw_player(red_goal, red_goal_color);
    draw_player(blue_goal, blue_goal_color);
}

void BoardRenderer::draw_reachable_cells(const Board& board, Player current_player, int actions_left) {
    auto cells_1_action = get_cells_at_distance1(board, current_player);
    auto cells_2_actions = get_cells_at_distance2(board, current_player);
    
    if (actions_left >= 2) {
        sf::Color color_2 = (current_player == Player::Red) ? GUI::PLAYER1_COLOR : GUI::PLAYER2_COLOR;
        color_2 = increment_color(color_2, GUI::REACHABLE_CELLS_2ACTIONS_COLOR_INCREMENT);
        
        for (const Cell& cell : cells_2_actions) {
            draw_cell(cell.row, cell.column, color_2);
        }
    }
    
    if (actions_left >= 1) {
        sf::Color color_1 = (current_player == Player::Red) ? GUI::PLAYER1_COLOR : GUI::PLAYER2_COLOR;
        color_1 = increment_color(color_1, GUI::REACHABLE_CELLS_1ACTION_COLOR_INCREMENT);
        
        for (const Cell& cell : cells_1_action) {
            draw_cell(cell.row, cell.column, color_1);
        }
    }
}

std::vector<Cell> BoardRenderer::get_cells_at_distance1(const Board& board, Player player) {
    std::vector<Cell> result;
    auto dirs = board.legal_directions(player);
    Cell pos = board.position(player);
    
    for (Direction dir : dirs) {
        result.push_back(pos.step(dir));
    }
    return result;
}

std::vector<Cell> BoardRenderer::get_cells_at_distance2(const Board& board, Player player) {
    std::vector<Cell> result;
    auto cells_1 = get_cells_at_distance1(board, player);
    
    for (const Cell& intermediate : cells_1) {
        // Create temporary board to check legal moves from intermediate position
        Board temp_board = board;
        // Find which direction gets us to intermediate
        Cell current = board.position(player);
        for (Direction dir : board.legal_directions(player)) {
            if (current.step(dir) == intermediate) {
                temp_board.take_step(player, dir);
                break;
            }
        }
        
        auto dirs_from_intermediate = temp_board.legal_directions(player);
        for (Direction dir : dirs_from_intermediate) {
            Cell target = intermediate.step(dir);
            // Avoid duplicates
            if (std::find(result.begin(), result.end(), target) == result.end() &&
                std::find(cells_1.begin(), cells_1.end(), target) == cells_1.end()) {
                result.push_back(target);
            }
        }
    }
    return result;
}

void BoardRenderer::highlight_cell(int row, int col) {
    auto pos = m_layout.cell_pos_to_coords(row, col);
    float left = pos.x;
    float top = pos.y;
    float right = left + m_layout.cell_width;
    float bottom = top + m_layout.cell_height;
    
    draw_empty_rect(GUI::HIGHLIGHT_COLOR, left, top, right, bottom,
                  m_layout.cell_highlight_width, m_layout.cell_highlight_height);
}

void BoardRenderer::highlight_vwall(int row, int col) {
    auto pos = m_layout.vwall_pos_to_coords(row, col);
    float left = pos.x;
    float top = pos.y;
    float right = left + m_layout.wall_width;
    float bottom = top + m_layout.cell_height;
    
    draw_empty_rect(GUI::HIGHLIGHT_COLOR, left, top, right, bottom,
                  m_layout.wall_highlight_width, m_layout.wall_highlight_height);
}

void BoardRenderer::highlight_hwall(int row, int col) {
    auto pos = m_layout.hwall_pos_to_coords(row, col);
    float left = pos.x;
    float top = pos.y;
    float right = left + m_layout.cell_width;
    float bottom = top + m_layout.wall_height;
    
    draw_empty_rect(GUI::HIGHLIGHT_COLOR, left, top, right, bottom,
                  m_layout.wall_highlight_width, m_layout.wall_highlight_height);
}

void BoardRenderer::draw_empty_rect(const sf::Color& color, float left, float top, float right, float bottom,
                                  float width, float height) {
    // Draw rectangle outline (top, bottom, left, right borders)
    sf::RectangleShape top_border(sf::Vector2f(right - left, height));
    top_border.setPosition(left, top);
    top_border.setFillColor(color);
    m_window.draw(top_border);
    
    sf::RectangleShape bottom_border(sf::Vector2f(right - left, height));
    bottom_border.setPosition(left, bottom - height);
    bottom_border.setFillColor(color);
    m_window.draw(bottom_border);
    
    sf::RectangleShape left_border(sf::Vector2f(width, bottom - top));
    left_border.setPosition(left, top);
    left_border.setFillColor(color);
    m_window.draw(left_border);
    
    sf::RectangleShape right_border(sf::Vector2f(width, bottom - top));
    right_border.setPosition(right - width, top);
    right_border.setFillColor(color);
    m_window.draw(right_border);
}

void BoardRenderer::write_text(const std::string& text, float left, float top, const sf::Color& color) {
    if (m_font.getInfo().family.empty()) {
        return; // No font loaded
    }
    
    sf::Text sftext;
    sftext.setFont(m_font);
    sftext.setString(text);
    sftext.setCharacterSize(GUI::FONT_SIZE);
    sftext.setFillColor(color);
    sftext.setPosition(left, top);
    m_window.draw(sftext);
}

void BoardRenderer::draw_game_info(const Board& board, Player current_player, int actions_left) {
    // Position text below the board
    float text_start_y = m_layout.margin_height + m_layout.perimeter_height + m_layout.board_height + m_layout.perimeter_height + 10;
    float text_x = m_layout.margin_width;
    
    std::string current_player_str = (current_player == Player::Red) ? "Red" : "Blue";
    write_text("Current Player: " + current_player_str, text_x, text_start_y);
    write_text("Actions Left: " + std::to_string(actions_left), text_x, text_start_y + 40);
    
    // Display distances to goals
    int red_distance = board.distance(board.position(Player::Red), board.goal(Player::Red));
    int blue_distance = board.distance(board.position(Player::Blue), board.goal(Player::Blue));
    
    write_text("Red Distance: " + std::to_string(red_distance), text_x + 250, text_start_y);
    write_text("Blue Distance: " + std::to_string(blue_distance), text_x + 250, text_start_y + 40);
}

void BoardRenderer::render_with_ai_thinking(const Board& board, Player current_player, int actions_left, 
                                          ElementType highlight_type, int highlight_row, int highlight_col) {
    // Regular rendering
    render(board, current_player, actions_left, highlight_type, highlight_row, highlight_col);
    
    // Add AI thinking overlay
    draw_ai_thinking_indicator();
}

void BoardRenderer::draw_ai_thinking_indicator() {
    // Draw a semi-transparent overlay
    float board_left = m_layout.margin_width + m_layout.perimeter_width;
    float board_top = m_layout.margin_height + m_layout.perimeter_height;
    float board_width = m_layout.board_width;
    float board_height = m_layout.board_height;
    
    sf::RectangleShape overlay(sf::Vector2f(board_width, board_height));
    overlay.setPosition(board_left, board_top);
    overlay.setFillColor(sf::Color(0, 0, 0, 128)); // Semi-transparent black
    m_window.draw(overlay);
    
    // Draw "AI Thinking..." text in center
    float center_x = board_left + board_width / 2 - 80; // Approximate text width offset
    float center_y = board_top + board_height / 2 - 16; // Half font size offset
    
    write_text("AI Thinking...", center_x, center_y, sf::Color::White);
}

} // namespace GUI 

