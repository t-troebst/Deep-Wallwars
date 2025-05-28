#include "board_renderer.hpp"
#include <iostream>

namespace GUI {

BoardRenderer::BoardRenderer(sf::RenderWindow& window, const LayoutDimensions& layout)
    : m_window(window), m_layout(layout) {
    
    // Try to load a default font - for now we'll handle missing font gracefully
    if (!m_font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
        // If system font doesn't load, we'll render without text for now
        std::cerr << "Warning: Could not load font, text will not be displayed\n";
    }
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
    drawBoard();
    drawReachableCells(board, current_player, actions_left);
    drawWalls(board);
    drawGoals(board);
    drawPlayers(board);
    
    // Draw highlighting
    if (m_highlight_type != ElementType::NONE) {
        switch (m_highlight_type) {
            case ElementType::CELL:
                highlightCell(m_highlight_row, m_highlight_col);
                break;
            case ElementType::VWALL:
                highlightVWall(m_highlight_row, m_highlight_col);
                break;
            case ElementType::HWALL:
                highlightHWall(m_highlight_row, m_highlight_col);
                break;
            default:
                break;
        }
    }
    
    drawGameInfo(board, current_player, actions_left);
}

void BoardRenderer::setHighlight(ElementType type, int row, int col) {
    m_highlight_type = type;
    m_highlight_row = row;
    m_highlight_col = col;
}

void BoardRenderer::clearHighlight() {
    m_highlight_type = ElementType::NONE;
    m_highlight_row = -1;
    m_highlight_col = -1;
}

void BoardRenderer::drawBoard() {
    drawBoardPerimeter();
    drawCells();
    drawCorners();
    drawWallShadows();
}

void BoardRenderer::drawBoardPerimeter() {
    float left = m_layout.margin_width;
    float top = m_layout.margin_height;
    float right = m_layout.margin_width + 2 * m_layout.perimeter_width + m_layout.board_width;
    float bottom = m_layout.margin_height + 2 * m_layout.perimeter_height + m_layout.board_height;
    
    drawEmptyRect(GUI::BLACK, left, top, right, bottom, 
                  m_layout.perimeter_width, m_layout.perimeter_height);
}

void BoardRenderer::drawCells() {
    for (int row = 0; row < m_layout.board_rows; ++row) {
        for (int col = 0; col < m_layout.board_cols; ++col) {
            drawCell(row, col, GUI::CELL_COLOR);
        }
    }
}

void BoardRenderer::drawCorners() {
    for (int row = 0; row < m_layout.board_rows - 1; ++row) {
        for (int col = 0; col < m_layout.board_cols - 1; ++col) {
            auto pos = m_layout.cornerPosToCoords(row, col);
            sf::RectangleShape corner(sf::Vector2f(m_layout.wall_width, m_layout.wall_height));
            corner.setPosition(pos);
            corner.setFillColor(GUI::CORNER_COLOR);
            m_window.draw(corner);
        }
    }
}

void BoardRenderer::drawWallShadows() {
    for (int row = 0; row < m_layout.board_rows; ++row) {
        for (int col = 0; col < m_layout.board_cols; ++col) {
            if (row < m_layout.board_rows - 1) {
                drawHWall(row, col, GUI::WALL_SHADOW_COLOR);
            }
            if (col < m_layout.board_cols - 1) {
                drawVWall(row, col, GUI::WALL_SHADOW_COLOR);
            }
        }
    }
}

void BoardRenderer::drawCell(int row, int col, const sf::Color& color) {
    auto pos = m_layout.cellPosToCoords(row, col);
    sf::RectangleShape cell(sf::Vector2f(m_layout.cell_width, m_layout.cell_height));
    cell.setPosition(pos);
    cell.setFillColor(color);
    m_window.draw(cell);
}

void BoardRenderer::drawHWall(int row, int col, const sf::Color& color) {
    auto pos = m_layout.hWallPosToCoords(row, col);
    sf::RectangleShape wall(sf::Vector2f(m_layout.cell_width, m_layout.wall_height));
    wall.setPosition(pos);
    wall.setFillColor(color);
    m_window.draw(wall);
}

void BoardRenderer::drawVWall(int row, int col, const sf::Color& color) {
    auto pos = m_layout.vWallPosToCoords(row, col);
    sf::RectangleShape wall(sf::Vector2f(m_layout.wall_width, m_layout.cell_height));
    wall.setPosition(pos);
    wall.setFillColor(color);
    m_window.draw(wall);
}

void BoardRenderer::drawWalls(const Board& board) {
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
                        wall_color = incrementColor(GUI::PLAYER1_COLOR, GUI::WALL_COLOR_INCREMENT);
                    } else if (owner == Player::Blue) {
                        wall_color = incrementColor(GUI::PLAYER2_COLOR, GUI::WALL_COLOR_INCREMENT);
                    } else {
                        // Boundary wall or unknown - use neutral color
                        wall_color = incrementColor(GUI::CORNER_COLOR, GUI::WALL_COLOR_INCREMENT);
                    }
                    drawHWall(row, col, wall_color);
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
                        wall_color = incrementColor(GUI::PLAYER1_COLOR, GUI::WALL_COLOR_INCREMENT);
                    } else if (owner == Player::Blue) {
                        wall_color = incrementColor(GUI::PLAYER2_COLOR, GUI::WALL_COLOR_INCREMENT);
                    } else {
                        // Boundary wall or unknown - use neutral color
                        wall_color = incrementColor(GUI::CORNER_COLOR, GUI::WALL_COLOR_INCREMENT);
                    }
                    drawVWall(row, col, wall_color);
                }
            }
        }
    }
}

void BoardRenderer::drawPlayer(const Cell& position, const sf::Color& color, int offset_x, int offset_y) {
    auto cell_pos = m_layout.cellPosToCoords(position.row, position.column);
    
    float ellipse_width = m_layout.cell_width * 0.6f;
    float ellipse_height = m_layout.cell_height * 0.6f;
    float ellipse_left = cell_pos.x + (m_layout.cell_width - ellipse_width) / 2 + offset_x;
    float ellipse_top = cell_pos.y + (m_layout.cell_height - ellipse_height) / 2 + offset_y;
    
    sf::CircleShape player(ellipse_width / 2);
    player.setPosition(ellipse_left, ellipse_top);
    player.setFillColor(color);
    player.setOutlineColor(GUI::BLACK);
    player.setOutlineThickness(1);
    m_window.draw(player);
}

void BoardRenderer::drawPlayers(const Board& board) {
    Cell red_pos = board.position(Player::Red);
    Cell blue_pos = board.position(Player::Blue);
    
    if (red_pos == blue_pos) {
        // Draw players with offset when on same cell
        drawPlayer(blue_pos, GUI::PLAYER2_COLOR, 
                  m_layout.shared_cell_offset_width, m_layout.shared_cell_offset_height);
        drawPlayer(red_pos, GUI::PLAYER1_COLOR);
    } else {
        drawPlayer(red_pos, GUI::PLAYER1_COLOR);
        drawPlayer(blue_pos, GUI::PLAYER2_COLOR);
    }
}

void BoardRenderer::drawGoals(const Board& board) {
    Cell red_goal = board.goal(Player::Red);
    Cell blue_goal = board.goal(Player::Blue);
    
    auto red_goal_color = incrementColor(GUI::PLAYER1_COLOR, GUI::GOAL_COLOR_INCREMENT);
    auto blue_goal_color = incrementColor(GUI::PLAYER2_COLOR, GUI::GOAL_COLOR_INCREMENT);
    
    drawPlayer(red_goal, red_goal_color);
    drawPlayer(blue_goal, blue_goal_color);
}

void BoardRenderer::drawReachableCells(const Board& board, Player current_player, int actions_left) {
    auto cells_1_action = getCellsAtDistance1(board, current_player);
    auto cells_2_actions = getCellsAtDistance2(board, current_player);
    
    if (actions_left >= 2) {
        sf::Color color_2 = (current_player == Player::Red) ? GUI::PLAYER1_COLOR : GUI::PLAYER2_COLOR;
        color_2 = incrementColor(color_2, GUI::REACHABLE_CELLS_2ACTIONS_COLOR_INCREMENT);
        
        for (const Cell& cell : cells_2_actions) {
            drawCell(cell.row, cell.column, color_2);
        }
    }
    
    if (actions_left >= 1) {
        sf::Color color_1 = (current_player == Player::Red) ? GUI::PLAYER1_COLOR : GUI::PLAYER2_COLOR;
        color_1 = incrementColor(color_1, GUI::REACHABLE_CELLS_1ACTION_COLOR_INCREMENT);
        
        for (const Cell& cell : cells_1_action) {
            drawCell(cell.row, cell.column, color_1);
        }
    }
}

std::vector<Cell> BoardRenderer::getCellsAtDistance1(const Board& board, Player player) {
    std::vector<Cell> result;
    auto dirs = board.legal_directions(player);
    Cell pos = board.position(player);
    
    for (Direction dir : dirs) {
        result.push_back(pos.step(dir));
    }
    return result;
}

std::vector<Cell> BoardRenderer::getCellsAtDistance2(const Board& board, Player player) {
    std::vector<Cell> result;
    auto cells_1 = getCellsAtDistance1(board, player);
    
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

void BoardRenderer::highlightCell(int row, int col) {
    auto pos = m_layout.cellPosToCoords(row, col);
    float left = pos.x;
    float top = pos.y;
    float right = left + m_layout.cell_width;
    float bottom = top + m_layout.cell_height;
    
    drawEmptyRect(GUI::HIGHLIGHT_COLOR, left, top, right, bottom,
                  m_layout.cell_highlight_width, m_layout.cell_highlight_height);
}

void BoardRenderer::highlightVWall(int row, int col) {
    auto pos = m_layout.vWallPosToCoords(row, col);
    float left = pos.x;
    float top = pos.y;
    float right = left + m_layout.wall_width;
    float bottom = top + m_layout.cell_height;
    
    drawEmptyRect(GUI::HIGHLIGHT_COLOR, left, top, right, bottom,
                  m_layout.wall_highlight_width, m_layout.wall_highlight_height);
}

void BoardRenderer::highlightHWall(int row, int col) {
    auto pos = m_layout.hWallPosToCoords(row, col);
    float left = pos.x;
    float top = pos.y;
    float right = left + m_layout.cell_width;
    float bottom = top + m_layout.wall_height;
    
    drawEmptyRect(GUI::HIGHLIGHT_COLOR, left, top, right, bottom,
                  m_layout.wall_highlight_width, m_layout.wall_highlight_height);
}

void BoardRenderer::drawEmptyRect(const sf::Color& color, float left, float top, float right, float bottom,
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

void BoardRenderer::writeText(const std::string& text, float left, float top, const sf::Color& color) {
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

void BoardRenderer::drawGameInfo(const Board& board, Player current_player, int actions_left) {
    // Position text below the board
    float text_start_y = m_layout.margin_height + m_layout.perimeter_height + m_layout.board_height + m_layout.perimeter_height + 10;
    float text_x = m_layout.margin_width;
    
    std::string current_player_str = (current_player == Player::Red) ? "Red" : "Blue";
    writeText("Current Player: " + current_player_str, text_x, text_start_y);
    writeText("Actions Left: " + std::to_string(actions_left), text_x, text_start_y + 40);
    
    // Display distances to goals
    int red_distance = board.distance(board.position(Player::Red), board.goal(Player::Red));
    int blue_distance = board.distance(board.position(Player::Blue), board.goal(Player::Blue));
    
    writeText("Red Distance: " + std::to_string(red_distance), text_x + 250, text_start_y);
    writeText("Blue Distance: " + std::to_string(blue_distance), text_x + 250, text_start_y + 40);
}

void BoardRenderer::renderWithAIThinking(const Board& board, Player current_player, int actions_left, 
                                          ElementType highlight_type, int highlight_row, int highlight_col) {
    // Regular rendering
    render(board, current_player, actions_left, highlight_type, highlight_row, highlight_col);
    
    // Add AI thinking overlay
    drawAIThinkingIndicator();
}

void BoardRenderer::drawAIThinkingIndicator() {
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
    
    writeText("AI Thinking...", center_x, center_y, sf::Color::White);
}

} // namespace GUI 
