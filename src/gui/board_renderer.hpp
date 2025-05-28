#pragma once

#include <SFML/Graphics.hpp>
#include "../gamestate.hpp"
#include "gui_constants.hpp"

namespace GUI {

class BoardRenderer {
public:
    BoardRenderer(sf::RenderWindow& window, const LayoutDimensions& layout);
    
    // Main rendering function
    void render(const Board& board, Player current_player, int actions_left, 
                ElementType highlight_type = ElementType::NONE, int highlight_row = -1, int highlight_col = -1);
    
    // Render with AI thinking indicator
    void renderWithAIThinking(const Board& board, Player current_player, int actions_left, 
                              ElementType highlight_type = ElementType::NONE, int highlight_row = -1, int highlight_col = -1);
    
    // Set highlighting for interactive feedback
    void setHighlight(ElementType type, int row, int col);
    void clearHighlight();
    
private:
    sf::RenderWindow& m_window;
    const LayoutDimensions& m_layout;
    sf::Font m_font;
    
    // Highlight state
    ElementType m_highlight_type = ElementType::NONE;
    int m_highlight_row = -1;
    int m_highlight_col = -1;
    
    // Drawing functions (ports from Python code)
    void drawBoard();
    void drawBoardPerimeter();
    void drawCells();
    void drawCorners();
    void drawWallShadows();
    
    void drawWalls(const Board& board);
    void drawPlayers(const Board& board);
    void drawGoals(const Board& board);
    void drawGameInfo(const Board& board, Player current_player, int actions_left);
    void drawReachableCells(const Board& board, Player current_player, int actions_left);
    void drawAIThinkingIndicator();
    
    // Individual drawing primitives
    void drawCell(int row, int col, const sf::Color& color);
    void drawHWall(int row, int col, const sf::Color& color);
    void drawVWall(int row, int col, const sf::Color& color);
    void drawPlayer(const Cell& position, const sf::Color& color, int offset_x = 0, int offset_y = 0);
    void drawEmptyRect(const sf::Color& color, float left, float top, float right, float bottom, 
                       float width, float height);
    
    // Highlighting functions
    void highlightCell(int row, int col);
    void highlightVWall(int row, int col);
    void highlightHWall(int row, int col);
    
    // Text rendering
    void writeText(const std::string& text, float left, float top, const sf::Color& color = GUI::BLACK);
    
    // Helper functions
    std::vector<Cell> getCellsAtDistance1(const Board& board, Player player);
    std::vector<Cell> getCellsAtDistance2(const Board& board, Player player);
};

} // namespace GUI 
