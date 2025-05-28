#pragma once

#include <SFML/Graphics.hpp>
#include <folly/experimental/coro/Task.h>
#include <memory>
#include "../gamestate.hpp"
#include "../mcts.hpp"
#include "../play.hpp"
#include "gui_constants.hpp"
#include "board_renderer.hpp"
#include "input_handler.hpp"

namespace GUI {

class GameGUI {
public:
    GameGUI(int window_width = GUI::DEFAULT_WINDOW_WIDTH, 
            int window_height = GUI::DEFAULT_WINDOW_HEIGHT,
            int board_cols = GUI::DEFAULT_BOARD_COLS,
            int board_rows = GUI::DEFAULT_BOARD_ROWS);
    
    // Static helper to get player choice before GUI creation
    static bool askHumanGoesFirst();
    
    // Static utility to check X11 display status
    static bool checkDisplayConnection();
    
    // Main function to run interactive game with GUI
    folly::coro::Task<GameRecorder> run_interactive_game(Board board, EvaluationFunction ai_model, InteractivePlayOptions opts, bool human_goes_first);
    
private:
    sf::RenderWindow m_window;
    LayoutDimensions m_layout;
    std::unique_ptr<BoardRenderer> m_renderer;
    InputHandler m_input_handler;
    
    // Game state tracking
    Player m_human_player;
    Player m_ai_player;
    int m_actions_left = 2;
    bool m_is_human_turn = true;  // Track whose turn it is
    std::vector<Action> m_human_actions;  // Track human actions for move recording
    bool m_game_over = false;
    Winner m_winner = Winner::Undecided;
    
    // AI computation state
    bool m_ai_thinking = false;
    
    // Event processing
    bool processEvents(const Board& board, MCTS& mcts, GameRecorder& recorder);
    void handleMouseClick(sf::Vector2i mouse_pos, Board& board, MCTS& mcts, GameRecorder& recorder);
    void handleKeyPress(sf::Keyboard::Key key, Board& board, MCTS& mcts, GameRecorder& recorder);
    
    // AI computation helpers
    folly::coro::Task<void> processAITurn(MCTS& mcts, int samples, GameRecorder& recorder);
    
    // Game logic helpers
    void advanceAction();
    void checkGameOver(const Board& board, GameRecorder& recorder);
    Player getCurrentPlayer() const;
    
    // Rendering
    void render(const Board& board);
    ElementType m_highlight_type = ElementType::NONE;
    int m_highlight_row = -1;
    int m_highlight_col = -1;
};

// Standalone GUI function for interactive play
folly::coro::Task<GameRecorder> interactive_play_gui(Board board, InteractivePlayOptions opts);

} // namespace GUI 
