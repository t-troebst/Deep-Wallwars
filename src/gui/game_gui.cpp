#include "game_gui.hpp"
#include <folly/logging/xlog.h>
#include <iostream>
#include <cstdlib>
#include <SFML/System.hpp>

namespace GUI {

GameGUI::GameGUI(int window_width, int window_height, int board_cols, int board_rows)
    : m_layout(window_width, window_height, board_cols, board_rows),
      m_input_handler(m_layout) {
    
    // Create window with minimal OpenGL context and retry logic for WSL2/X11 issues
    sf::ContextSettings settings;
    settings.majorVersion = 2;
    settings.minorVersion = 1;
    settings.depthBits = 0;
    settings.stencilBits = 0;
    settings.antialiasingLevel = 0;
    
    // Try to create window with retry logic for X11/WSL2 issues
    bool window_created = false;
    int retry_count = 0;
    const int max_retries = 3;
    
    while (!window_created && retry_count < max_retries) {
        try {
            if (retry_count > 0) {
                XLOGF(INFO, "Retrying SFML window creation (attempt {})", retry_count + 1);
                // Small delay between retries
                sf::sleep(sf::milliseconds(500));
            }
            
            m_window.create(sf::VideoMode(window_width, window_height), "Deep Wallwars", 
                           sf::Style::Default, settings);
            
            // Test if window was actually created successfully
            if (m_window.isOpen()) {
                window_created = true;
                XLOGF(INFO, "SFML window created successfully");
            } else {
                throw std::runtime_error("Window creation returned success but window is not open");
            }
        } catch (const std::exception& e) {
            retry_count++;
            XLOGF(WARN, "SFML window creation failed (attempt {}): {}", retry_count, e.what());
            
            if (retry_count >= max_retries) {
                throw std::runtime_error("Failed to create SFML window after " + std::to_string(max_retries) + 
                                       " attempts. This may be due to X11/display issues. Try: wsl --shutdown from Windows, then restart WSL.");
            }
        }
    }
    
    // Initialize renderer after successful window creation
    m_renderer = std::make_unique<BoardRenderer>(m_window, m_layout);
    
    m_window.setFramerateLimit(GUI::FPS);
    
    m_human_player = Player::Red;
    m_ai_player = Player::Blue;
}

bool GameGUI::askHumanGoesFirst() {
    std::cout << "Do you want to go first? (y/n): ";
    char first;
    std::cin >> first;
    return (first == 'y' || first == 'Y');
}

folly::coro::Task<GameRecorder> GameGUI::run_interactive_game(Board board, EvaluationFunction ai_model, InteractivePlayOptions opts, bool human_goes_first) {
    m_human_player = human_goes_first ? Player::Red : Player::Blue;
    m_ai_player = other_player(m_human_player);
    
    GameRecorder recorder(board, 
                         human_goes_first ? "Player" : "Deep Wallwars",
                         human_goes_first ? "Deep Wallwars" : "Player");
    
    MCTS mcts{ai_model, std::move(board), 
              {.max_parallelism = opts.max_parallel_samples, .seed = opts.seed}};
    
    m_is_human_turn = human_goes_first;
    m_actions_left = 2;
    m_game_over = false;
    
    // Render initial state immediately
    render(mcts.current_board());
    m_window.display();
    
    while (m_window.isOpen() && !m_game_over) {
        // Handle events (always process events for responsiveness)
        if (!processEvents(mcts.current_board(), mcts, recorder)) {
            break; // Window closed
        }
        
        // Check for game over
        checkGameOver(mcts.current_board(), recorder);
        if (m_game_over) {
            break;
        }
        
        // AI turn logic - process in smaller chunks for responsiveness
        if (!m_is_human_turn && !m_game_over && !m_ai_thinking) {
            m_ai_thinking = true;
            co_await processAITurn(mcts, opts.samples, recorder);
            m_ai_thinking = false;
            if (!m_game_over) {
                m_is_human_turn = true;
                m_actions_left = 2;
            }
        }
        
        // Render everything
        if (m_ai_thinking) {
            m_renderer->renderWithAIThinking(mcts.current_board(), getCurrentPlayer(), m_actions_left,
                                           m_highlight_type, m_highlight_row, m_highlight_col);
        } else {
            render(mcts.current_board());
        }
        m_window.display();
        
        // Small sleep to prevent busy waiting
        sf::sleep(sf::milliseconds(16)); // ~60 FPS
    }
    
    // Safe JSON serialization with error handling
    try {
        XLOGF(INFO, "Game finished, json string: {}", recorder.to_json());
    } catch (const std::exception& e) {
        XLOGF(ERR, "Failed to serialize game to JSON: {}", e.what());
    }
    co_return recorder;
}

bool GameGUI::processEvents(const Board& board, MCTS& mcts, GameRecorder& recorder) {
    sf::Event event;
    while (m_window.pollEvent(event)) {
        switch (event.type) {
            case sf::Event::Closed:
                m_window.close();
                return false;
                
            case sf::Event::MouseButtonPressed:
                if (event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2i mouse_pos(event.mouseButton.x, event.mouseButton.y);
                    // We need a non-const reference for mouse clicks, so we'll get it from mcts
                    Board& mutable_board = const_cast<Board&>(mcts.current_board());
                    handleMouseClick(mouse_pos, mutable_board, mcts, recorder);
                }
                break;
                
            case sf::Event::KeyPressed:
                {
                    // We need a non-const reference for key presses, so we'll get it from mcts  
                    Board& mutable_board = const_cast<Board&>(mcts.current_board());
                    handleKeyPress(event.key.code, mutable_board, mcts, recorder);
                }
                break;
                
            case sf::Event::MouseMoved:
                {
                    // Update highlighting
                    sf::Vector2i mouse_pos(event.mouseMove.x, event.mouseMove.y);
                    auto [element_type, row, col] = m_input_handler.getElementAtPosition(mouse_pos);
                    m_highlight_type = element_type;
                    m_highlight_row = row;
                    m_highlight_col = col;
                }
                break;
                
            default:
                break;
        }
    }
    return true;
}

void GameGUI::handleMouseClick(sf::Vector2i mouse_pos, Board& board, MCTS& mcts, GameRecorder& recorder) {
    if (m_game_over || !m_is_human_turn) {
        return; // Game over or not human's turn
    }
    
    auto action = m_input_handler.handleMouseClick(mouse_pos, board, m_human_player);
    
    switch (action.type) {
        case InputHandler::MouseAction::MOVE_TO_CELL: {
            // Find the direction to move to the target cell
            Cell current_pos = board.position(m_human_player);
            Cell target = action.target_cell;
            
            if (m_input_handler.isCellReachableIn1Action(board, m_human_player, target)) {
                // Find the direction
                auto dirs = board.legal_directions(m_human_player);
                for (Direction dir : dirs) {
                    if (current_pos.step(dir) == target) {
                        mcts.force_action(dir);
                        m_human_actions.push_back(dir);  // Store for move recording
                        advanceAction();
                        // Check if we completed a full move (2 actions)
                        if (m_actions_left == 2) {
                            // Record the complete human move
                            if (m_human_actions.size() == 2) {
                                try {
                                    Move human_move{m_human_actions[0], m_human_actions[1]};
                                    recorder.record_move(m_human_player, human_move);
                                } catch (const std::exception& e) {
                                    XLOGF(WARN, "Failed to record human move: {}", e.what());
                                }
                                m_human_actions.clear();
                            }
                            m_is_human_turn = false; // Switch to AI turn
                        }
                        break;
                    }
                }
            }
            break;
        }
        
        case InputHandler::MouseAction::PLACE_WALL: {
            Wall wall = action.target_wall;
            mcts.force_action(wall);
            m_human_actions.push_back(wall);  // Store for move recording
            advanceAction();
            // Check if we completed a full move (2 actions)
            if (m_actions_left == 2) {
                // Record the complete human move
                if (m_human_actions.size() == 2) {
                    try {
                        Move human_move{m_human_actions[0], m_human_actions[1]};
                        recorder.record_move(m_human_player, human_move);
                    } catch (const std::exception& e) {
                        XLOGF(WARN, "Failed to record human move: {}", e.what());
                    }
                    m_human_actions.clear();
                }
                m_is_human_turn = false; // Switch to AI turn
            }
            break;
        }
        
        default:
            break;
    }
}

void GameGUI::handleKeyPress(sf::Keyboard::Key key, Board& board, MCTS& mcts, GameRecorder& recorder) {
    if (m_game_over) {
        return;
    }
    
    if (key == sf::Keyboard::Escape) {
        m_window.close();
        return;
    }
    
    if (!m_is_human_turn) {
        return; // Not human's turn
    }
    
    auto direction = m_input_handler.handleKeyPress(key);
    if (direction) {
        // Check if move is legal
        auto legal_dirs = board.legal_directions(m_human_player);
        for (Direction dir : legal_dirs) {
            if (dir == *direction) {
                mcts.force_action(dir);
                m_human_actions.push_back(dir);  // Store for move recording
                advanceAction();
                // Check if we completed a full move (2 actions)
                if (m_actions_left == 2) {
                    // Record the complete human move
                    if (m_human_actions.size() == 2) {
                        try {
                            Move human_move{m_human_actions[0], m_human_actions[1]};
                            recorder.record_move(m_human_player, human_move);
                        } catch (const std::exception& e) {
                            XLOGF(WARN, "Failed to record human move: {}", e.what());
                        }
                        m_human_actions.clear();
                    }
                    m_is_human_turn = false; // Switch to AI turn
                }
                break;
            }
        }
    }
}

void GameGUI::advanceAction() {
    m_actions_left--;
    // Note: The actual turn management is handled by MCTS
    // We just track actions left for display purposes
    if (m_actions_left <= 0) {
        m_actions_left = 2; // Reset for next turn
    }
}

void GameGUI::checkGameOver(const Board& board, GameRecorder& recorder) {
    Winner winner = board.winner();
    if (winner != Winner::Undecided) {
        recorder.record_winner(winner);
        m_game_over = true;
        m_winner = winner;
        
        std::string winner_str;
        switch (winner) {
            case Winner::Red:
                winner_str = "Red";
                break;
            case Winner::Blue:
                winner_str = "Blue";
                break;
            case Winner::Draw:
                winner_str = "Draw";
                break;
            default:
                winner_str = "Unknown";
        }
        XLOGF(INFO, "Game Over! Winner: {}", winner_str);
    }
}

Player GameGUI::getCurrentPlayer() const {
    return m_is_human_turn ? m_human_player : m_ai_player;
}

void GameGUI::render(const Board& board) {
    if (!m_is_human_turn && !m_game_over) {
        // Show AI thinking indicator
        m_renderer->renderWithAIThinking(board, getCurrentPlayer(), m_actions_left, 
                                       m_highlight_type, m_highlight_row, m_highlight_col);
    } else {
        // Normal rendering
        m_renderer->render(board, getCurrentPlayer(), m_actions_left, 
                         m_highlight_type, m_highlight_row, m_highlight_col);
    }
}

folly::coro::Task<void> GameGUI::processAITurn(MCTS& mcts, int samples, GameRecorder& recorder) {
    // Deactivate OpenGL context before TensorRT operations to avoid conflicts
    m_window.setActive(false);
    
    try {
        Cell ai_start_cell = mcts.current_board().position(m_ai_player);
        
        // Process AI move with TensorRT (OpenGL context is inactive)
        auto ai_move = co_await mcts.sample_and_commit_to_move(samples);
        
        // Reactivate OpenGL context for rendering
        m_window.setActive(true);
        
        if (ai_move) {
            recorder.record_move(m_ai_player, *ai_move);
            
            // Safe logging with error handling
            try {
                XLOGF(INFO, "AI played: {}", ai_move->standard_notation(ai_start_cell));
            } catch (const std::exception& e) {
                XLOGF(WARN, "AI played a move but couldn't convert to standard notation: {}", e.what());
            }
            
            checkGameOver(mcts.current_board(), recorder);
        } else {
            // AI has no moves, human wins
            XLOGF(INFO, "AI has no legal moves, human wins");
            recorder.record_winner(winner_from_player(m_human_player));
            m_game_over = true;
        }
    } catch (const std::exception& e) {
        // Ensure OpenGL context is reactivated even on error
        m_window.setActive(true);
        
        XLOGF(ERR, "AI computation failed: {}", e.what());
        // Treat as AI having no moves
        recorder.record_winner(winner_from_player(m_human_player));
        m_game_over = true;
    }
}

// Standalone GUI function for interactive play
folly::coro::Task<GameRecorder> interactive_play_gui(Board board, InteractivePlayOptions opts) {
    // Ask for player choice before creating GUI window
    bool human_goes_first = GameGUI::askHumanGoesFirst();
    
    // Limit parallelism for GUI mode to prevent threading issues
    int safe_parallelism = std::min(opts.max_parallel_samples, 4);
    if (safe_parallelism != opts.max_parallel_samples) {
        std::cout << "Limiting parallelism to " << safe_parallelism << " for GUI mode (was " << opts.max_parallel_samples << ")\n";
        opts.max_parallel_samples = safe_parallelism;
    }
    
    GameGUI gui(GUI::DEFAULT_WINDOW_WIDTH, GUI::DEFAULT_WINDOW_HEIGHT, 
                board.columns(), board.rows());
    co_return co_await gui.run_interactive_game(std::move(board), opts.model, opts, human_goes_first);
}

} // namespace GUI 
