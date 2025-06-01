#include "game_gui.hpp"

#include <folly/experimental/coro/BlockingWait.h>
#include <folly/logging/xlog.h>

#include <SFML/System.hpp>
#include <cstdlib>
#include <iostream>

namespace GUI {

GameGUI::GameGUI(int window_width, int window_height, int board_cols, int board_rows)
    : m_layout(window_width, window_height, board_cols, board_rows), m_input_handler(m_layout) {
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
    int const max_retries = 3;

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
        } catch (std::exception const& e) {
            retry_count++;
            XLOGF(WARN, "SFML window creation failed (attempt {}): {}", retry_count, e.what());

            if (retry_count >= max_retries) {
                throw std::runtime_error("Failed to create SFML window after " +
                                         std::to_string(max_retries) +
                                         " attempts. This may be due to X11/display issues. Try: "
                                         "wsl --shutdown from Windows, then restart WSL.");
            }
        }
    }

    // Initialize renderer after successful window creation
    m_renderer = std::make_unique<BoardRenderer>(m_window, m_layout);

    m_window.setFramerateLimit(GUI::FPS);

    m_human_player = Player::Red;
    m_ai_player = Player::Blue;
}

bool GameGUI::ask_human_goes_first() {
    std::cout << "Do you want to go first? (y/n): ";
    char first;
    std::cin >> first;
    return (first == 'y' || first == 'Y');
}

GameRecorder GameGUI::run_interactive_game(Board board, EvaluationFunction ai_model,
                                           InteractivePlayOptions opts, bool human_goes_first,
                                           folly::CPUThreadPoolExecutor& thread_pool) {
    // Store thread pool reference for AI work
    m_thread_pool = &thread_pool;

    m_human_player = human_goes_first ? Player::Red : Player::Blue;
    m_ai_player = other_player(m_human_player);

    GameRecorder recorder(board, human_goes_first ? "Player" : "Deep Wallwars",
                          human_goes_first ? "Deep Wallwars" : "Player");

    MCTS mcts{ai_model, board, {.max_parallelism = opts.max_parallel_samples, .seed = opts.seed}};

    m_game_state = human_goes_first ? GameState::HumanTurn : GameState::AITurn;
    m_actions_left = 2;

    // Render initial state immediately
    render(board, mcts, opts.samples);
    m_window.display();

    // Main rendering loop - stays on main thread
    while (m_window.isOpen() && m_game_state != GameState::GameOver) {
        // Handle events (always process events for responsiveness)
        if (!process_events(board, mcts, recorder)) {
            break;  // Window closed
        }

        // Check for game over
        check_game_over(board, recorder);
        if (m_game_state == GameState::GameOver) {
            break;
        }

        // AI turn logic
        if (m_game_state != GameState::HumanTurn) {
            process_ai_turn(mcts, board, opts.samples, recorder);
        }

        // Render everything - always on main thread
        render(board, mcts, opts.samples);
        m_window.display();
    }

    // Safe JSON serialization with error handling
    try {
        XLOGF(INFO, "Game finished, json string: {}", recorder.to_json());
    } catch (std::exception const& e) {
        XLOGF(ERR, "Failed to serialize game to JSON: {}", e.what());
    }
    return recorder;
}

bool GameGUI::process_events(Board& board, MCTS& mcts, GameRecorder& recorder) {
    sf::Event event;
    while (m_window.pollEvent(event)) {
        switch (event.type) {
            case sf::Event::Closed:
                m_window.close();
                return false;

            case sf::Event::MouseButtonPressed:
                if (event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2i mouse_pos(event.mouseButton.x, event.mouseButton.y);
                    handle_mouse_click(mouse_pos, board, mcts, recorder);
                }
                break;

            case sf::Event::KeyPressed: {
                handle_key_press(event.key.code, board, mcts, recorder);
            } break;

            case sf::Event::MouseMoved: {
                // Update highlighting
                sf::Vector2i mouse_pos(event.mouseMove.x, event.mouseMove.y);
                auto [element_type, row, col] = m_input_handler.get_element_at_position(mouse_pos);
                m_highlight_type = element_type;
                m_highlight_row = row;
                m_highlight_col = col;
            } break;

            default:
                break;
        }
    }
    return true;
}

void GameGUI::handle_mouse_click(sf::Vector2i mouse_pos, Board& board, MCTS& mcts,
                                 GameRecorder& recorder) {
    if (m_game_state != GameState::HumanTurn) {
        return;  // Game over or not human's turn
    }

    auto action = m_input_handler.handle_mouse_click(mouse_pos, board, m_human_player);

    switch (action.type) {
        case InputHandler::MouseAction::MOVE_TO_CELL: {
            // Find the direction to move to the target cell
            Cell current_pos = board.position(m_human_player);
            Cell target = action.target_cell;

            if (m_input_handler.is_cell_reachable_in_1_action(board, m_human_player, target)) {
                // Find the direction
                auto dirs = board.legal_directions(m_human_player);
                for (Direction dir : dirs) {
                    if (current_pos.step(dir) == target) {
                        mcts.force_action(dir);
                        m_human_actions.push_back(dir);  // Store for move recording
                        board.do_action(m_human_player, dir);
                        advance_action();
                        // Check if we completed a full move (2 actions)
                        if (m_actions_left == 2) {
                            // Record the complete human move
                            if (m_human_actions.size() == 2) {
                                try {
                                    Move human_move{m_human_actions[0], m_human_actions[1]};
                                    recorder.record_move(m_human_player, human_move);
                                } catch (std::exception const& e) {
                                    XLOGF(WARN, "Failed to record human move: {}", e.what());
                                }
                                m_human_actions.clear();
                            }
                            m_game_state = GameState::AITurn;
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
            board.do_action(m_human_player, wall);
            m_human_actions.push_back(wall);  // Store for move recording
            advance_action();
            // Check if we completed a full move (2 actions)
            if (m_actions_left == 2) {
                // Record the complete human move
                if (m_human_actions.size() == 2) {
                    try {
                        Move human_move{m_human_actions[0], m_human_actions[1]};
                        recorder.record_move(m_human_player, human_move);
                    } catch (std::exception const& e) {
                        XLOGF(WARN, "Failed to record human move: {}", e.what());
                    }
                    m_human_actions.clear();
                }
                m_game_state = GameState::AITurn;
            }
            break;
        }

        default:
            break;
    }
}

void GameGUI::handle_key_press(sf::Keyboard::Key key, Board& board, MCTS& mcts,
                               GameRecorder& recorder) {
    if (m_game_state != GameState::HumanTurn) {
        return;
    }

    if (key == sf::Keyboard::Escape) {
        m_window.close();
        return;
    }

    auto direction = m_input_handler.handle_key_press(key);
    if (direction) {
        // Check if move is legal
        auto legal_dirs = board.legal_directions(m_human_player);
        for (Direction dir : legal_dirs) {
            if (dir == *direction) {
                mcts.force_action(dir);
                board.do_action(m_human_player, dir);
                m_human_actions.push_back(dir);  // Store for move recording
                advance_action();
                // Check if we completed a full move (2 actions)
                if (m_actions_left == 2) {
                    // Record the complete human move
                    if (m_human_actions.size() == 2) {
                        try {
                            Move human_move{m_human_actions[0], m_human_actions[1]};
                            recorder.record_move(m_human_player, human_move);
                        } catch (std::exception const& e) {
                            XLOGF(WARN, "Failed to record human move: {}", e.what());
                        }
                        m_human_actions.clear();
                    }
                    m_game_state = GameState::AITurn;
                }
                break;
            }
        }
    }
}

void GameGUI::advance_action() {
    m_actions_left--;
    // Note: The actual turn management is handled by MCTS
    // We just track actions left for display purposes
    if (m_actions_left <= 0) {
        m_actions_left = 2;  // Reset for next turn
    }
}

void GameGUI::check_game_over(Board const& board, GameRecorder& recorder) {
    Winner winner = board.winner();
    if (winner != Winner::Undecided) {
        recorder.record_winner(winner);
        m_game_state = GameState::GameOver;

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

void GameGUI::render(Board const& board, MCTS const& mcts, int total_samples) {
    m_renderer->render(board, m_game_state == GameState::HumanTurn ? m_human_player : m_ai_player,
                       m_actions_left, m_score_for_human, m_highlight_type, m_highlight_row,
                       m_highlight_col);

    if (m_game_state == GameState::WaitingForAI) {
        m_renderer->draw_ai_thinking_indicator(mcts.samples_done(), total_samples);
    }
}

void GameGUI::process_ai_turn(MCTS& mcts, Board& board, int samples, GameRecorder& recorder) {
    if (m_game_state == GameState::AITurn) {
        m_ai_move = mcts.sample_and_commit_to_move(samples).scheduleOn(m_thread_pool).start();
        m_game_state = GameState::WaitingForAI;
        return;
    }

    assert(m_game_state == GameState::WaitingForAI);
    if (!m_ai_move.isReady()) {
        return;
    }

    auto ai_move = m_ai_move.value();

    if (!ai_move) {
        // AI has no moves, human wins
        XLOGF(INFO, "AI has no legal moves, human wins");
        recorder.record_winner(winner_from_player(m_human_player));
        m_game_state = GameState::GameOver;
        return;
    }

    // Record the move
    recorder.record_move(m_ai_player, *ai_move);
    board.do_action(m_ai_player, ai_move->first);
    board.do_action(m_ai_player, ai_move->second);
    m_game_state = GameState::HumanTurn;
    m_score_for_human = mcts.root_value();
    m_actions_left = 2;
    check_game_over(mcts.current_board(), recorder);
}

// Standalone GUI function for interactive play
GameRecorder interactive_play_gui(Board board, InteractivePlayOptions opts,
                                  folly::CPUThreadPoolExecutor& thread_pool) {
    // Ask for player choice before creating GUI window
    bool human_goes_first = GameGUI::ask_human_goes_first();

    GameGUI gui(GUI::DEFAULT_WINDOW_WIDTH, GUI::DEFAULT_WINDOW_HEIGHT, board.columns(),
                board.rows());
    return gui.run_interactive_game(std::move(board), opts.model, opts, human_goes_first,
                                    thread_pool);
}

}  // namespace GUI
