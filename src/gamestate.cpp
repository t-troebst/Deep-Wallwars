#include "gamestate.hpp"

#include <folly/Hash.h>
#include <folly/Overload.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <format>
#include <iostream>
#include <ranges>
#include <sstream>

namespace ranges = std::ranges;
namespace views = std::ranges::views;

static constexpr std::array<char, 13> kColumnLabels = {'a', 'b', 'c', 'd', 'e', 'f', 'g',
                                                       'h', 'i', 'j', 'k', 'l', 'm'};
static constexpr std::array<char, 10> kRowLabels = {'1', '2', '3', '4', '5',
                                                    '6', '7', '8', '9', 'X'};

Direction flip_horizontal(Direction dir) {
    switch (dir) {
        case Direction::Right:
            return Direction::Left;
        case Direction::Left:
            return Direction::Right;
        case Direction::Down:
        case Direction::Up:
            return dir;
    }

    throw std::runtime_error("Unreachable: invalid direction (flip)!");
}

Winner winner_from_player(Player player) {
    switch (player) {
        case Player::Red:
            return Winner::Red;
        case Player::Blue:
            return Winner::Blue;
    }

    throw std::runtime_error("Unreachable: invalid player!");
}

Player other_player(Player player) {
    switch (player) {
        case Player::Red:
            return Player::Blue;
        case Player::Blue:
            return Player::Red;
    }

    throw std::runtime_error("Unreachable: invalid player!");
}

Cell Cell::step(Direction direction) const {
    switch (direction) {
        case Direction::Right:
            return {column + 1, row};
        case Direction::Down:
            return {column, row + 1};
        case Direction::Left:
            return {column - 1, row};
        case Direction::Up:
            return {column, row - 1};
    }

    throw std::runtime_error("Unreachable: invalid direction (step)!");
}

Wall::Wall(Cell cell, Type type) : cell{cell}, type{type} {}

Wall::Wall(Cell c, Direction dir) {
    switch (dir) {
        case Direction::Right:
            cell = c;
            type = Right;
            return;
        case Direction::Down:
            cell = c;
            type = Down;
            return;
        case Direction::Left:
            cell = {c.column - 1, c.row};
            type = Right;
            return;
        case Direction::Up:
            cell = {c.column, c.row - 1};
            type = Down;
            return;
    }

    throw std::runtime_error("Unreachable: invalid direction (wall)!");
}

Direction Wall::direction() const {
    return type == Wall::Down ? Direction::Down : Direction::Right;
}

namespace std {
std::uint64_t hash<Cell>::operator()(Cell cell) const {
    return folly::hash::hash_combine(cell.column, cell.row);
}

std::uint64_t hash<Wall>::operator()(Wall wall) const {
    return folly::hash::hash_combine(wall.cell, wall.type);
}
}  // namespace std

Turn Turn::next() const {
    if (action == First) {
        return Turn{player, Second};
    } else {
        return Turn{player == Player::Red ? Player::Blue : Player::Red, First};
    }
}

Turn Turn::prev() const {
    if (action == Second) {
        return Turn{player, First};
    } else {
        return Turn{player == Player::Red ? Player::Blue : Player::Red, Second};
    }
}

std::string Move::standard_notation(Cell start) const {
    std::stringstream out;
    // Whew, what ugly formatting by clang-format...
    folly::variant_match(
        first,
        [&](Direction dir) {
            Cell cell = start.step(dir);

            folly::variant_match(
                second, [&](Direction dir2) { out << cell.step(dir2); },
                [&](Wall wall) { out << cell << ' ' << wall; });
        },
        [&](Wall wall) {
            folly::variant_match(
                second, [&](Direction dir) { out << start.step(dir) << ' ' << wall; },
                [&](Wall wall2) {
                    if (wall < wall2) {
                        out << wall << ' ' << wall2;
                    } else {
                        out << wall2 << ' ' << wall;
                    }
                });
        });

    return out.str();
}

std::ostream& operator<<(std::ostream& out, Direction dir) {
    switch (dir) {
        case Direction::Right:
            out << "Right";
            break;
        case Direction::Down:
            out << "Down";
            break;
        case Direction::Left:
            out << "Left";
            break;
        case Direction::Up:
            out << "Up";
            break;
        default:
            out << "??";
    }

    return out;
}

std::ostream& operator<<(std::ostream& out, Player player) {
    switch (player) {
        case Player::Red:
            out << "Red";
            break;
        case Player::Blue:
            out << "Blue";
            break;
        default:
            out << "??";
    }

    return out;
}

std::ostream& operator<<(std::ostream& out, Cell cell) {
    if (cell.row < 0 || cell.row >= int(kRowLabels.size()) || cell.column < 0 ||
        cell.column >= int(kColumnLabels.size())) {
        throw std::runtime_error(std::format(
            "Cell coordinates ({}, {}) cannot be expressed as standard notation:", cell.column,
            cell.row));
    }

    out << kColumnLabels[cell.column] << kRowLabels[cell.row];

    return out;
}

std::ostream& operator<<(std::ostream& out, Wall wall) {
    out << wall.cell << (wall.type == Wall::Right ? '>' : 'v');
    return out;
}

std::ostream& operator<<(std::ostream& out, Action const& action) {
    std::visit([&](auto const& action) { out << action; }, action);
    return out;
}

std::ostream& operator<<(std::ostream& out, Move const& move) {
    out << move.first << ' ' << move.second;
    return out;
}

std::ostream& operator<<(std::ostream& out, Turn turn) {
    out << turn.player << ":";

    switch (turn.action) {
        case Turn::First:
            out << "First";
            break;
        case Turn::Second:
            out << "Second";
            break;
        default:
            out << "??";
    }

    return out;
}

std::istream& operator>>(std::istream& in, Cell& cell) {
    char column_label;
    char row_label;
    in >> column_label >> row_label;

    cell.column = column_label - 'a';
    cell.row = row_label == 'X' ? 9 : row_label - '1';

    // TODO: validate

    return in;
}

std::istream& operator>>(std::istream& in, Wall& wall) {
    char dir;
    in >> wall.cell >> dir;

    switch (dir) {
        case 'v':
            wall.type = Wall::Down;
            break;
        case '>':
            wall.type = Wall::Right;
            break;
        default:
            throw std::runtime_error("Invalid wall direction!");
    }

    return in;
}

std::istream& operator>>(std::istream& in, Direction& dir) {
    std::string direction;
    in >> direction;

    if (direction == "right") {
        dir = Direction::Right;
    } else if (direction == "left") {
        dir = Direction::Left;
    } else if (direction == "down") {
        dir = Direction::Down;
    } else if (direction == "up") {
        dir = Direction::Up;
    } else {
        throw std::runtime_error("Invalid direction!");
    }

    return in;
}

Board::Board(int columns, int rows, Cell red_start, Cell red_goal, Cell blue_start, Cell blue_goal)
    : m_red{red_start, red_goal},
      m_blue{blue_start, blue_goal},
      m_columns{columns},
      m_rows{rows},
      m_board(columns * rows) {
    state_at(red_start).has_red_player = true;
    state_at(blue_start).has_blue_player = true;
    state_at(red_goal).has_red_goal = true;
    state_at(blue_goal).has_blue_goal = true;
}

Board::Board(int columns, int rows, Cell red_start, Cell red_goal)
    : Board{columns,
            rows,
            red_start,
            red_goal,
            {columns - 1 - red_start.column, red_start.row},
            {columns - 1 - red_goal.column, red_goal.row}} {}

Board::Board(int columns, int rows) : Board{columns, rows, {0, 0}, {columns - 1, rows - 1}} {}

bool Board::is_blocked(Wall wall) const {
    if (wall.cell.column < 0 || wall.cell.row < 0 || wall.cell.column >= m_columns ||
        wall.cell.row >= m_rows) {
        return true;
    }

    if (wall.type == Wall::Down) {
        if (wall.cell.row == m_rows - 1) {
            return true;
        }

        State const state = state_at(wall.cell);

        if (state.has_red_down_wall || state.has_blue_down_wall) {
            return true;
        }
    } else {
        if (wall.cell.column == m_columns - 1) {
            return true;
        }

        State const state = state_at(wall.cell);

        if (state.has_red_right_wall || state.has_blue_right_wall) {
            return true;
        }
    }

    return false;
}

std::vector<Direction> Board::legal_directions(Player player) const {
    Cell const pos = player == Player::Red ? m_red.position : m_blue.position;
    auto dirs = kDirections | views::filter([&](Direction dir) { return !is_blocked({pos, dir}); });
    return {dirs.begin(), dirs.end()};
}

void Board::find_bridges(Cell start, Cell target, std::vector<int>& levels, std::set<Wall>& bridges,
                         std::vector<Board::StackFrame>& stack) const {
    // Initialize start cell
    levels[index_from_cell(start)] = 1;
    stack[0] = {start, 1, 0, start == target, 1};
    int stack_size = 1;

    while (stack_size > 0) {
        auto& frame = stack[stack_size - 1];

        // Look for unprocessed neighbors starting from current dir_index
        bool found_unprocessed = false;
        for (int dir_idx = frame.dir_index; dir_idx < 4; ++dir_idx) {
            Direction dir = kDirections[dir_idx];
            Wall wall{frame.cell, dir};

            if (is_blocked(wall))
                continue;

            Cell neighbor = frame.cell.step(dir);
            int neighbor_level = levels[index_from_cell(neighbor)];

            if (neighbor_level == frame.level - 1)
                continue;  // parent

            if (neighbor_level == -1) {
                // Found unprocessed neighbor - add to stack
                levels[index_from_cell(neighbor)] = frame.level + 1;
                frame.dir_index = dir_idx + 1;  // Resume from next direction when we return
                stack[stack_size++] = {neighbor, frame.level + 1, 0, neighbor == target,
                                       frame.level + 1};
                found_unprocessed = true;
                break;
            } else {
                // Already visited - update min_level
                frame.min_level = std::min(frame.min_level, neighbor_level);
            }
        }

        if (!found_unprocessed) {
            // All neighbors processed - do postprocessing and pop
            stack_size--;

            if (stack_size > 0) {
                auto& parent = stack[stack_size - 1];
                parent.target_found = parent.target_found || frame.target_found;
                parent.min_level = std::min(parent.min_level, frame.min_level);

                // Check bridge condition
                if (frame.target_found && frame.min_level > parent.level) {
                    Direction dir = kDirections[parent.dir_index - 1];
                    bridges.insert({parent.cell, dir});
                }
            }
        }
    }
}

std::vector<Wall> Board::legal_walls() const {
    std::set<Wall> illegal_walls;
    std::vector<int> levels(m_columns * m_rows, -1);
    std::vector<StackFrame> stack(m_columns * m_rows);
    find_bridges(m_blue.position, m_blue.goal, levels, illegal_walls, stack);
    ranges::fill(levels, -1);
    find_bridges(m_red.position, m_red.goal, levels, illegal_walls, stack);

    std::vector<Wall> result;

    for (int column = 0; column < m_columns; ++column) {
        for (int row = 0; row < m_rows; ++row) {
            for (Wall::Type type : {Wall::Down, Wall::Right}) {
                Wall const wall{{column, row}, type};

                if (!is_blocked(wall) && !illegal_walls.contains(wall)) {
                    result.push_back(wall);
                }
            }
        }
    }

    return result;
}

std::vector<Action> Board::legal_actions(Player player) const {
    // Inefficient but whatever for now
    auto const dirs = legal_directions(player);
    auto const walls = legal_walls();

    std::vector<Action> result;
    result.reserve(dirs.size() + walls.size());
    result.insert(result.end(), dirs.begin(), dirs.end());
    result.insert(result.end(), walls.begin(), walls.end());

    return result;
}

void Board::take_step(Player player, Direction dir) {
    Cell& position = player == Player::Red ? m_red.position : m_blue.position;

    if (is_blocked({position, dir})) {
        throw std::runtime_error("Trying to move through blocked wall!");
    }

    State& state = state_at(position);

    if (player == Player::Red) {
        state.has_red_player = false;
        position = position.step(dir);
        state_at(position).has_red_player = true;
    } else {
        state.has_blue_player = false;
        position = position.step(dir);
        state_at(position).has_blue_player = true;
    }
}

void Board::place_wall(Player player, Wall wall) {
    State& state = state_at(wall.cell);

    if (is_blocked(wall)) {
        throw std::runtime_error("Trying to place on top of existing wall!");
    }

    // TODO: should at least add a debug check for disconnecting players from their goals?

    if (player == Player::Red) {
        if (wall.type == Wall::Right) {
            state.has_red_right_wall = true;
        } else {
            state.has_red_down_wall = true;
        }
    } else {
        if (wall.type == Wall::Right) {
            state.has_blue_right_wall = true;
        } else {
            state.has_blue_down_wall = true;
        }
    }
}

void Board::do_action(Player player, Action action) {
    folly::variant_match(
        action, [&](Direction dir) { take_step(player, dir); },
        [&](Wall wall) { place_wall(player, wall); });
}

Winner Board::winner() const {
    if (m_red.position == m_red.goal) {
        if (distance(m_blue.position, m_blue.goal) <= 2) {
            return Winner::Draw;
        }
        return Winner::Red;
    }

    if (m_blue.position == m_blue.goal) {
        return Winner::Blue;
    }

    return Winner::Undecided;
}

double Board::score_for(Player player) const {
    Winner current_winner = winner();

    if (current_winner == Winner::Draw) {
        return 0.0;
    }

    if (current_winner == Winner::Red) {
        return player == Player::Red ? 1.0 : -1.0;
    }

    if (current_winner == Winner::Blue) {
        return player == Player::Blue ? 1.0 : -1.0;
    }

    double dist = distance(position(player), goal(player));
    Player opponent = other_player(player);
    double opponent_dist = distance(position(opponent), goal(opponent));

    return dist < opponent_dist ? 1.0 - dist / opponent_dist : -1.0 + opponent_dist / dist;
}

int Board::distance(Cell start, Cell target) const {
    std::vector<bool> visited(m_columns * m_rows, false);
    std::deque<std::pair<Cell, int>> queue = {{start, 0}};

    while (!queue.empty()) {
        auto const [top, dist] = queue.front();
        queue.pop_front();

        if (top == target) {
            return dist;
        }

        visited[index_from_cell(top)] = true;

        for (Direction dir : kDirections) {
            if (is_blocked({top, dir})) {
                continue;
            }

            Cell const neighbor = top.step(dir);

            if (!visited[index_from_cell(neighbor)]) {
                queue.push_back({neighbor, dist + 1});
            }
        }
    }

    return -1;
}

void Board::fill_relative_distances(Cell start, std::span<float> dists) const {
    if (int(dists.size()) != m_columns * m_rows) {
        throw std::runtime_error("dists size does not match!");
    }

    std::ranges::fill(dists, 1.0f);

    std::deque<std::pair<Cell, int>> queue = {{start, 0}};

    while (!queue.empty()) {
        auto const [top, dist] = queue.front();
        queue.pop_front();

        dists[index_from_cell(top)] = float(dist) / (m_columns * m_rows);

        for (Direction dir : kDirections) {
            if (is_blocked({top, dir})) {
                continue;
            }

            Cell const neighbor = top.step(dir);

            if (dists[index_from_cell(neighbor)] == 1.0f) {
                queue.push_back({neighbor, dist + 1});
            }
        }
    }
}

std::vector<std::array<bool, 4>> Board::blocked_directions() const {
    std::vector<std::array<bool, 4>> result(m_columns * m_rows);

    for (int i = 0; i < m_columns * m_rows; ++i) {
        Cell cell = cell_at_index(i);

        for (Direction dir : kDirections) {
            result[i][int(dir)] = is_blocked({cell, dir});
        }
    }

    return result;
}

void Board::fill_relative_distances(Cell start, std::span<float> dists,
                                    std::vector<std::array<bool, 4>> const& blocked_dirs) const {
    if (int(dists.size()) != m_columns * m_rows) {
        throw std::runtime_error("dists size does not match!");
    }

    std::ranges::fill(dists, 1.0f);

    std::deque<std::pair<Cell, int>> queue = {{start, 0}};

    while (!queue.empty()) {
        auto const [top, dist] = queue.front();
        queue.pop_front();

        int i = index_from_cell(top);

        dists[i] = float(dist) / (m_columns * m_rows);

        for (Direction dir : kDirections) {
            if (blocked_dirs[i][int(dir)]) {
                continue;
            }

            Cell const neighbor = top.step(dir);

            if (dists[index_from_cell(neighbor)] == 1.0f) {
                queue.push_back({neighbor, dist + 1});
            }
        }
    }
}

Cell Board::cell_at_index(int i) const {
    return {i / m_rows, i % m_rows};
}

int Board::index_from_cell(Cell cell) const {
    return cell.column * m_rows + cell.row;
}

Cell Board::position(Player player) const {
    return player == Player::Red ? m_red.position : m_blue.position;
}

Cell Board::goal(Player player) const {
    return player == Player::Red ? m_red.goal : m_blue.goal;
}

int Board::columns() const {
    return m_columns;
}

int Board::rows() const {
    return m_rows;
}

Board::State& Board::state_at(Cell cell) {
    return m_board[index_from_cell(cell)];
}

Board::State Board::state_at(Cell cell) const {
    return m_board[index_from_cell(cell)];
}

Cell Board::flip_horizontal(Cell cell) const {
    return {m_columns - 1 - cell.column, cell.row};
}

Wall Board::flip_horizontal(Wall wall) const {
    return Wall{flip_horizontal(wall.cell), ::flip_horizontal(wall.direction())};
}

std::uint64_t std::hash<Board>::operator()(Board const& board) const {
    std::uint64_t position_hash =
        folly::hash::hash_combine(board.position(Player::Red), board.position(Player::Blue),
                                  board.goal(Player::Red), board.goal(Player::Blue));

    return folly::hash::hash_range(
        board.m_board.begin(), board.m_board.end(), position_hash,
        [](Board::State state) { return std::bit_cast<std::uint8_t>(state); });
}
