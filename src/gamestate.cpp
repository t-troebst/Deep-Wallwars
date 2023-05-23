#include "gamestate.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <exception>
#include <format>
#include <iostream>
#include <ranges>
#include <sstream>

#include "util.hpp"

namespace ranges = std::ranges;
namespace views = std::ranges::views;

static constexpr std::array<char, 13> kColumnLabels = {'a', 'b', 'c', 'd', 'e', 'f', 'g',
                                                       'h', 'i', 'j', 'k', 'l', 'm'};
static constexpr std::array<char, 10> kRowLabels = {'1', '2', '3', '4', '5',
                                                    '6', '7', '8', '9', 'X'};

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
            return {column, row + 1};
        case Direction::Down:
            return {column + 1, row};
        case Direction::Left:
            return {column, row - 1};
        case Direction::Up:
            return {column - 1, row};
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
            cell = {c.column, c.row - 1};
            type = Right;
            return;
        case Direction::Up:
            cell = {c.column - 1, c.row};
            type = Down;
            return;
    }

    throw std::runtime_error("Unreachable: invalid direction (wall)!");
}

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
    std::visit(overload{[&](Direction dir) {
                            Cell cell = start.step(dir);
                            std::visit(overload{[&](Direction dir2) { out << cell.step(dir2); },
                                                [&](Wall wall) { out << cell << ' ' << wall; }},
                                       second);
                        },
                        [&](Wall wall) {
                            std::visit(overload{[&](Direction dir) {
                                                    out << start.step(dir) << ' ' << wall;
                                                },
                                                [&](Wall wall2) {
                                                    if (wall < wall2) {
                                                        out << wall << ' ' << wall2;
                                                    } else {
                                                        out << wall2 << ' ' << wall;
                                                    }
                                                }},
                                       second);
                        }},
               first);

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
    cell.row = row_label == 'X' ? 9 : row_label - '0';

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
            {red_start.column, rows - 1 - red_start.row},
            {red_goal.column, rows - 1 - red_goal.row}} {}

Board::Board(int columns, int rows) : Board{columns, rows, {0, 0}, {columns - 1, rows - 1}} {}

bool Board::is_blocked(Wall wall) const {
    if (wall.cell.column < 0 || wall.cell.row < 0 || wall.cell.column >= m_columns ||
        wall.cell.row >= m_rows) {
        return true;
    }

    if (wall.type == Wall::Down) {
        if (wall.cell.column == m_columns - 1) {
            return true;
        }

        State const state = state_at(wall.cell);

        if (state.has_red_down_wall || state.has_blue_down_wall) {
            return true;
        }
    } else {
        if (wall.cell.row == m_rows - 1) {
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

std::pair<bool, int> Board::find_bridges(Cell start, Cell target, int level,
                                         std::vector<int>& levels, std::set<Wall>& bridges) const {
    levels[index_from_cell(start)] = level;
    bool target_found = start == target;
    int min_level = level;

    for (Direction dir : kDirections) {
        Wall const wall{start, dir};

        if (is_blocked(wall)) {
            continue;
        }

        Cell const neighbor = start.step(dir);
        int const neighbor_level = levels[index_from_cell(neighbor)];

        if (neighbor_level == level - 1) {
            continue;
        }

        if (neighbor_level == -1) {
            auto const [sub_found, sub_level] =
                find_bridges(neighbor, target, level + 1, levels, bridges);
            target_found = target_found || sub_found;
            min_level = std::min(min_level, sub_level);

            if (sub_found && sub_level > level) {
                bridges.insert(wall);
            }
        } else {
            min_level = std::min(min_level, neighbor_level);
        }
    }

    return {target_found, min_level};
}

std::vector<Wall> Board::legal_walls() const {
    std::set<Wall> illegal_walls;
    std::vector<int> levels(m_columns * m_rows, -1);

    find_bridges(m_blue.position, m_blue.goal, 1, levels, illegal_walls);
    ranges::fill(levels, -1);
    find_bridges(m_red.position, m_red.goal, 1, levels, illegal_walls);

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

    (player == Player::Red ? state.has_red_player : state.has_blue_player) = false;
    position = position.step(dir);

    State& new_state = state_at(position);
    (player == Player::Red ? new_state.has_red_player : new_state.has_blue_player) = true;
}

void Board::place_wall(Player player, Wall wall) {
    State& state = state_at(wall.cell);

    if (is_blocked(wall)) {
        throw std::runtime_error("Trying to place on top of existing wall!");
    }

    // TODO: should at least add a debug check for disconnecting players from their goals?

    if (player == Player::Red) {
        (wall.type == Wall::Right ? state.has_red_right_wall : state.has_red_down_wall) = true;
    } else {
        (wall.type == Wall::Right ? state.has_blue_right_wall : state.has_blue_down_wall) = true;
    }
}

void Board::do_action(Player player, Action action) {
    std::visit(overload{[&](Direction dir) { take_step(player, dir); },
                        [&](Wall wall) { place_wall(player, wall); }},
               action);
}

std::optional<Player> Board::winner() const {
    if (m_red.position == m_red.goal) {
        return Player::Red;
    }

    if (m_blue.position == m_blue.goal) {
        return Player::Blue;
    }

    return {};
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
