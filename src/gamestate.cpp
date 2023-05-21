#include "gamestate.hpp"

#include <algorithm>
#include <deque>
#include <exception>
#include <iostream>
#include <ranges>

#include "util.hpp"

namespace ranges = std::ranges;
namespace views = std::ranges::views;

Cell Cell::step(Direction direction) const {
    switch (direction) {
        case Direction::Left:
            return {x - 1, y};
        case Direction::Right:
            return {x + 1, y};
        case Direction::Down:
            return {x, y - 1};
        case Direction::Up:
            return {x, y + 1};
    }

    throw std::runtime_error("Unreachable");
}

Wall Wall::normalize() const {
    if (direction == Direction::Left) {
        return {{cell.x - 1, cell.y}, Direction::Right};
    }

    if (direction == Direction::Down) {
        return {{cell.x, cell.y - 1}, Direction::Up};
    }

    return *this;
}

Turn Turn::next() const {
    if (sub_turn == First) {
        return Turn{player, Second};
    } else {
        return Turn{player == Player::Red ? Player::Blue : Player::Red, First};
    }
}

Turn Turn::prev() const {
    if (sub_turn == Second) {
        return Turn{player, First};
    } else {
        return Turn{player == Player::Red ? Player::Blue : Player::Red, Second};
    }
}

std::ostream& operator<<(std::ostream& out, Direction dir) {
    switch (dir) {
        case Direction::Left:
            out << "Left";
            break;
        case Direction::Right:
            out << "Right";
            break;
        case Direction::Down:
            out << "Down";
            break;
        case Direction::Up:
            out << "Up";
            break;
        default:
            out << "Unknown Direction";
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
            out << "Unknown Player";
    }

    return out;
}

std::ostream& operator<<(std::ostream& out, Cell cell) {
    out << '(' << cell.x << ", " << cell.y << ')';
    return out;
}

std::ostream& operator<<(std::ostream& out, Wall wall) {
    out << '{' << wall.cell << ", " << wall.direction << '}';
    return out;
}

std::ostream& operator<<(std::ostream& out, Move move) {
    std::visit(overload{[&](Direction dir) { out << "Step " << dir; },
                        [&](Wall wall) { out << "Place " << wall; }},
               move);

    return out;
}

std::ostream& operator<<(std::ostream& out, Turn turn) {
    out << turn.player << ":";

    switch (turn.sub_turn) {
        case Turn::First:
            out << "First";
            break;
        case Turn::Second:
            out << "Second";
            break;
        default:
            out << "Unknown Subturn";
    }

    return out;
}

std::istream& operator>>(std::istream& in, Direction& dir) {
    std::string val;
    in >> val;

    if (val == "Left") {
        dir = Direction::Left;
    } else if (val == "Right") {
        dir = Direction::Right;
    } else if (val == "Down") {
        dir = Direction::Down;
    } else if (val == "Up") {
        dir = Direction::Up;
    } else {
        in.fail();
    }

    return in;
}

std::istream& operator>>(std::istream& in, Player& player) {
    std::string val;
    in >> val;

    if (val == "Red") {
        player = Player::Red;
    } else if (val == "Blue") {
        player = Player::Blue;
    } else {
        in.fail();
    }

    return in;
}

std::istream& operator>>(std::istream& in, Cell& cell) {
    char c;
    // Validation who?
    in >> c >> cell.x >> c >> cell.y >> c;
    return in;
}

std::istream& operator>>(std::istream& in, Wall& wall) {
    char c;
    in >> c >> wall.cell >> c >> wall.direction >> c;
    return in;
}

std::istream& operator>>(std::istream& in, Move& move) {
    std::string type;
    in >> type;

    if (type == "Step") {
        Direction dir;
        in >> dir;
        move = dir;
    } else if (type == "Place") {
        Wall wall;
        in >> wall;
        move = wall.normalize();
    } else {
        in.fail();
    }

    return in;
}

std::istream& operator>>(std::istream& in, Turn& turn) {
    char c;
    std::string sub_turn;
    in >> turn.player >> c >> sub_turn;

    if (sub_turn == "First") {
        turn.sub_turn = Turn::First;
    } else if (sub_turn == "Second") {
        turn.sub_turn = Turn::Second;
    } else {
        in.fail();
    }

    return in;
}

Board::Board(int width, int height, Cell red_start, Cell red_goal, Cell blue_start, Cell blue_goal)
    : m_red{red_start, red_goal},
      m_blue{blue_start, blue_goal},
      m_width{width},
      m_height{height},
      m_board(width * height) {
    state_at(red_start).has_red_player = true;
    state_at(blue_start).has_blue_player = true;
    state_at(red_goal).has_red_goal = true;
    state_at(blue_goal).has_blue_goal = true;
}

bool Board::is_blocked(Wall wall) const {
    wall = wall.normalize();

    if (wall.cell.x < 0 || wall.cell.y < 0 || wall.cell.x >= m_width || wall.cell.y >= m_height) {
        return true;
    }

    if (wall.cell.x == m_width - 1 && wall.direction == Direction::Right) {
        return true;
    }

    if (wall.cell.y == m_height - 1 && wall.direction == Direction::Up) {
        return true;
    }

    State const state = state_at(wall.cell);

    if (wall.direction == Direction::Right &&
        (state.has_red_right_wall || state.has_blue_right_wall)) {
        return true;
    }

    if (wall.direction == Direction::Up && (state.has_red_up_wall || state.has_blue_up_wall)) {
        return true;
    }

    return false;
}

std::vector<Direction> Board::legal_directions(Player player) const {
    Cell const pos = player == Player::Red ? m_red.position : m_blue.position;
    auto dirs = directions | views::filter([&](Direction dir) { return !is_blocked({pos, dir}); });
    return {dirs.begin(), dirs.end()};
}

std::pair<bool, int> Board::find_bridges(Cell start, Cell target, int level,
                                         std::vector<int>& levels, std::set<Wall>& bridges) const {
    auto const level_at = [&](Cell square) -> int& { return levels[square.x * m_height + square.y]; };

    level_at(start) = level;
    bool target_found = start == target;
    int min_level = level;

    for (Direction dir : directions) {
        Wall const wall{start, dir};

        if (is_blocked(wall)) {
            continue;
        }

        Cell const neighbor = start.step(dir);
        int const neighbor_level = level_at(neighbor);

        if (neighbor_level == level - 1) {
            continue;
        }

        if (neighbor_level == -1) {
            auto const [sub_found, sub_level] =
                find_bridges(neighbor, target, level + 1, levels, bridges);
            target_found = target_found || sub_found;
            min_level = std::min(min_level, sub_level);

            if (sub_found && sub_level > level) {
                bridges.insert(wall.normalize());
            }
        } else {
            min_level = std::min(min_level, neighbor_level);
        }
    }

    return {target_found, min_level};
}

std::vector<Wall> Board::legal_walls() const {
    std::set<Wall> illegal_walls;
    std::vector<int> levels(m_width * m_height, -1);

    find_bridges(m_blue.position, m_blue.goal, 1, levels, illegal_walls);
    ranges::fill(levels, -1);
    find_bridges(m_red.position, m_red.goal, 1, levels, illegal_walls);

    std::vector<Wall> result;

    for (int x = 0; x < m_width; ++x) {
        for (int y = 0; y < m_height; ++y) {
            for (Direction const dir : {Direction::Right, Direction::Up}) {
                Wall const wall{{x, y}, dir};

                if (!is_blocked(wall) && !illegal_walls.contains(wall)) {
                    result.push_back(wall);
                }
            }
        }
    }

    return result;
}

std::vector<Move> Board::legal_moves(Player player) const {
    // Inefficient but whatever for now
    auto const dirs = legal_directions(player);
    auto const walls = legal_walls();

    std::vector<Move> result;
    result.reserve(dirs.size() + walls.size());
    result.insert(result.end(), dirs.begin(), dirs.end());
    result.insert(result.end(), walls.begin(), walls.end());

    return result;
}

void Board::take_step(Player player, Direction dir) {
    Cell& position = player == Player::Red ? m_red.position : m_blue.position;

    if (is_blocked({position, dir})) {
        std::cout << position.x << ", " << position.y << ": " << int(dir) << '\n';
        throw std::runtime_error("Trying to move through blocked wall!");
    }

    State& state = state_at(position);

    (player == Player::Red ? state.has_red_player : state.has_blue_player) = false;
    position = position.step(dir);

    State& new_state = state_at(position);
    (player == Player::Red ? new_state.has_red_player : new_state.has_blue_player) = true;
}

void Board::place_wall(Player player, Wall wall) {
    wall = wall.normalize();
    State& state = state_at(wall.cell);

    if (player == Player::Red) {
        (wall.direction == Direction::Right ? state.has_red_right_wall : state.has_red_up_wall) =
            true;
    } else {
        (wall.direction == Direction::Right ? state.has_blue_right_wall : state.has_blue_up_wall) =
            true;
    }
}

void Board::do_move(Player player, Move move) {
    std::visit(overload{[&](Direction dir) { take_step(player, dir); },
                        [&](Wall wall) { place_wall(player, wall); }},
               move);
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
    std::vector<bool> visited(m_width * m_height, false);
    std::deque<std::pair<Cell, int>> queue = {{start, 0}};

    auto const visited_at = [&](Cell cell) { return visited[cell.x + cell.y * m_width]; };

    while (!queue.empty()) {
        auto const [top, dist] = queue.front();
        queue.pop_front();

        if (top == target) {
            return dist;
        }

        visited_at(top) = true;

        for (Direction dir : directions) {
            if (is_blocked({top, dir})) {
                continue;
            }

            Cell const neighbor = top.step(dir);

            if (!visited_at(neighbor)) {
                queue.push_back({neighbor, dist + 1});
            }
        }
    }

    return -1;
}

Cell Board::position(Player player) const {
    return player == Player::Red ? m_red.position : m_blue.position;
}

Cell Board::goal(Player player) const {
    return player == Player::Red ? m_red.goal : m_blue.goal;
}

Board::State& Board::state_at(Cell cell) {
    return m_board[cell.x + cell.y * m_width];
}

Board::State Board::state_at(Cell cell) const {
    return m_board[cell.x + cell.y * m_width];
}
