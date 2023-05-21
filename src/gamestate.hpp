#pragma once

#include <array>
#include <compare>
#include <iosfwd>
#include <optional>
#include <set>
#include <variant>
#include <vector>

enum class Direction { Left, Right, Down, Up };

constexpr std::array<Direction, 4> directions = {Direction::Left, Direction::Right, Direction::Down,
                                                 Direction::Up};

enum class Player { Red, Blue };

struct Cell {
    int x;
    int y;

    std::strong_ordering operator<=>(Cell const& other) const = default;
    bool operator==(Cell const& other) const = default;

    [[nodiscard]] Cell step(Direction direction) const;
};

struct Wall {
    Cell cell;
    Direction direction;

    std::strong_ordering operator<=>(Wall const& other) const = default;
    bool operator==(Wall const& other) const = default;

    [[nodiscard]] Wall normalize() const;
};

using Move = std::variant<Direction, Wall>;

struct Turn {
    Player player;
    enum { First, Second } sub_turn;

    bool operator==(Turn const& other) const = default;

    [[nodiscard]] Turn next() const;
    [[nodiscard]] Turn prev() const;
};

std::ostream& operator<<(std::ostream& out, Direction dir);
std::ostream& operator<<(std::ostream& out, Player player);
std::ostream& operator<<(std::ostream& out, Cell cell);
std::ostream& operator<<(std::ostream& out, Wall all);
std::ostream& operator<<(std::ostream& out, Move move);
std::ostream& operator<<(std::ostream& out, Turn turn);

std::istream& operator>>(std::istream& out, Direction& dir);
std::istream& operator>>(std::istream& out, Player& player);
std::istream& operator>>(std::istream& out, Cell& cell);
std::istream& operator>>(std::istream& out, Wall& all);
std::istream& operator>>(std::istream& out, Move& move);
std::istream& operator>>(std::istream& out, Turn& turn);

class Board {
public:
    Board(int width, int height, Cell red_start, Cell red_goal, Cell blue_start, Cell blue_goal);

    bool is_blocked(Wall wall) const;

    std::vector<Direction> legal_directions(Player player) const;
    std::vector<Wall> legal_walls() const;
    std::vector<Move> legal_moves(Player player) const;

    void take_step(Player player, Direction direction);
    void place_wall(Player player, Wall wall);

    void do_move(Player player, Move move);

    std::optional<Player> winner() const;

    Cell position(Player player) const;
    Cell goal(Player player) const;

    int distance(Cell start, Cell target) const;

    int width() const {
        return m_width;
    }

    int height() const {
        return m_height;
    }

private:
    struct State {
        bool has_red_player : 1 = false;
        bool has_blue_player : 1 = false;
        bool has_red_right_wall : 1 = false;
        bool has_red_up_wall : 1 = false;
        bool has_blue_right_wall : 1 = false;
        bool has_blue_up_wall : 1 = false;
        bool has_red_goal : 1 = false;
        bool has_blue_goal : 1 = false;
    };

    struct {
        Cell position;
        Cell goal;
    } m_red, m_blue;

    int m_width;
    int m_height;

    std::vector<State> m_board;

    State& state_at(Cell cell);
    State state_at(Cell cell) const;

    std::pair<bool, int> find_bridges(Cell start, Cell target, int level, std::vector<int>& levels,
                                      std::set<Wall>& bridges) const;
};
