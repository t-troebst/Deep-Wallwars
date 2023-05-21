#include "play.hpp"

#include <algorithm>
#include <execution>
#include <iostream>
#include <random>
#include <sstream>

Player computer_play(std::unique_ptr<MCTSPolicy> policy1, std::unique_ptr<MCTSPolicy> policy2,
                     Board const& board, std::ostream& out, int seed) {
    MCTS mcts_1{board, {Player::Red, Turn::First}, std::move(policy1)};
    MCTS mcts_2{board, {Player::Red, Turn::First}, std::move(policy2)};
    std::mt19937_64 twister(seed);

    while (true) {
        for (int i = 0; i < 2; ++i) {
            double const s = mcts_1.sample(100);
            out << Player::Red << " estimates value at " << s << '\n';
            Move const m1 = mcts_1.commit_to_move(twister, 0.2);

            out << Player::Red << " does: " << m1 << '\n';

            if (auto const winner = mcts_1.current_board().winner(); winner) {
                out << *winner << " has won!\n";
                return *winner;
            }

            mcts_2.force_move(m1);
        }

        for (int i = 0; i < 2; ++i) {
            double const s = mcts_2.sample(100);
            out << Player::Blue << " estimates value at " << s << '\n';
            Move const m2 = mcts_2.commit_to_move(twister, 0.2);

            out << Player::Blue << " does: " << m2 << '\n';

            if (auto const winner = mcts_2.current_board().winner(); winner) {
                out << *winner << " has won!\n";
                return *winner;
            }

            mcts_1.force_move(m2);
        }
    }
}

double computer_eval(std::unique_ptr<MCTSPolicy> policy1, std::unique_ptr<MCTSPolicy> policy2,
                     Board const& board, std::ostream& out, int games) {
    std::vector<std::pair<Player, Player>> wins(games);
    int policy1_wins = 0;

    std::for_each(std::execution::par, wins.begin(), wins.end(),
                  [&](std::pair<Player, Player>& winners) {
                      std::random_device rd{};
                      std::stringstream game_out;

                      winners.first =
                          computer_play(policy1->clone(), policy2->clone(), board, game_out, rd());
                      winners.second =
                          computer_play(policy2->clone(), policy1->clone(), board, game_out, rd());

                      std::cout << winners.first << " won first game and " << winners.second
                                << " won second game.\n";
                  });

    for (auto const& winners : wins) {
        if (winners.first == Player::Red) {
            ++policy1_wins;
        }
        if (winners.second == Player::Blue) {
            ++policy1_wins;
        }
    }

    return double(policy1_wins) / (2 * games);
}

Player human_play(std::unique_ptr<MCTSPolicy> policy, Board const& board, std::ostream& out,
                  std::istream& in) {
    MCTS mcts{board, {Player::Red, Turn::First}, std::move(policy)};

    in >> std::skipws;

    while (true) {
        Move m;

        for (int i = 0; i < 2; ++i) {
            out << "Enter move: ";
            in >> m;
            out << "Doing move: " << m << '\n';
            mcts.force_move(m);

            if (auto const winner = mcts.current_board().winner(); winner) {
                out << *winner << " has won!\n";
                return *winner;
            }
        }

        for (int i = 0; i < 2; ++i) {
            double const s = mcts.sample(2000);
            out << Player::Blue << " estimates value at " << s << '\n';
            Move const m2 = mcts.commit_to_move();

            out << Player::Blue << " does: " << m2 << '\n';

            if (auto const winner = mcts.current_board().winner(); winner) {
                out << *winner << " has won!\n";
                return *winner;
            }
        }
    }
}
