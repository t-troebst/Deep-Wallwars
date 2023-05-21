#pragma once

#include <iosfwd>

#include "gamestate.hpp"
#include "mcts.hpp"

Player computer_play(std::unique_ptr<MCTSPolicy> policy1, std::unique_ptr<MCTSPolicy> policy2,
                     Board const& board, std::ostream& out, int seed = 42);

double computer_eval(std::unique_ptr<MCTSPolicy> policy1, std::unique_ptr<MCTSPolicy> policy2,
                     Board const& board, std::ostream& out, int games);

Player human_play(std::unique_ptr<MCTSPolicy> policy, Board const& board, std::ostream& out,
                  std::istream& in);
