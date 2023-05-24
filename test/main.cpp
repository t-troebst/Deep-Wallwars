#include <catch2/catch_session.hpp>
#include <folly/init/Init.h>

int main(int argc, char** argv) {
    folly::Init init(&argc, &argv, true);
    return Catch::Session().run(argc, argv);
}
