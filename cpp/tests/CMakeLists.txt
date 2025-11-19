# Test executable
add_executable(balatro_tests
    test_card.cpp
    test_hand_eval.cpp
    test_scoring.cpp
    test_simulator.cpp
)

target_link_libraries(balatro_tests
    balatro_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(balatro_tests)
