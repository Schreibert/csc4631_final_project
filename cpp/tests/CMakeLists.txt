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

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(balatro_tests
    DISCOVERY_MODE PRE_TEST
)

# Also add a simple test that just runs the executable
# This works better with Visual Studio generator
add_test(NAME balatro_all_tests COMMAND balatro_tests)
