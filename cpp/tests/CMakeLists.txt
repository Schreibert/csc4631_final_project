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

# Enhanced observation golden test (standalone, no GTest)
add_executable(test_enhanced_observation
    test_enhanced_observation.cpp
)

target_link_libraries(test_enhanced_observation
    balatro_core
)

add_test(NAME enhanced_observation_test COMMAND test_enhanced_observation)

# Custom target to build all tests together
add_custom_target(all_tests DEPENDS balatro_tests test_enhanced_observation)
