#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "balatro/simulator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_balatro_core, m) {
    m.doc() = "Balatro poker simulator core module (v0.2 - Direct Card Control)";

    // Expose constants
    m.attr("HAND_SIZE") = balatro::HAND_SIZE;
    m.attr("NUM_RANKS") = balatro::NUM_RANKS;
    m.attr("NUM_SUITS") = balatro::NUM_SUITS;

    // HandType enum (for hand evaluation)
    py::enum_<balatro::HandType>(m, "HandType")
        .value("HIGH_CARD", balatro::HandType::HIGH_CARD)
        .value("PAIR", balatro::HandType::PAIR)
        .value("TWO_PAIR", balatro::HandType::TWO_PAIR)
        .value("THREE_OF_A_KIND", balatro::HandType::THREE_OF_A_KIND)
        .value("STRAIGHT", balatro::HandType::STRAIGHT)
        .value("FLUSH", balatro::HandType::FLUSH)
        .value("FULL_HOUSE", balatro::HandType::FULL_HOUSE)
        .value("FOUR_OF_A_KIND", balatro::HandType::FOUR_OF_A_KIND)
        .value("STRAIGHT_FLUSH", balatro::HandType::STRAIGHT_FLUSH)
        .export_values();

    // HandEvaluation struct
    py::class_<balatro::HandEvaluation>(m, "HandEvaluation")
        .def(py::init<>())
        .def_readonly("type", &balatro::HandEvaluation::type)
        .def_readonly("rank_sum", &balatro::HandEvaluation::rank_sum)
        .def("__repr__", [](const balatro::HandEvaluation& eval) {
            return "<HandEvaluation type=" + std::to_string(static_cast<int>(eval.type)) +
                   " rank_sum=" + std::to_string(eval.rank_sum) + ">";
        });

    // ActionOutcome struct (for action enumeration)
    py::class_<balatro::ActionOutcome>(m, "ActionOutcome")
        .def(py::init<>())
        .def_readwrite("action", &balatro::ActionOutcome::action)
        .def_readwrite("valid", &balatro::ActionOutcome::valid)
        .def_readwrite("predicted_chips", &balatro::ActionOutcome::predicted_chips)
        .def_readwrite("predicted_hand_type", &balatro::ActionOutcome::predicted_hand_type)
        .def("__repr__", [](const balatro::ActionOutcome& outcome) {
            return "<ActionOutcome valid=" + std::string(outcome.valid ? "True" : "False") +
                   " chips=" + std::to_string(outcome.predicted_chips) + ">";
        });

    // Observation struct with full card visibility
    py::class_<balatro::Observation>(m, "Observation")
        .def(py::init<>())
        // State features
        .def_readwrite("plays_left", &balatro::Observation::plays_left)
        .def_readwrite("discards_left", &balatro::Observation::discards_left)
        .def_readwrite("chips", &balatro::Observation::chips)
        .def_readwrite("chips_to_target", &balatro::Observation::chips_to_target)
        .def_readwrite("deck_remaining", &balatro::Observation::deck_remaining)
        .def_readwrite("discard_pile_size", &balatro::Observation::discard_pile_size)
        .def_readwrite("num_face_cards", &balatro::Observation::num_face_cards)
        .def_readwrite("num_aces", &balatro::Observation::num_aces)
        // Hand analysis features
        .def_readwrite("has_pair", &balatro::Observation::has_pair)
        .def_readwrite("has_trips", &balatro::Observation::has_trips)
        .def_readwrite("straight_potential", &balatro::Observation::straight_potential)
        .def_readwrite("flush_potential", &balatro::Observation::flush_potential)
        // Best hand analysis (for RL agents)
        .def_readwrite("best_hand_type", &balatro::Observation::best_hand_type)
        .def_readwrite("best_hand_score", &balatro::Observation::best_hand_score)
        // Complete hand pattern flags
        .def_readwrite("has_two_pair", &balatro::Observation::has_two_pair)
        .def_readwrite("has_full_house", &balatro::Observation::has_full_house)
        .def_readwrite("has_four_of_kind", &balatro::Observation::has_four_of_kind)
        .def_readwrite("has_straight", &balatro::Observation::has_straight)
        .def_readwrite("has_flush", &balatro::Observation::has_flush)
        .def_readwrite("has_straight_flush", &balatro::Observation::has_straight_flush)
        // Card arrays as numpy arrays
        .def_property_readonly("card_ranks",
            [](const balatro::Observation& obs) {
                // Allocate on heap and transfer ownership to Python
                auto* data = new std::vector<int>(obs.card_ranks, obs.card_ranks + balatro::HAND_SIZE);
                auto capsule = py::capsule(data, [](void *v) { delete reinterpret_cast<std::vector<int>*>(v); });
                return py::array_t<int>(
                    {balatro::HAND_SIZE},  // shape
                    {sizeof(int)},          // strides
                    data->data(),           // data pointer
                    capsule                 // ownership
                );
            })
        .def_property_readonly("card_suits",
            [](const balatro::Observation& obs) {
                // Allocate on heap and transfer ownership to Python
                auto* data = new std::vector<int>(obs.card_suits, obs.card_suits + balatro::HAND_SIZE);
                auto capsule = py::capsule(data, [](void *v) { delete reinterpret_cast<std::vector<int>*>(v); });
                return py::array_t<int>(
                    {balatro::HAND_SIZE},  // shape
                    {sizeof(int)},          // strides
                    data->data(),           // data pointer
                    capsule                 // ownership
                );
            })
        .def("__repr__", [](const balatro::Observation& obs) {
            return "<Observation plays=" + std::to_string(obs.plays_left) +
                   " discards=" + std::to_string(obs.discards_left) +
                   " chips=" + std::to_string(obs.chips) + ">";
        });

    // Action type enum
    py::enum_<balatro::Action::Type>(m, "ActionType")
        .value("PLAY", balatro::Action::Type::PLAY)
        .value("DISCARD", balatro::Action::Type::DISCARD)
        .export_values();

    // Action struct for direct card selection
    py::class_<balatro::Action>(m, "Action")
        .def(py::init<>())
        .def(py::init<balatro::Action::Type, const std::array<bool, balatro::HAND_SIZE>&>(),
             py::arg("type"),
             py::arg("card_mask"))
        .def_readwrite("type", &balatro::Action::type)
        .def_property("card_mask",
            [](const balatro::Action& action) {
                py::list mask;
                for (bool b : action.card_mask) {
                    mask.append(b);
                }
                return mask;
            },
            [](balatro::Action& action, py::list mask) {
                if (mask.size() != balatro::HAND_SIZE) {
                    throw std::runtime_error("card_mask must have length " +
                                           std::to_string(balatro::HAND_SIZE));
                }
                for (size_t i = 0; i < balatro::HAND_SIZE; ++i) {
                    action.card_mask[i] = mask[i].cast<bool>();
                }
            })
        .def("__repr__", [](const balatro::Action& action) {
            std::string type_str = (action.type == balatro::Action::PLAY) ? "PLAY" : "DISCARD";
            std::string mask_str = "[";
            for (size_t i = 0; i < balatro::HAND_SIZE; ++i) {
                mask_str += action.card_mask[i] ? "1" : "0";
                if (i < balatro::HAND_SIZE - 1) mask_str += ",";
            }
            mask_str += "]";
            return "<Action type=" + type_str + " mask=" + mask_str + ">";
        });

    // Action validation result
    py::class_<balatro::ActionValidationResult>(m, "ActionValidationResult")
        .def(py::init<>())
        .def_readonly("valid", &balatro::ActionValidationResult::valid)
        .def_readonly("error_message", &balatro::ActionValidationResult::error_message)
        .def("__repr__", [](const balatro::ActionValidationResult& result) {
            if (result.valid) {
                return std::string("<ActionValidationResult valid=True>");
            } else {
                return std::string("<ActionValidationResult valid=False error='") +
                       result.error_message + std::string("'>");
            }
        });

    // StepBatchResult struct
    py::class_<balatro::StepBatchResult>(m, "StepBatchResult")
        .def_readonly("final_obs", &balatro::StepBatchResult::final_obs)
        .def_readonly("rewards", &balatro::StepBatchResult::rewards)
        .def_readonly("done", &balatro::StepBatchResult::done)
        .def_readonly("win", &balatro::StepBatchResult::win)
        .def("__repr__", [](const balatro::StepBatchResult& result) {
            return "<StepBatchResult done=" + std::to_string(result.done) +
                   " win=" + std::to_string(result.win) + ">";
        });

    // Simulator class
    py::class_<balatro::Simulator>(m, "Simulator")
        .def(py::init<>())
        .def("reset", &balatro::Simulator::reset,
             py::arg("target_score"),
             py::arg("seed"),
             "Reset environment to new episode")
        .def("step_batch", &balatro::Simulator::step_batch,
             py::arg("actions"),
             "Execute a batch of actions")
        .def("validate_action", &balatro::Simulator::validate_action,
             py::arg("action"),
             "Validate an action before execution")
        // RL helper methods
        .def("get_best_hand", &balatro::Simulator::get_best_hand,
             "Get the best possible hand from current state")
        .def("predict_play_score", &balatro::Simulator::predict_play_score,
             py::arg("card_mask"),
             "Predict score for a PLAY action without executing it")
        .def("enumerate_all_actions", &balatro::Simulator::enumerate_all_actions,
             "Enumerate all valid actions with predicted outcomes")
        .def("__repr__", [](const balatro::Simulator&) {
            return "<Simulator>";
        });
}
