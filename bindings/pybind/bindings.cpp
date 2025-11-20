#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "balatro/simulator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_balatro_core, m) {
    m.doc() = "Balatro poker simulator core module (v0.2 - Direct Card Control)";

    // Expose constants
    m.attr("HAND_SIZE") = balatro::HAND_SIZE;
    m.attr("NUM_RANKS") = balatro::NUM_RANKS;
    m.attr("NUM_SUITS") = balatro::NUM_SUITS;

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
        // Hand analysis features
        .def_readwrite("has_pair", &balatro::Observation::has_pair)
        .def_readwrite("has_trips", &balatro::Observation::has_trips)
        .def_readwrite("straight_potential", &balatro::Observation::straight_potential)
        .def_readwrite("flush_potential", &balatro::Observation::flush_potential)
        // Card arrays as numpy arrays
        .def_property_readonly("card_ranks",
            [](const balatro::Observation& obs) {
                return py::array_t<int>(balatro::HAND_SIZE, obs.card_ranks);
            })
        .def_property_readonly("card_suits",
            [](const balatro::Observation& obs) {
                return py::array_t<int>(balatro::HAND_SIZE, obs.card_suits);
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
        .def("__repr__", [](const balatro::Simulator&) {
            return "<Simulator>";
        });
}
