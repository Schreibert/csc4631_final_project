#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "balatro/simulator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_balatro_core, m) {
    m.doc() = "Balatro poker simulator core module";

    // Expose constants
    m.attr("OBS_SIZE") = balatro::OBS_SIZE;
    m.attr("NUM_ACTIONS") = balatro::NUM_ACTIONS;
    m.attr("HAND_SIZE") = balatro::HAND_SIZE;

    // Action codes
    m.attr("ACTION_PLAY_BEST") = balatro::ACTION_PLAY_BEST;
    m.attr("ACTION_PLAY_PAIR") = balatro::ACTION_PLAY_PAIR;
    m.attr("ACTION_DISCARD_NON_PAIRED") = balatro::ACTION_DISCARD_NON_PAIRED;
    m.attr("ACTION_DISCARD_LOWEST_3") = balatro::ACTION_DISCARD_LOWEST_3;
    m.attr("ACTION_PLAY_HIGHEST_3") = balatro::ACTION_PLAY_HIGHEST_3;
    m.attr("ACTION_RANDOM_PLAY") = balatro::ACTION_RANDOM_PLAY;
    m.attr("ACTION_RANDOM_DISCARD") = balatro::ACTION_RANDOM_DISCARD;

    // Observation indices
    m.attr("OBS_PLAYS_LEFT") = balatro::OBS_PLAYS_LEFT;
    m.attr("OBS_DISCARDS_LEFT") = balatro::OBS_DISCARDS_LEFT;
    m.attr("OBS_CHIPS_TO_TARGET") = balatro::OBS_CHIPS_TO_TARGET;
    m.attr("OBS_HAS_PAIR") = balatro::OBS_HAS_PAIR;
    m.attr("OBS_HAS_TRIPS") = balatro::OBS_HAS_TRIPS;
    m.attr("OBS_STRAIGHT_POT") = balatro::OBS_STRAIGHT_POT;
    m.attr("OBS_FLUSH_POT") = balatro::OBS_FLUSH_POT;
    m.attr("OBS_MAX_RANK_BUCKET") = balatro::OBS_MAX_RANK_BUCKET;

    // StepBatchResult struct
    py::class_<balatro::StepBatchResult>(m, "StepBatchResult")
        .def_readonly("final_obs", &balatro::StepBatchResult::final_obs)
        .def_readonly("rewards", &balatro::StepBatchResult::rewards)
        .def_readonly("done", &balatro::StepBatchResult::done)
        .def_readonly("win", &balatro::StepBatchResult::win);

    // Simulator class
    py::class_<balatro::Simulator>(m, "Simulator")
        .def(py::init<>())
        .def("reset", &balatro::Simulator::reset,
             py::arg("target_score"),
             py::arg("seed"),
             "Reset environment to new episode")
        .def("step_batch", &balatro::Simulator::step_batch,
             py::arg("actions"),
             "Execute a batch of actions");
}
