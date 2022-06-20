#pragma once

#include <exception>

#include "Canvas/Utils/Common.hpp"


namespace canvas {

/// Exception class.
class ExceptionWithInfo: public std::exception {
public:
    std::string info;

    explicit ExceptionWithInfo(std::string info="An unknown exception occurs"):
            info(std::move(info)) {}

    [[nodiscard]] const char* what() const noexcept override {
        return info.c_str();
    }
};

class TimeoutException: ExceptionWithInfo {
public:
    explicit TimeoutException(canvas_timeval_t timeout):
            ExceptionWithInfo("Timeout in " + std::to_string(timeout.count()) + " sec(s)") {}
};

} // namespace canvas
