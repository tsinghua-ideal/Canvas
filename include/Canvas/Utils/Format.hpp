#pragma once

#include <iostream>


namespace canvas {

struct Indent {
    int level, num_spaces;

    explicit Indent(int num_spaces=4):
        level(0), num_spaces(num_spaces) {}

    friend std::ostream& operator << (std::ostream& os, const Indent& indent) {
        for (int i = 0; i < indent.level * indent.num_spaces; ++ i)
            os << " ";
        return os;
    }
};

struct IndentOS {
    std::ostream& os;
    Indent indent;

    explicit IndentOS(std::ostream& os, int num_spaces=4): os(os), indent(num_spaces) {}

    void BeginScope() { indent.level ++; }

    void EndScope() { indent.level --; }

    std::ostream& operator () (bool do_indent=true) const { return do_indent ? (os << indent) : os; }
};

} // End namespace canvas
