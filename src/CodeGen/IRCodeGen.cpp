#include "Canvas/CodeGen/IRCodeGen.hpp"


namespace canvas {

Code IRCodeGen::GenImpl(const Solution& solution, std::string name) {
    auto net_specs = solution.specs;
    auto graph = solution.graph;
    auto preferences = solution.preferences;
    auto net_fills = solution.fills;

    // Fills and specs need each other to check algebra legality
    assert((net_fills == nullptr) == (net_specs == nullptr));

    // Basics
    if (name.empty())
        name = "Kernel_" + std::to_string(solution.Hash());
    Write() << "name: " << name << std::endl;
    Write() << "dyn_vars: " << graph->DynamicVarCount() << std::endl;
    Write() << "g: " << preferences.g << std::endl;
    Write() << "r: " << preferences.r << std::endl;

    // Layer specifications
    if (not net_fills or graph->DynamicVarCount() == 0) {
        Write() << "layers: 0" << std::endl;
    } else {
        int n_layers = (int) net_fills->Size();
        auto& line = (Write() << "layers: " << n_layers);
        for (int i = 0; i < n_layers; ++ i)
            for (const auto& fill: net_fills->At(i).x)
                line << " " << fill;
        line << std::endl;
    }

    // Shapes
    Write() << "shapes:" << std::endl;
    BeginScope();
    for (const auto& t: graph->tensors)
        Write() << "t_" << t->id << " = Shape" << t->shape << std::endl;
    EndScope();

    auto DisplayTensorVector = [](std::ostream& os, const auto& vec) -> std::ostream& {
        bool displayed = false;
        os << "(";
        for (const auto& t: vec)
            os << (displayed ? ", t_" : "t_") << t->id, displayed = true;
        return os << ")";
    };

    // Links
    Write() << "links:" << std::endl;
    BeginScope();
    for (const auto& p: graph->primitives) {
        auto& os = DisplayTensorVector(Write(), p->outs);
        os << " = " << p->name;
        DisplayTensorVector(os, p->ins)<< std::endl;
    }
    EndScope();
    Write() << std::endl;
    return Dump();
}

} // End namespace canvas
