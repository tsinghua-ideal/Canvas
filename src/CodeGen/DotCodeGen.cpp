#include "Canvas/CodeGen/DotCodeGen.hpp"


namespace canvas {

Code DotCodeGen::GenImpl(const Solution& solution, std::string name) {
    auto graph = solution.graph;

    Write() << "digraph OpGraph {" << std::endl;
    BeginScope();

    // Title label
    if (name.empty())
        name = "OpGraph";
    Write() << "label = \"" << name << " [" << graph->Hash() << "]\"" << std::endl;

    // Font name
    Write() << "fontname = \"Graphik\";" << std::endl;

    // Tensor settings
    Write() << "subgraph tensors {" << std::endl;
    BeginScope();
    Write() << "node [shape = circle, color = black, fontname = \"Graphik\"]" << std::endl;
    Write();
    for (const auto& t: graph->tensors) {
        Write(false) << "t_" << t->id << "; ";
    }
    Write(false) << std::endl;
    EndScope();
    Write() << "}" << std::endl;

    // Clusters (Producer op)
    for (const auto& t: graph->tensors) {
        assert(t->producer);
        Write() << "subgraph cluster_" << t->id << " {" << std::endl;
        BeginScope();
        Write() << "fontcolor = blue;" << std::endl;
        Write() << "label = \"" << t->producer->id << ": " << t->producer->name << "\\n" << *t->producer << "\";" << std::endl;
        std::string suffix;
        Write() << "t_" << t->id << " [label = \"T" << t->id << suffix << "\\n" << t->shape << "\"];" << std::endl;
        EndScope();
        Write() << "}" << std::endl;
    }

    // Links
    for (const auto& t: graph->tensors) {
        for (const auto& p: t->consumers) {
            for (const auto& out_t: p->outs) {
                Write() << "t_" << t->id << " -> t_" << out_t->id << ";" << std::endl;
            }
        }
    }

    // End digraph
    EndScope();
    Write() << "}" << std::endl;
    Write() << std::endl;

    return Dump();
}

} // End namespace canvas
