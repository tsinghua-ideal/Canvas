#pragma once

#include "Canvas/Core/Graph.hpp"
#include "Canvas/Primitives/Factory.hpp"


namespace canvas {

GraphSP CreateConvolutionGraph();

GraphSP CreateDepthWiseConvolutionGraph();

GraphSP CreateDepthWiseSeparableConvolutionGraph();

GraphSP CreateGroupedConvolutionGraph();

GraphSP CreateInvolutionGraph();

GraphSP CreateSEConvolutionGraph();

} // End namespace canvas
