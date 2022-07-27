#include <bitset>
#include <map>

#include "Canvas/Search/ReceptiveAnalyzer.hpp"
#include "Canvas/Primitives/Factory.hpp"


namespace canvas {

int ReceptiveAnalyzer::GetReceptiveSize(const GraphSP& graph) {
    auto Index = [](int x, int y) -> int {
        while (x < 0)
            x += kMaxReceptiveLength;
        while (y < 0)
            y += kMaxReceptiveLength;
        if (x >= kMaxReceptiveLength)
            x %= kMaxReceptiveLength;
        if (y >= kMaxReceptiveLength)
            y %= kMaxReceptiveLength;
        return x * kMaxReceptiveLength + y;
    };

    typedef std::bitset<kMaxReceptiveLength * kMaxReceptiveLength> Bitmap;
    std::map<TensorSP, Bitmap> receptive_field;
    // Gather information in a reversed order.
    for (const auto& p: graph->primitives) {
        if (auto in = DynamicCast<InputPrimitive>(p)) {
            receptive_field[in->outs[0]][Index(kCenterIndex, kCenterIndex)] = true;
        } else if (DynamicCast<ConvolutionPrimitive>(p) or DynamicCast<UnfoldPrimitive>(p)) {
            int kh = 0, kw = 0, dh = 1, dw = 1;
            if (auto conv = DynamicCast<ConvolutionPrimitive>(p))
                kh = conv->kh, kw = conv->kw, dh = conv->dh, dw = conv->dw;
            else if (auto unfold = DynamicCast<UnfoldPrimitive>(p)) {
                if (unfold->type == UnfoldH or unfold->type == UnfoldHW)
                    kh = unfold->k, dh = unfold->d;
                if (unfold->type == UnfoldW or unfold->type == UnfoldHW)
                    kw = unfold->k, dw = unfold->d;
            }
            auto& original = receptive_field[p->ins[0]];
            auto field = Bitmap();
            for (int i = 0; i < kMaxReceptiveLength; ++ i)
                for (int j = 0; j < kMaxReceptiveLength; ++ j)
                    if (original[Index(i, j)])
                        for (int khi = -kh / 2; khi <= kh / 2; ++ khi)
                            for (int khj = -kw / 2; khj <= kw / 2; ++ khj)
                                field[Index(i + khi * dh, j + khj * dw)] = true;
            receptive_field[p->outs[0]] = field;
        } else if (auto shift = DynamicCast<ShiftPrimitive>(p)) {
            int sh = 0, sw = 0;
            auto shape = shift->ins[0]->shape;
            for (const auto& index: shift->indices) {
                if (shape[index].static_power[StaticVarPos::VH] > 0)
                    sh = shift->k;
                if (shape[index].static_power[StaticVarPos::VW] > 0)
                    sw = shift->k;
            }
            auto field = Bitmap();
            auto& original = receptive_field[p->ins[0]];
            sh = RandomInt(-sh, sh), sw = RandomInt(-sw, sw);
            for (int i = 0; i < kMaxReceptiveLength; ++ i)
                for (int j = 0; j < kMaxReceptiveLength; ++ j)
                    if (original[Index(i, j)])
                        field[Index(i + sh, j + sw)] = true;
            receptive_field[p->outs[0]] = field;
        } else {
            assert(p->outs.size() == 1);
            receptive_field[p->outs[0]] = Bitmap();
            for (const auto& t: p->ins)
                receptive_field[p->outs[0]] |= receptive_field[t];
        }
    }
    assert(receptive_field.count(graph->Out()));
    return static_cast<int>(receptive_field[graph->Out()].count());
}

} // namespace canvas
