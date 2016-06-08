/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "dnnLstmLayer.hpp"
#include "../helpers/limitedError.cuh"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"

#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }

void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cudnnErrCheck(stat) { cudnnErrCheck_((stat), __FILE__, __LINE__); }
void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) {
   if (stat != CUDNN_STATUS_SUCCESS) {
      fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
   }
}

__global__ void initGPUData_ker(float *data, int numElements, float value) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < numElements) {
      data[tid] = value;
   }
}

void initGPUData(float *data, int numElements, float value) {
   dim3 gridDim;
   dim3 blockDim;

   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

   initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}
namespace internal {
namespace {

    typedef activation_functions::Logistic gate_act_fn_t;
    typedef activation_functions::Tanh     cell_input_act_fn_t;
    typedef activation_functions::Tanh     cell_output_act_fn_t;

    struct ComputeBlockOutputFn
    {
        int    effLayerSize;
        int    prevOutputDistance;
        real_t bias;

        const char *patTypes;

        const real_t *niBiasWeights;
        const real_t *igBiasWeights;
        const real_t *fgBiasWeights;
        const real_t *ogBiasWeights;

        const real_t *igPeepWeights;
        const real_t *fgPeepWeights;
        const real_t *ogPeepWeights;

        real_t *cellStates;
        real_t *niActs;
        real_t *igActs;
        real_t *fgActs;
        real_t *ogActs;

        __host__ __device__ real_t operator() (const int &outputIdx, const thrust::tuple<bool, bool> &t) const
        {
            // unpack the tuple
            bool firstCall    = t.get<0>();
            bool checkPatType = t.get<1>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set the all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    if (prevOutputDistance > 0)
                        cellStates[outputIdx] = 0;
                    return 0;
                }
            }

            // calculate indices
            int blockIdx = outputIdx % effLayerSize;

            // load the niag activations
            real_t niAct = niActs[outputIdx];
            real_t igAct = igActs[outputIdx];
            real_t fgAct = fgActs[outputIdx];
            real_t ogAct = ogActs[outputIdx];

            // add bias activations
            niAct += bias * niBiasWeights[blockIdx];
            igAct += bias * igBiasWeights[blockIdx];
            fgAct += bias * fgBiasWeights[blockIdx];
            ogAct += bias * ogBiasWeights[blockIdx];

            // add activation from peephole weights
            if (!firstCall) {
                real_t prevCellState = cellStates[outputIdx + prevOutputDistance];

                igAct += prevCellState * igPeepWeights[blockIdx];
                fgAct += prevCellState * fgPeepWeights[blockIdx];
            }

            // apply the activation functions
            niAct = cell_input_act_fn_t::fn(niAct);
            igAct = gate_act_fn_t      ::fn(igAct);
            fgAct = gate_act_fn_t      ::fn(fgAct);

            // store the niag activations
            niActs[outputIdx] = niAct;
            igActs[outputIdx] = igAct;
            fgActs[outputIdx] = fgAct;

            // calculate the cell state and store the result
            real_t cellState = niAct * igAct;

            if (!firstCall)
                cellState += cellStates[outputIdx + prevOutputDistance] * fgAct;

            cellStates[outputIdx] = cellState;

            // calculate the output gate activation and store the result
            ogAct += cellState * ogPeepWeights[blockIdx];
            ogAct = gate_act_fn_t::fn(ogAct);
            ogActs[outputIdx] = ogAct;

            // calculate the block output
            real_t output = cell_output_act_fn_t::fn(cellState) * ogAct;

            return output;
        }
    };

    struct ResortOutputsFn
    {
        int layerSize;
        int effLayerSize;

        const real_t *fwOutputs;
        const real_t *bwOutputs;

        __host__ __device__ real_t operator() (const int &outputIdx) const
        {
            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                return fwOutputs[offset];
            else
                return bwOutputs[offset - effLayerSize];
        }
    };

    struct ResortOutputsFnC
    {
        int layerSize;
        int effLayerSize;
        int maxSize;

        const real_t *fwOutputs;
        const real_t *bwOutputs;

        __host__ __device__ real_t operator() (const int &outputIdx) const
        {
            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;
            //            (patIdx + 2) * effLayerSize + (valIdx - effLayerSize)
            int offset2 = (patIdx + 2) * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                return fwOutputs[offset];
            else
                return (offset2 < maxSize)? bwOutputs[offset2] : (real_t)0.0;
        }
    };

    struct ResortOutputErrorsFn
    {
        int layerSize;
        int effLayerSize;

        real_t *fwOutputErrors;
        real_t *bwOutputErrors;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t outputErr = t.get<0>();
            int    outputIdx = t.get<1>();

            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                fwOutputErrors[offset] = outputErr;
            else
                bwOutputErrors[offset - effLayerSize] = outputErr;
        }
    };

    struct ResortOutputErrorsFnC
    {
        int layerSize;
        int effLayerSize;
        int maxSize;

        real_t *fwOutputErrors;
        real_t *bwOutputErrors;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t outputErr = t.get<0>();
            int    outputIdx = t.get<1>();

            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;
            int offset2 = (patIdx + 2) * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                fwOutputErrors[offset] = outputErr;
            else if (offset2 < maxSize)
                bwOutputErrors[offset2] = outputErr;
        }
    };

    struct ComputeBlockErrorsFn
    {
        int effLayerSize;
        int prevOutputDistance;

        const char *patTypes;

        const real_t *igPeepWeights;
        const real_t *fgPeepWeights;
        const real_t *ogPeepWeights;

        const real_t *cellStates;
        const real_t *niActs;
        const real_t *igActs;
        const real_t *fgActs;
        const real_t *ogActs;

        real_t *cellStateErrors;
        real_t *niDeltas;
        real_t *igDeltas;
        real_t *fgDeltas;
        real_t *ogDeltas;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int, bool, bool, bool> &t) const
        {
            // unpack the tuple
            real_t outputErr    = t.get<0>();
            int    outputIdx    = t.get<1>();
            bool   firstCall    = t.get<2>();
            bool   lastCall     = t.get<3>();
            bool   checkPatType = t.get<4>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    niDeltas       [outputIdx] = 0;
                    igDeltas       [outputIdx] = 0;
                    fgDeltas       [outputIdx] = 0;
                    ogDeltas       [outputIdx] = 0;
                    cellStateErrors[outputIdx] = 0;
                    return;
                }
            }

            // calculate indices
            int blockIdx = outputIdx % effLayerSize;

            // load the niag activations, the cell state and the output error
            real_t niAct     = niActs      [outputIdx];
            real_t igAct     = igActs      [outputIdx];
            real_t ogAct     = ogActs      [outputIdx];
            real_t cellState = cellStates  [outputIdx];

            // calculate the output gate delta
            real_t ogDelta = gate_act_fn_t::deriv(ogAct) * cell_output_act_fn_t::fn(cellState) * outputErr;

            // calculate the cell state error
            real_t ogPeepWeight = ogPeepWeights[blockIdx];
            real_t cellStateErr = ogAct * cell_output_act_fn_t::deriv(cell_output_act_fn_t::fn(cellState)) * outputErr + ogPeepWeight * ogDelta;

            if (!firstCall) {
                real_t nextFgAct        = fgActs         [outputIdx - prevOutputDistance];
                real_t nextCellStateErr = cellStateErrors[outputIdx - prevOutputDistance];
                real_t nextIgDelta      = igDeltas       [outputIdx - prevOutputDistance];
                real_t nextFgDelta      = fgDeltas       [outputIdx - prevOutputDistance];

                real_t igPeepWeight = igPeepWeights[blockIdx];
                real_t fgPeepWeight = fgPeepWeights[blockIdx];

                cellStateErr += nextFgAct * nextCellStateErr + igPeepWeight * nextIgDelta + fgPeepWeight * nextFgDelta;
            }

            // calculate the net input delta
            real_t niDelta = igAct * cell_input_act_fn_t::deriv(niAct) * cellStateErr;

            // calculate the forget gate delta
            real_t fgDelta = 0;

            if (!lastCall) {
                real_t fgAct         = fgActs    [outputIdx];
                real_t prevCellState = cellStates[outputIdx + prevOutputDistance];

                fgDelta = gate_act_fn_t::deriv(fgAct) * prevCellState * cellStateErr;
            }

            // calculate the input gate delta
            real_t igDelta = gate_act_fn_t::deriv(igAct) * niAct * cellStateErr;

            // store the niag deltas and the cell state error
            niDeltas       [outputIdx] = helpers::limitedError(niDelta);
            igDeltas       [outputIdx] = helpers::limitedError(igDelta);
            fgDeltas       [outputIdx] = helpers::limitedError(fgDelta);
            ogDeltas       [outputIdx] = helpers::limitedError(ogDelta);
            cellStateErrors[outputIdx] = cellStateErr;
        }
    };

    struct ComputeWeightUpdateFn
    {
        int    layerSize;
        int    effLayerSize;
        int    precLayerSize;
        int    timestepDistance;
        int    parallelSequences;
        int    patternsCount;
        int    biasWeightsOffset;
        int    internalWeightsOffset;
        int    peepholeWeightsOffset;
        real_t bias;

        const real_t *plOutputs;
        const real_t *fwOutputs;
        const real_t *bwOutputs;
        const real_t *fwCellStates;
        const real_t *bwCellStates;
        const real_t *fwNiDeltas;
        const real_t *bwNiDeltas;
        const real_t *fwIgDeltas;
        const real_t *bwIgDeltas;
        const real_t *fwFgDeltas;
        const real_t *bwFgDeltas;
        const real_t *fwOgDeltas;
        const real_t *bwOgDeltas;

        __host__ __device__ real_t operator() (const int &weightIdx) const
        {
            // determine the weight type
            //
            // weightType = 0bXXYY with XX = {input, bias, internal, peephole}
            //                     and  YY = {NI, IG, FG, OG}
            //
            // weightType = 0b0000 ( 0): NI input weight
            //              0b0001 ( 1): IG input weight
            //              0b0010 ( 2): FG input weight
            //              0b0011 ( 3): OG input weight
            //              0b0100 ( 4): NI bias weight
            //              0b0101 ( 5): IG bias weight
            //              0b0110 ( 6): FG bias weight
            //              0b0111 ( 7): OG bias weight
            //              0b1000 ( 8): NI internal weight
            //              0b1001 ( 9): IG internal weight
            //              0b1010 (10): FG internal weight
            //              0b1011 (11): OG internal weight
            //              0b1100 (12): not used
            //              0b1101 (13): IG peephole weight
            //              0b1110 (14): FG peephole weight
            //              0b1111 (15): OG peephole weight
            int inwc = layerSize * precLayerSize;
            int biwc = layerSize;
            int itwc = layerSize * effLayerSize;
            int pewc = layerSize;

            int weightType = (int)(weightIdx >= 0                     + 1 * inwc) +
                             (int)(weightIdx >= 0                     + 2 * inwc) +
                             (int)(weightIdx >= 0                     + 3 * inwc) +
                             (int)(weightIdx >= 0                     + 4 * inwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 1 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 2 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 3 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 4 * biwc) +
                             (int)(weightIdx >= internalWeightsOffset + 1 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 2 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 3 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 4 * itwc) * 2 +
                             (int)(weightIdx >= peepholeWeightsOffset + 1 * pewc) +
                             (int)(weightIdx >= peepholeWeightsOffset + 2 * pewc);

            int weightTypeX = weightType & 0xC;
            int weightTypeY = weightType & 0x3;

            // calculate indices, offsets and increments
            const real_t *offOutputs;
            int           tgtBlockIdx;
            int           offOutputsInc;
            bool          skipFirstPattern = false;
            bool          skipLastPattern  = false;
            bool          isBwStateWeight;

            switch (weightTypeX) {
            // input weight
            case 0x0:
                {{
                    // calculate indices
                    int inputWeightIdx = weightIdx;
                    int plBlockIdx     = inputWeightIdx % precLayerSize;
                    int blockIdx       = (inputWeightIdx - weightTypeY * (biasWeightsOffset/4)) / precLayerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = &plOutputs[plBlockIdx];
                    offOutputsInc = precLayerSize;
                }}
                break;

            // bias weight
            case 0x4:
                {{
                    // calculate indices
                    int biasWeightIdx = weightIdx - biasWeightsOffset;
                    int blockIdx      = biasWeightIdx - weightTypeY * layerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = NULL;
                    offOutputsInc = 0;
                }}
                break;

            // internal weight
            case 0x8:
                {{
                    // calculate indices
                    int internalWeightIdx = weightIdx - internalWeightsOffset;
                    int srcBlockIdx       = internalWeightIdx % effLayerSize;
                    int blockIdx          = internalWeightIdx / effLayerSize - weightTypeY * layerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = (isBwStateWeight ? &bwOutputs[srcBlockIdx] : &fwOutputs[srcBlockIdx]);
                    offOutputsInc = effLayerSize;

                    if (isBwStateWeight) {
                        offOutputs += timestepDistance;
                        skipLastPattern = true;
                    }
                    else {
                        offOutputs -= timestepDistance;
                        skipFirstPattern = true;
                    }
                }}
                break;

            // peephole weight
            default:
                {{
                    // calculate indices
                    int peepholeWeightIdx = weightIdx - peepholeWeightsOffset;
                    int blockIdx          = peepholeWeightIdx - (weightTypeY-1) * layerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // select the appropriate cell states and adjust the block index
                    const real_t *cellStates = (isBwStateWeight ? bwCellStates : fwCellStates);

                    // set the timeshift
                    int timeShift;
                    if (weightTypeY == 0x3) {
                        timeShift = 0;
                    }
                    else {
                        if (isBwStateWeight) {
                            timeShift       = timestepDistance;
                            skipLastPattern = true;
                        }
                        else {
                            timeShift        = -timestepDistance;
                            skipFirstPattern = true;
                        }
                    }

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = &cellStates[blockIdx + timeShift];
                    offOutputsInc = effLayerSize;
                }}
                break;
            }

            // determine the start of the delta values
            const real_t *niagDeltasLut[] = {
                fwNiDeltas,
                fwIgDeltas,
                fwFgDeltas,
                fwOgDeltas,
                bwNiDeltas,
                bwIgDeltas,
                bwFgDeltas,
                bwOgDeltas
            };

            // calculate the weight update over all patterns
            const real_t *offDeltas = &niagDeltasLut[weightTypeY + (isBwStateWeight ? 4 : 0)][tgtBlockIdx];

            if (skipFirstPattern) {
                offOutputs += parallelSequences * offOutputsInc;
                offDeltas  += parallelSequences * effLayerSize;
            }

            int numPatterns = patternsCount;
            if (skipFirstPattern || skipLastPattern)
                numPatterns -= parallelSequences;

            real_t wu = 0;
            for (int i = 0; i < numPatterns; ++i) {
                wu += (offOutputs ? *offOutputs : bias) * *offDeltas;

                offOutputs += offOutputsInc;
                offDeltas  += effLayerSize;
            }

            return wu;
        }
    };

} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice>
    dnnLstmLayer<TDevice>::dnnLstmLayer(const helpers::JsonValue &layerChild,
                                        const helpers::JsonValue &weightsSection,
                                        Layer<TDevice> &precedingLayer,
                                        bool bidirectional,
                                        int num_layer,
                                        float dropout)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 4, (bidirectional ? 2 : 4) * helpers::safeJsonGetInt(layerChild, "size") + 3, precedingLayer)
        , m_isBidirectional      (bidirectional)
        , numLayers      (num_layer)
    {
        if (m_isBidirectional && this->size() % 2 != 0)
            throw std::runtime_error("Cannot create a bidirectional layer with an odd layer size");



         // Set up tensor descriptors. x/y/dx/dy are arrays, one per time step.
        int seqLength = this->curMaxSeqLength();
        int hiddenSize = this->size() / (m_isBidirectional ? 2 : 1);
        int miniBatch = this->parallelSequences();
        int inputSize = this->precedingLayer().size();
        cudnnErrCheck(cudnnCreate(&cudnnHandle));

        // cudnnTensorDescriptor_t *xDesc, *yDesc, *dxDesc, *dyDesc;
        // cudnnTensorDescriptor_t hxDesc, cxDesc;
        // cudnnTensorDescriptor_t hyDesc, cyDesc;
        // cudnnTensorDescriptor_t dhxDesc, dcxDesc;
        // cudnnTensorDescriptor_t dhyDesc, dcyDesc;
        //
        // cudaErrCheck(cudaMalloc((void**)&x, seqLength * inputSize * miniBatch * sizeof(float)));
        x = helpers::getRawPointer(this->precedingLayer().outputs());
        // cudaErrCheck(cudaMalloc((void**)&hx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&cx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));

        // cudaErrCheck(cudaMalloc((void**)&dx, seqLength * inputSize * miniBatch * sizeof(float)));
        dx = helpers::getRawPointer(this->precedingLayer().outputErrors());
        // cudaErrCheck(cudaMalloc((void**)&dhx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&dcx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));

        // cudaErrCheck(cudaMalloc((void**)&y, seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
        // outputs()
        y = helpers::getRawPointer(this->outputs());
        // cudaErrCheck(cudaMalloc((void**)&hy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&cy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));

        // cudaErrCheck(cudaMalloc((void**)&dy, seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
        // outputErrors()
        dy = helpers::getRawPointer(this->outputErrors());
        // cudaErrCheck(cudaMalloc((void**)&dhy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void**)&dcy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));

        xDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
        yDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
        dxDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
        dyDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));

        int dimA[3];
        int strideA[3];

        for (int i = 0; i < seqLength; i++) {
            cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc[i]));
            cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc[i]));
            cudnnErrCheck(cudnnCreateTensorDescriptor(&dxDesc[i]));
            cudnnErrCheck(cudnnCreateTensorDescriptor(&dyDesc[i]));

            dimA[0] = miniBatch;
            dimA[1] = inputSize;
            dimA[2] = 1;

            strideA[0] = dimA[2] * dimA[1];
            strideA[1] = dimA[2];
            strideA[2] = 1;

            cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
            cudnnErrCheck(cudnnSetTensorNdDescriptor(dxDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

            dimA[0] = miniBatch;
            dimA[1] = bidirectional ? hiddenSize * 2 : hiddenSize;
            dimA[2] = 1;

            strideA[0] = dimA[2] * dimA[1];
            strideA[1] = dimA[2];
            strideA[2] = 1;

            cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
            cudnnErrCheck(cudnnSetTensorNdDescriptor(dyDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
        }


        dimA[0] = numLayers * (bidirectional ? 2 : 1);
        dimA[1] = miniBatch;
        dimA[2] = hiddenSize;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;
        cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&dhxDesc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&dcxDesc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&dhyDesc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&dcyDesc));

        cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(dhxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(dcxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(dhyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(dcyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));


        // -------------------------
        // Set up the dropout descriptor (needed for the RNN descriptor)
        // -------------------------
        unsigned long long seed = 1337ull; // Pick a seed.

        // cudnnDropoutDescriptor_t dropoutDesc;
        cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc));

        // How much memory does dropout need for states?
        // These states are used to generate random numbers internally
        // and should not be freed until the RNN descriptor is no longer used
        // size_t stateSize;
        // void *states;
        cudnnErrCheck(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));

        cudaErrCheck(cudaMalloc(&states, stateSize));

        cudnnErrCheck(cudnnSetDropoutDescriptor(dropoutDesc,
                                   cudnnHandle,
                                   dropout,
                                   states,
                                   stateSize,
                                   seed));

        // -------------------------
        // Set up the RNN descriptor
        // -------------------------
        // cudnnRNNDescriptor_t rnnDesc;
        // cudnnRNNMode_t RNNMode;

        cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc));
        RNNMode = CUDNN_LSTM;

        cudnnErrCheck(cudnnSetRNNDescriptor(rnnDesc,
                                            hiddenSize,
                                            numLayers,
                                            dropoutDesc,
                                            CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                                            bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                                            RNNMode,
                                            CUDNN_DATA_FLOAT));


        // -------------------------
        // Set up parameters
        // -------------------------
        // This needs to be done after the rnn descriptor is set as otherwise
        // we don't know how many parameters we have to allocate
        // void *w;
        // void *dw;


        cudnnErrCheck(cudnnCreateFilterDescriptor(&wDesc));
        cudnnErrCheck(cudnnCreateFilterDescriptor(&dwDesc));

        cudnnErrCheck(cudnnGetRNNParamsSize(cudnnHandle, rnnDesc, xDesc[0], &weightsSize, CUDNN_DATA_FLOAT));
        assert(weightsSize == this->weights().size());

        int dimW[3];
        dimW[0] =  weightsSize / sizeof(float);
        dimW[1] = 1;
        dimW[2] = 1;

        cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
        cudnnErrCheck(cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));

        // cudaErrCheck(cudaMalloc((void**)&w,  weightsSize));
        w = helpers::getRawPointer(this->weights());
        // cudaErrCheck(cudaMalloc((void**)&dw, weightsSize));
        dw = helpers::getRawPointer(this->_weightUpdates());


        // -------------------------
        // Set up work space and reserved memory
        // -------------------------
        // void *workspace;
        // void *reserveSpace;
        //
        // size_t workSize;
        // size_t reserveSize;

        // Need for every pass
        cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc, seqLength, xDesc, &workSize));
        // Only needed in training, shouldn't be touched between passes.
        cudnnErrCheck(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc, seqLength, xDesc, &reserveSize));

        cudaErrCheck(cudaMalloc((void**)&workspace, workSize));
        cudaErrCheck(cudaMalloc((void**)&reserveSpace, reserveSize));

        // *********************************************************************************************************
        // Initialise weights and inputs
        // *********************************************************************************************************
        // We initialise to something simple.
        // Matrices are initialised to 1 / matrixSize, biases to 1, data is 1.
        //initGPUData((float*)x, seqLength * inputSize * miniBatch, 1.f);
        // if (hx != NULL) initGPUData((float*)hx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
        // if (cx != NULL) initGPUData((float*)cx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);

        // initGPUData((float*)dy, seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
        // if (dhy != NULL) initGPUData((float*)dhy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
        // if (dcy != NULL) initGPUData((float*)dcy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);


        // Weights
        int numLinearLayers = 0;
        if (RNNMode == CUDNN_RNN_RELU || RNNMode == CUDNN_RNN_TANH) {
          numLinearLayers = 2;
        }
        else if (RNNMode == CUDNN_LSTM) {
          numLinearLayers = 8;
        }
        else if (RNNMode == CUDNN_GRU) {
          numLinearLayers = 6;
        }

        for (int layer = 0; layer < numLayers * (bidirectional ? 2 : 1); layer++) {
            for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
                cudnnFilterDescriptor_t linLayerMatDesc;
                cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
                float *linLayerMat;

                cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams( cudnnHandle,
                                                                rnnDesc,
                                                                layer,
                                                                xDesc[0],
                                                                wDesc,
                                                                w,
                                                                linLayerID,
                                                                linLayerMatDesc,
                                                                (void**)&linLayerMat));

                cudnnDataType_t dataType;
                cudnnTensorFormat_t format;
                int nbDims;
                int filterDimA[3];
                cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc,
                                                          3,
                                                          &dataType,
                                                          &format,
                                                          &nbDims,
                                                          filterDimA));

                initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f / (float)(filterDimA[0] * filterDimA[1] * filterDimA[2]));

                cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));

                cudnnFilterDescriptor_t linLayerBiasDesc;
                cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
                float *linLayerBias;

                cudnnErrCheck(cudnnGetRNNLinLayerBiasParams( cudnnHandle,
                                                                rnnDesc,
                                                                layer,
                                                                xDesc[0],
                                                                wDesc,
                                                                w,
                                                                linLayerID,
                                                                linLayerBiasDesc,
                                                                (void**)&linLayerBias));

                cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
                                                          3,
                                                          &dataType,
                                                          &format,
                                                          &nbDims,
                                                          filterDimA));

                initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f);

                cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
            }
        }

      // *********************************************************************************************************
      // At this point all of the setup is done. We now need to pass through the RNN.
      // *********************************************************************************************************



      cudaErrCheck(cudaDeviceSynchronize());

    }

    template <typename TDevice>
    dnnLstmLayer<TDevice>::~dnnLstmLayer()
    {

      // cudaFree(x);
      cudaFree(hx);
      cudaFree(cx);
      // cudaFree(y);
      cudaFree(hy);
      cudaFree(cy);
      cudaFree(dx);
      cudaFree(dhx);
      cudaFree(dcx);
      // cudaFree(dy);
      cudaFree(dhy);
      cudaFree(dcy);
      cudaFree(workspace);
      cudaFree(reserveSpace);
      // cudaFree(w);
      // cudaFree(dw);

    }

    template <typename TDevice>
    const std::string& dnnLstmLayer<TDevice>::type() const
    {
        static const std::string su("lstm");
        static const std::string sb("blstm");
        return (m_isBidirectional ? sb : su);
    }

    template <typename TDevice>
    bool dnnLstmLayer<TDevice>::isBidirectional() const
    {
        return m_isBidirectional;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::cellStates() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStates;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::cellStateErrors() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStateErrors;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::netInputActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::netInputDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::inputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::inputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::forgetGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::forgetGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::outputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& dnnLstmLayer<TDevice>::outputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogDeltas;
    }

    template <typename TDevice>
    void dnnLstmLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        TrainableLayer<TDevice>::loadSequences(fraction);
        m_precLayerOutputsMatrix = helpers::Matrix<TDevice>(&this->precedingLayer().outputs(), this->precedingLayer().size(), this->curMaxSeqLength() * this->parallelSequences());

        // update the niag matrices
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            int rows = this->size() / (m_isBidirectional ? 2 : 1);
            int cols = this->curMaxSeqLength() * this->parallelSequences();
            // printf("row, col : %d, %d  data->size() : %d  dataOffset : %d\n", rows, cols, data->size(), dataOffset);
            // printf("row, col : %d, %d \n", rows, cols);

            fwbw->niActsMatrix = helpers::Matrix<TDevice>(&fwbw->niActs, rows, cols);
            fwbw->igActsMatrix = helpers::Matrix<TDevice>(&fwbw->igActs, rows, cols);
            fwbw->fgActsMatrix = helpers::Matrix<TDevice>(&fwbw->fgActs, rows, cols);
            fwbw->ogActsMatrix = helpers::Matrix<TDevice>(&fwbw->ogActs, rows, cols);

            fwbw->niDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->niDeltas, rows, cols);
            fwbw->igDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->igDeltas, rows, cols);
            fwbw->fgDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->fgDeltas, rows, cols);
            fwbw->ogDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->ogDeltas, rows, cols);
        }
    }

    template <typename TDevice>
    void dnnLstmLayer<TDevice>::computeForwardPass()
    {
        // for unidirectional LSTM, we can write the outputs directly in the layer output vector
        // if (!m_isBidirectional) {
        //     m_fw.tmpOutputs.swap(this->_outputs());
        // }

        // sum up the activations from the preceding layer
        cudnnErrCheck(cudnnRNNForwardTraining(cudnnHandle,
                                         rnnDesc,
                                         this->curMaxSeqLength(),
                                         xDesc,
                                         x,
                                         hxDesc,
                                         hx,
                                         cxDesc,
                                         cx,
                                         wDesc,
                                         w,
                                         yDesc,
                                         y,
                                         hyDesc,
                                         hy,
                                         cyDesc,
                                         cy,
                                         workspace,
                                         workSize,
                                         reserveSpace,
                                         reserveSize));
        cudaDeviceSynchronize();
    }

    template <typename TDevice>
    void dnnLstmLayer<TDevice>::computeBackwardPass()
    {
        // for unidirectional LSTM, we can write the output errors directly in the layer output errors vector

        cudnnErrCheck(cudnnRNNBackwardData(cudnnHandle,
                                           rnnDesc,
                                           this->curMaxSeqLength(),
                                           yDesc,
                                           y,
                                           dyDesc,
                                           dy,
                                           dhyDesc,
                                           dhy,
                                           dcyDesc,
                                           dcy,
                                           wDesc,
                                           w,
                                           hxDesc,
                                           hx,
                                           cxDesc,
                                           cx,
                                           dxDesc,
                                           dx,
                                           dhxDesc,
                                           dhx,
                                           dcxDesc,
                                           dcx,
                                           workspace,
                                           workSize,
                                           reserveSpace,
                                           reserveSize ));


        // cudnnRNNBackwardWeights adds to the data in dw.
        cudaErrCheck(cudaMemset(dw, 0, weightsSize));

        cudnnErrCheck(cudnnRNNBackwardWeights( cudnnHandle,
                                               rnnDesc,
                                               this->curMaxSeqLength(),
                                               xDesc,
                                               x,
                                               hxDesc,
                                               hx,
                                               yDesc,
                                               y,
                                               workspace,
                                               workSize,
                                               dwDesc,
                                               dw,
                                               reserveSpace,
                                               reserveSize ));

        cudaDeviceSynchronize();

    }


    // explicit template instantiations
    template class dnnLstmLayer<Gpu>;

} // namespace layers
