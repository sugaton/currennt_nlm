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

#include "dnnSoftmaxLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../activation_functions/Identity.cuh"

#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <cudnn.h>

#include <typeinfo>
#define SKIP_MARKER helpers::NumericLimits<real_t>::max()


void checkCUDNN(cudnnStatus_t stat)
{
    if (stat != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error( std::string("cudnn failture\nError: ") + cudnnGetErrorString(stat) );
}

namespace internal {
namespace {

    struct CalculateOffsetFn
    {
        int layerSize;

        const real_t *outputs;

        const char *patTypes;

        __host__ __device__ real_t operator() (const int &patIdx) const
        {
            // check if the pattern belongs to a sequence;
            // if not we return a certain number to avoid
            // looking up patTypes for future calculations
            if (patTypes[patIdx] == PATTYPE_NONE)
                return SKIP_MARKER;

            // search for the min and max output
            real_t max = helpers::NumericLimits<real_t>::min();
            real_t min = helpers::NumericLimits<real_t>::max();

            const real_t *offOutputs = &outputs[patIdx * layerSize];

            for (int i = 0; i < layerSize; ++i) {
                real_t x = offOutputs[i];
                min = helpers::min(min, x);
                max = helpers::max(max, x);
            }

            // calculate the offset
            real_t offset = (real_t)0.5 * (min + max);

            return offset;
        }
    };

    struct CalculateExpFn
    {
        int layerSize;

        const real_t *offsets;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            real_t output = t.get<0>();
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // check if we can stop the calculation
            real_t offset = offsets[patIdx];
            if (offset == SKIP_MARKER)
                return;

            // calculate the exponent
            real_t x = helpers::safeExp(output - offset);

            // store the result
            t.get<0>() = x;
        }
    };

    struct SumUpOutputsFn
    {
        int layerSize;

        const real_t *outputs;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int patIdx = t.get<1>();

            // check if the pattern belongs to a sequence
            if (t.get<0>() == SKIP_MARKER)
                return;

            // sum up the outputs
            const real_t *offOutputs = &outputs[patIdx * layerSize];

            real_t sum = 0;
            for (int i = 0; i < layerSize; ++i)
                sum += offOutputs[i];

            // store the result
            t.get<0>() = sum;
        }
    };

    struct residue_eq
    {
        int size;
        residue_eq(const int &_size) : size(_size){};
        __host__ __device__ bool operator() (const int &i, const int &j) const {
            return (i / size == j / size);
        }
    };

    struct addVector
    {
        int size;
        real_t* input;
        real_t* weight;
        real_t* output;
        __device__ void operator() (const int idx)
        {
            int j = idx % size;
            int layer = idx / size;
            if (weight[idx] == (real_t)0.0) return;
            atomicAdd(output + j, weight[layer] * input[idx]);
        }
    };

    struct SumUpOutputsAtomicFn
    {
        int layerSize;

        const real_t *outputs;
        real_t *patTmp;

        __host__ __device__ void operator() (const int &id) const
        {
            // unpack the tuple
            int patIdx  = id / layerSize;
            int localid = id % layerSize;

            // check if the pattern belongs to a sequence
            if (patTmp[patIdx] == SKIP_MARKER)
                return;

            // sum up the outputs
            // const real_t *offOutputs = &outputs[patIdx * layerSize];

            // for (int i = 0; i < layerSize; ++i)
            //     sum += offOutputs[i];
            #if defined(__CUDA_ARCH__)
                atomicAdd(patTmp + patIdx, outputs[id]);
            #else
                patTmp[patIdx] += outputs[id];
            #endif
            // store the result
            // t.get<0>() = sum;
        }
    };
    struct SumUpOutputsAtomicHFn{
        // /*
        int layerSize;

        const real_t *outputs;
        real_t *patTmp;

        __host__ void operator() (const int &id) const
        {
            // unpack the tuple
            int patIdx  = id / layerSize;
            int localid = id % layerSize;

            // check if the pattern belongs to a sequence
            if (patTmp[patIdx] == SKIP_MARKER)
                return;

            // sum up the outputs
            // const real_t *offOutputs = &outputs[patIdx * layerSize];

            // for (int i = 0; i < layerSize; ++i)
            //     sum += offOutputs[i];
            patTmp[patIdx] += outputs[id];
            // store the result
            // t.get<0>() = sum;
        }
    };

    struct NormalizeOutputsFn
    {
        int layerSize;

        const real_t *normFacts;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // check if we can stop the calculation
            real_t normFact = normFacts[patIdx];
            if (normFact == SKIP_MARKER)
                return;

            // calculate the normalized value
            real_t x = t.get<0>() / normFact;

            // store the result
            t.get<0>() = x;
        }
    };

    struct CalculateErrorOffsetFn
    {
        int layerSize;

        const real_t *outputs;
        const real_t *outputErrors;

        const char *patTypes;

        __host__ __device__ real_t operator() (const int &patIdx) const
        {
            // check if the pattern belongs to a sequence;
            // if not we return a certain number to avoid
            // looking up patTypes for future calculations
            if (patTypes[patIdx] == PATTYPE_NONE)
                return SKIP_MARKER;

            // calculate the offset
            const real_t *offOutputs      = &outputs     [patIdx * layerSize];
            const real_t *offOutputErrors = &outputErrors[patIdx * layerSize];

            real_t offset = 0;
            for (int i = 0; i < layerSize; ++i)
                offset += offOutputs[i] * offOutputErrors[i];

            return offset;
        }
    };

    struct CalculateErrorsFn
    {
        int layerSize;

        const real_t *errorOffsets;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<2>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // check if we can stop the calculation
            real_t offset = errorOffsets[patIdx];
            if (offset == SKIP_MARKER)
                return;

            // calculate the delta
            real_t error  = t.get<0>();
            real_t output = t.get<1>();

            real_t x = output * (error - offset);

            // store the result
            t.get<0>() = x;
        }
    };

} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice, typename TFfActFn>
    dnnSoftmaxLayer<TDevice, TFfActFn>::dnnSoftmaxLayer(
        const helpers::JsonValue &layerChild,
        const helpers::JsonValue &weightsSection,
        Layer<TDevice> &precedingLayer)
        : FeedForwardLayer<TDevice, TFfActFn>(layerChild, weightsSection, precedingLayer)
    {
        // resize the vector for temporary values
        // m_patTmp.resize(this->patTypes().size());
        int maxSeqL = precedingLayer.maxSeqLength();
        m_patTmp.resize(this->parallelSequences() * maxSeqL  * this->size());

        // for cudnn softmax
        //cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc, deltaInTensorDesc, deltaOutTensorDesc;
        cudnnCreate(&cudnnHandle);
        m_one = 1.0;
        m_zero = 0.0;
printf("size %d %d %d\n", this->size(), this->parallelSequences() , maxSeqL);
printf("line %d\n", __LINE__);
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
printf("line %d\n", __LINE__);
        checkCUDNN( cudnnCreateTensorDescriptor(&sftTensorDesc) );
printf("line %d\n", __LINE__);
        checkCUDNN( cudnnCreateTensorDescriptor(&deltaInTensorDesc) );
printf("line %d\n", __LINE__);
        checkCUDNN( cudnnCreateTensorDescriptor(&deltaOutTensorDesc) );
printf("line %d\n", __LINE__);
        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   this->parallelSequences() * maxSeqL, this->size(), 1, 1 ) );
printf("line %d\n", __LINE__);
        checkCUDNN( cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   this->parallelSequences() * maxSeqL, this->size(), 1, 1 ) );
printf("line %d\n", __LINE__);
        checkCUDNN( cudnnSetTensor4dDescriptor(deltaInTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   this->parallelSequences() * maxSeqL, this->size(), 1, 1 ) );
printf("line %d\n", __LINE__);
        checkCUDNN( cudnnSetTensor4dDescriptor(deltaOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   this->parallelSequences() * maxSeqL, this->size(), 1, 1 ) );
    }

    template <typename TDevice, typename TFfActFn>
    dnnSoftmaxLayer<TDevice, TFfActFn>::~dnnSoftmaxLayer()
    {
    }

    template <typename TDevice, typename TFfActFn>
    const std::string& dnnSoftmaxLayer<TDevice, TFfActFn>::type() const
    {
        static const std::string s = "softmax";
        return s;
    }

    template <typename TDevice, typename TFfActFn>
    void dnnSoftmaxLayer<TDevice, TFfActFn>::computeForwardPass()
    {
        // compute the forward pass of the feedforward layer
        FeedForwardLayer<TDevice, TFfActFn>::computeForwardPass();

        checkCUDNN( cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &m_one, srcTensorDesc, helpers::getRawPointer(this->_outputs()), &m_zero, sftTensorDesc, helpers::getRawPointer(m_patTmp)) );
        thrust::copy(m_patTmp.begin(), m_patTmp.end(), this->_outputs().begin());

        /*
        // calculate the offset to center the activations for safer exponentiation
        {{
            internal::CalculateOffsetFn fn;
            fn.layerSize = this->size();
            fn.outputs   = helpers::getRawPointer(this->_outputs());
            fn.patTypes  = helpers::getRawPointer(this->patTypes());

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->parallelSequences(),
                m_patTmp.begin(),
                fn
                );
        }}

        // calculate the exponent
        {{
            internal::CalculateExpFn fn;
            fn.layerSize = this->size();
            fn.offsets   = helpers::getRawPointer(m_patTmp);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->_outputs().begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(this->_outputs().begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }}
        #ifndef NEWSOFTMAX
        // sum up all outputs for each pattern
        {{
            internal::SumUpOutputsFn fn;
            fn.layerSize = this->size();
            fn.outputs   = helpers::getRawPointer(this->_outputs());

            int n = this->curMaxSeqLength() * this->parallelSequences();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(m_patTmp.begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(m_patTmp.begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }}
        #else
        {{
            int n = this->size() * this->curMaxSeqLength() * this->parallelSequences();
            thrust::reduce_by_key(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + n,
                this->_outputs().begin(),
                thrust::make_discard_iterator(),
                m_patTmp.begin(),
                internal::residue_eq(this->size()) );

        }}
        #endif
        // normalize the outputs
        {{
            internal::NormalizeOutputsFn fn;
            fn.layerSize = this->size();
            fn.normFacts = helpers::getRawPointer(m_patTmp);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->_outputs().begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(this->_outputs().begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }}

        m_patTmp.resize(this->patTypes().size());
        */
    }
    template <typename TDevice, typename TFfActFn>
    void dnnSoftmaxLayer<TDevice, TFfActFn>::computeBackwardPass()
    {

        checkCUDNN( cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &m_one, srcTensorDesc, helpers::getRawPointer(this->_outputs()),
                                        deltaInTensorDesc, helpers::getRawPointer(this->outputErrors()),
                                        &m_zero, deltaOutTensorDesc, helpers::getRawPointer(m_patTmp)) );

        thrust::copy(m_patTmp.begin(), m_patTmp.end(), this->outputErrors().begin());
        // compute the backward pass of the feedforward layer
        FeedForwardLayer<TDevice, TFfActFn>::computeBackwardPass();
    }


    // explicit template instantiations
    // #ifndef NEWSOFTMAX
    template class dnnSoftmaxLayer<Cpu, activation_functions::Identity>;
    // #endif
    template class dnnSoftmaxLayer<Gpu, activation_functions::Identity>;

} // namespace layers
