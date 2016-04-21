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

#include "ComparingLayerClassificationLayer.hpp"
#include "../helpers/NumericLimits.cuh"
#include "../helpers/max.cuh"
#include "../helpers/getRawPointer.cuh"

#include <stdexcept>
#include <cassert>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#define SKIP_MARKER helpers::NumericLimits<real_t>::max()


namespace internal {
namespace {

    struct ComputeCrossEntropyErrorFn
    {
        int layerSize;

        const real_t *outputs;

        __host__ __device__ real_t operator() (const thrust::tuple<int, int> &t) const
        {
            // unpack the tuple
            int targetClass = t.get<0>();
            int patIdx      = t.get<1>();

            // calculate the CEE
            if (targetClass == -1)
                return 0;
            else {
                int outputIdx     = outputIdx = patIdx * layerSize + targetClass;
                real_t targetProb = helpers::max(helpers::NumericLimits<real_t>::min(), outputs[outputIdx]);
                return log(targetProb);
            }
        }
    };


    struct ComputeEntropyFn
    {
        int layerSize;

        const real_t *outputs;

        __host__ __device__ real_t operator() (const thrust::tuple<int, int> &t) const
        {
            // unpack the tuple
            int targetClass = t.get<0>();
            int patIdx      = t.get<1>();

            // calculate the CEE
            if (targetClass == -1)
                return 0;
            else {
                int outputIdx     = outputIdx = patIdx * layerSize + targetClass;
                real_t targetProb = helpers::max(helpers::NumericLimits<real_t>::min(), outputs[outputIdx]);
                return targetProb * log2(targetProb);
            }
        }
    };

    struct CountCorrectClassificationsFn
    {
        int layerSize;

        const real_t *outputs;

        __host__ __device__ int operator() (const thrust::tuple<int, int> &t) const
        {
            // unpack the tuple
            int targetClass = t.get<0>();
            int patIdx      = t.get<1>();

            // check for dummy
            if (targetClass == -1)
                return 0;

            // determine the estimated target class
            const real_t *offOutputs = outputs + patIdx * layerSize;
            real_t maxProb = 0;
            int estClass   = 0;

            for (int i = 0; i < layerSize; ++i) {
                real_t out = offOutputs[i];
                if (out > maxProb) {
                    maxProb  = out;
                    estClass = i;
                }
            }

            // check if the we correctly classified the timestep
            if (targetClass == estClass)
                return 1;
            else
                return 0;
        }
    };


    struct ComputeOutputErrorFn
    {
        int layerSize;

        const real_t *outputs;
        const real_t *scores;
        real_t       *outputErrors;
        const int *targetClasses;
        const int *candidate;
        const real_t Z;
        const real_t answersScore;


        // __host__ __device__ void operator() (const thrust::tuple<int, int> &t) const
        __device__ void operator() (const int &t) const
        {
            // unpack the tuple
            int patIdx      = t / layersize;
            int layerIdx    = t % layersize;
            int targetClass = targetClasses[patIdx];
            int flag        = candidate[patIdx];
            int tag         = targetClasses[patIdx];
            // check if we need to continue
            if (flag == 0)
                return;

            if (tag == -1){
                outputErrors[t] = (- answersScore) * outputs[layerIdx]  / (Z * Z);
                atomicAdd(outputErrors+layerIdx, (- answersScore) * outputs[t] / (Z * Z));
                return;
            }
            else{
                outputErrors[t] = (Z - scores[patIdx]) * outputs[layerIdx] / (Z * Z);
                atomicAdd(outputErrors+layerIdx, (Z - scores[patIdx]) * outputs[t] / (Z * Z));
            }
        }
    };

    // template <typename ScoreFn>
    struct Dot
    {
        int size;
        real_t *layers;
        real_t *work;
        real_t *output;
        __device__ void operator() (const int idx) const
        {
            int j = idx % size;
            int offset = idx / size;
            atommicAdd(output + offset, layers[j] * layers[idx]);
        }
        __host__ void operator() (const int idx) const
        {
            int j = idx % size;
            int offset = idx / size;
            output[offset] += layers[j] * layers[idx];
        }
    };
    /*
    struct Dot
    {
        real_t fn(real_t *x, real_t *y, real_t *z, int size)
        {
            thrust::transform(x,
                              x + size,
                              y,
                              z,
                              thrust::multiplies<real_t>());
            return thrust::reduce(z, z + size);
        }
    };
    */
    struct ArgMax
    {
        real_t *array;
        __host__ __device__ int operator() (const int &lidx, const int &ridx) const
        {
            return (array[lidx] >= array[ridx])? lidx : ridx;  //precede left element
        }
    };

} // anonymous namespace
} // namespace anonymous


namespace layers {

    template <typename TDevice>
    ComparingLayerClassificationLayer<TDevice>::ComparingLayerClassificationLayer(const helpers::JsonValue &layerChild, Layer<TDevice> &precedingLayer)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer, precedingLayer.size(), false)
    {
        if (this->size() == 1)
            throw std::runtime_error("The ComparingLayer classification post output layer cannot be used for an output layer size of 1");

        // resize the pattern target classes vector
        m_patTargetClasses.resize(this->patTypes().size());
    }

    template <typename TDevice>
    ComparingLayerClassificationLayer<TDevice>::~ComparingLayerClassificationLayer()
    {
    }

    template <typename TDevice>
    int ComparingLayerClassificationLayer<TDevice>::countCorrectClassifications()
    {
        internal::CountCorrectClassificationsFn fn;
        fn.layerSize = this->size();
        fn.outputs   = helpers::getRawPointer(this->_actualOutputs());

        int n = this->curMaxSeqLength() * this->parallelSequences();

        int correctClassifications = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn,
            0,
            thrust::plus<int>()
            );

        return correctClassifications;
    }

    template <typename TDevice>
    const std::string& ComparingLayerClassificationLayer<TDevice>::type() const
    {
        static std::string s("ComparingLayer_classification");
        return s;
    }

    template <typename TDevice>
    void ComparingLayerClassificationLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        PostOutputLayer<TDevice>::loadSequences(fraction);

        thrust::copy(fraction.targetClasses().begin(), fraction.targetClasses().end(), m_patTargetClasses.begin());
        thrust::copy(fraction.targetClasses().begin(), fraction.targetClasses().end(), m_targetClasses_cpu.begin());
        bool ifstart = false;
        for (int i = 0; i < m_targetClasses_cpu.size(); ++i){
            if (m_targetClasses_cpu[i] == 0 && i != 0){
                if (ifstart){
                    m_docRangeEnd = i;
                    break;
                }
                else{
                    m_docRangeBegin = i + 1;
                    ifstart = true;
                }
            }
        }
        thrust::fill(m_candidate_cpu.begin(), m_candidate_cpu.end(), 0);
    }

    template <typename TDevice>
    real_t ComparingLayerClassificationLayer<TDevice>::calculateError()
    {
        // calculate the cross entropy error
        internal::ComputeCrossEntropyErrorFn fn;
        fn.layerSize = this->size();
        fn.outputs   = helpers::getRawPointer(this->_actualOutputs());

        int n = this->curMaxSeqLength() * this->parallelSequences();

        real_t error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn,
            (real_t)0,
            thrust::plus<real_t>()
            );

        return -error;
    }


    template <typename TDevice>
    real_t ComparingLayerClassificationLayer<TDevice>::calculateEntropy()
    {
        // calculate the cross entropy error
        internal::ComputeEntropyFn fn;
        fn.layerSize = this->size();
        fn.outputs   = helpers::getRawPointer(this->_actualOutputs());

        int n = this->curMaxSeqLength() * this->parallelSequences();

        real_t error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(m_patTargetClasses.begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn,
            (real_t)0,
            thrust::plus<real_t>()
            );

        return -error;
    }

    template <typename TDevice>
    void ComparingLayerClassificationLayer<TDevice>::computeForwardPass()
    {
        //copying state
        for (int i = 0; i < m_docRangeBegin - 1; ++i){
            if (i == 0)
                thrust::copy(this->_actualOutputs().begin() + m_targetClasses_cpu[0] * this->size(),
                             this->_actualOutputs().begin() + (m_targetClasses_cpu[0] + 1) * this->size(),
                             m_CompLayers.begin());
            else{
                if (m_comptype == 1){ // always comparing last state layer
                    int current_layer = ((i + 1) * this->curMaxSeqLength - 1);
                    thrust::copy(this->_actualOutputs().begin() +  current_layer * this->size(),
                                 this->_actualOutputs().begin() + (current_layer + 1) * this->size(),
                                 m_CompLayers.begin() + i * this->size());
                }
                // TODO copy all state if m_comptype == 2
            }
        }

        // calculating Dot score
        internal:internal:Dot fn;
        fn.layers = m_CompLayers.data().get();
        fn.work = m_CompLayer_work.data().get();
        fn.output = m_compscores.data().get();

        thrust::for_each(thrust::counting_iterator<int>(0) + this->size(),
                         thrust::counting_iterator<int>(0) + this->size() * this->curMaxSeqLength() * this->parallelSequences(),
                          fn);

        // normalization
        m_Z = thrust::reduce(m_compscores.begin(), m_compscores.end());
        // thrust::transform(m_compscores.begin(),
        //                   m_compscores.end(),
        //                   thrust::make_constant_iterator(Z),
        //                   m_compscores.begin(),
        //                   thrust::divides<float>());

        // Finding each of the states that has maximum score in each documentation.
        int _begin, _end, cand;
        internal::ArgMax argmax;
        argmax.array = m_compscores.data();

        for (int i = m_docRangeBegin; i < m_docRangeEnd - 1; ++i){
            _begin = m_targetClasses_cpu[i];
            _end = m_targetClasses_cpu[i+1];
            cand = thrust::reduce(thrust::counting_iterator<int>(0) + _begin,
                                  thrust::counting_iterator<int>(0) + _end,
                                  -1,
                                  argmax);
            m_candidate_cpu.at(cand) = 1;
        }
        m_candidate = m_candidate_cpu;
    }

    template <typename TDevice>
    void ComparingLayerClassificationLayer<TDevice>::computeBackwardPass()
    {
        int n = this->curMaxSeqLength() * this->parallelSequences();
        int m = m_candidate_cpu.size();

        // set all errors to zero
        assert (n * this->size() <= this->_outputErrors().size());
        thrust::fill_n(this->_outputErrors().begin(), n * this->size(), (real_t)0);

        // calculate the errors
        internal::ComputeOutputErrorFn fn;
        fn.layerSize     = this->size();
        fn.outputs       = helpers::getRawPointer(this->_actualOutputs());
        fn.scores        = helpers::getRawPointer(m_compscores);
        fn.outputErrors  = helpers::getRawPointer(this->_outputErrors());
        fn.targetClasses = helpers::getRawPointer(m_patTargetClasses);
        fn.candidate     = helpers::getRawPointer(m_candidate);
        fn.Z             = m_Z;
        fn.answersScore  = m_Z;

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(m_candidate.begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(m_candidate.begin()+n, thrust::counting_iterator<int>(0)+n*this->size())),
            fn
            );
    }


    // explicit template instantiations
    // template class ComparingLayerClassificationLayer<Cpu>;
    template class ComparingLayerClassificationLayer<Gpu>;

} // namespace layers
