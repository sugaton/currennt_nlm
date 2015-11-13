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

#include "lmSteepestDescentOptimizer.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../rnnlm/LookupLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../rapidjson/document.h"

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    struct UpdateWeightFn
    {
        real_t learningRate;
        real_t momentum;

        const real_t *weights;
        const real_t *weightUpdates;
        real_t       *weightDeltas;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            // calculate and store the weight delta
            real_t delta = momentum * weightDeltas[weightIdx] - learningRate * weightUpdates[weightIdx];
            weightDeltas[weightIdx] = delta;

            // calculate the new weight
            real_t newWeight = weights[weightIdx] + delta;

            return newWeight;
        }
    };

} // anonymous namespace
} // namespace internal


namespace optimizers {

    template <typename TDevice>
    void lmSteepestDescentOptimizer<TDevice>::_updateWeights()
    {
        internal::UpdateWeightFn updateWeightFn;
        updateWeightFn.momentum     = m_momentum;

        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
            if (i == 1){
                layers::LookupLayer<TDevice> *layer =  dynamic_cast<layers::LookupLayer<TDevice>*>( this->_neuralNetwork().layers()[i].get() );
                if (layer->type() != "lookup") continue;
                //update embeddings in lookup layer
                int w;
                for (int i = 0; i < layer->precedingLayer().intoutputs().size(); ++i){
                    w = layer->precedingLayer().intoutputs()[i];
                    real_vector* emb = layer->embeddings(w, i);  // if embeddings(w, i) is on the gpu memory, emb has the direct raw pointer of embeddings
                                                      // if it is on the cpu memory, emb is the temporal device vector(m_device_vectors.at(i)), thus we should feed back to original space by calling Embedding.replace()
                    updateWeightFn.weights       = helpers::getRawPointer(*emb);
                    updateWeightFn.weightUpdates = helpers::getRawPointer(this->_curWeightUpdates()[1]) + i * layer->size();
                    updateWeightFn.weightDeltas  = helpers::getRawPointer(m_weightDeltas[1]) + i * layer->size();

                    thrust::transform(
                        thrust::counting_iterator<int>(0),
                        thrust::counting_iterator<int>((int)emb->size()),
                        emb->begin(),
                        updateWeightFn
                        );
                  if ( layer->get_emb(w)->type() != std::string(typeid(TDevice).name()) ){
                      layer->get_emb(w)->replace(emb);
                  }
                }
                continue;
            }

          	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
            if (!layer)
                continue;

            updateWeightFn.learningRate = m_learningRate;
            if (layer->learningRate() >= 0.0)
                updateWeightFn.learningRate = layer->learningRate();
            //std::cout << "layer " << layer->name() << ": learning rate " << updateWeightFn.learningRate << std::endl;

            updateWeightFn.weights       = helpers::getRawPointer(layer->weights());
            updateWeightFn.weightUpdates = helpers::getRawPointer(this->_curWeightUpdates()[i]);
            updateWeightFn.weightDeltas  = helpers::getRawPointer(m_weightDeltas[i]);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>((int)layer->weights().size()),
                layer->weights().begin(),
                updateWeightFn
                );
      }
    }

    template <typename TDevice>
    lmSteepestDescentOptimizer<TDevice>::lmSteepestDescentOptimizer(
        NeuralNetwork<TDevice> &neuralNetwork, data_sets::Corpus &trainingSet, data_sets::Corpus &validationSet,
        data_sets::Corpus &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery,
        real_t learningRate, real_t momentum)
        : lmOptimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery)
        , m_learningRate    (learningRate)
        , m_learningRateFirst(learningRate)
        , m_momentum        (momentum)
    {
        // intialize the weight deltas vectors with zeros
        m_weightDeltas = this->_curWeightUpdates();
        for (size_t i = 0; i < m_weightDeltas.size(); ++i)
            thrust::fill(m_weightDeltas[i].begin(), m_weightDeltas[i].end(), 0);
    }

    template <typename TDevice>
    lmSteepestDescentOptimizer<TDevice>::~lmSteepestDescentOptimizer()
    {
    }

    template <typename TDevice>
    void lmSteepestDescentOptimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        lmOptimizer<TDevice>::exportState(jsonDoc);

        lmOptimizer<TDevice>::_exportWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", m_weightDeltas);
    }

    template <typename TDevice>
    void lmSteepestDescentOptimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        lmOptimizer<TDevice>::importState(jsonDoc);

        lmOptimizer<TDevice>::_importWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", &m_weightDeltas);
    }

    template <typename TDevice>
    void lmSteepestDescentOptimizer<TDevice>::setLearningRateFirst(real_t learningRateFirst)
    {
        m_learningRateFirst = learningRateFirst;
    }


    // explicit template instantiations
    template class lmSteepestDescentOptimizer<Cpu>;
    template class lmSteepestDescentOptimizer<Gpu>;

} // namespace optimizers
