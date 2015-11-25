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
#include <thrust/copy.h>
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

    /**
     * Stores the sum of updates into _UpdateSums()
     **/
    template <typename TDevice>
    void lmSteepestDescentOptimizer<TDevice>::_SumUpdates(std::map<int, int> &emb_posi){

        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
              thrust::fill(this->_UpdateSums()[i].begin(), this->_UpdateSums()[i].end(), 0.0);
        }
        int next_posi = 0;
        thrust::plus<real_t> pls;
        for (int device = 0; device < this->_numDevice(); ++device){
            cudaSetDevice(device);
            int N = this->_layersize() * device;
            for (size_t i = 2; i < this->_neuralNetwork().layers().size()-1; ++i) {
                thrust::copy(this->_curWeightUpdates()[i + N].begin(), this->_curWeightUpdates()[i + N].end(), this->_allWeightUpdates()[i].begin());
                thrust::transform(this->_UpdateSums()[i].begin(),
                                  this->_UpdateSums()[i].end(),
                                  this->_allWeightUpdates()[i].begin(),
                                  this->_UpdateSums()[i].begin(),
                                  pls);
            }
            //  for lookup
            thrust::copy(this->_curWeightUpdates()[1 + N].begin(), this->_curWeightUpdates()[1 + N].end(), this->_allWeightUpdates()[1].begin());
            layers::LookupLayer<TDevice> *layer =  dynamic_cast<layers::LookupLayer<TDevice>*>( this->_neuralNetwork().layers(device)[1].get() );
            int d = layer->size();
            for (int i = 0; i < layer->precedingLayer().intoutputs().size(); ++i){
                int w = layer->precedingLayer().intoutputs()[i];
                if (emb_posi.find(w) == emb_posi.end())
                    emb_posi[w] = next_posi++;
                // printf("accessed %d / %d\n", emb_posi[w] * d, d);
                // host
                thrust::transform(this->_UpdateSums()[1].begin() + emb_posi[w] * d,
                                  this->_UpdateSums()[1].begin() + (emb_posi[w] + 1) * d,
                                  this->_allWeightUpdates()[1].begin() + i * d,
                                  this->_UpdateSums()[1].begin() + emb_posi[w] * d,
                                  pls);
            }
        }
    }

    template <typename TDevice>
    void lmSteepestDescentOptimizer<TDevice>::_updateWeightsMultiGpu()
    {
        internal::UpdateWeightFn updateWeightFn;
        updateWeightFn.momentum     = m_momentum;

        std::map<int, int> emb_posi;
        _SumUpdates(emb_posi);

        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
            for (int device = 0; device < this->_numDevice(); ++device){
                cudaSetDevice(device);
                int N = this->_layersize() * device;
                if (i == 1){
                    layers::LookupLayer<TDevice> *layer =  dynamic_cast<layers::LookupLayer<TDevice>*>( this->_neuralNetwork().layers(device)[i].get() );
                    if (layer->type() != "lookup") continue;
                    //update embeddings in lookup layer
                    int w, p;
                    int j = 0;
                    for (auto w_p = emb_posi.begin(); w_p != emb_posi.end(); ++w_p){
                        w = w_p->first;
                        p = w_p->second;
                        real_vector* emb = layer->embeddings(w, j);  // if embeddings(w, i) is on the gpu memory, emb has the direct raw pointer of embeddings
                                                          // if it is on the cpu memory, emb is the temporal device vector(m_device_vectors.at(i)), thus we should feed back to original space by calling Embedding.replace()

                        thrust::copy(this->_UpdateSums()[i].begin() + p * layer->size(),
                                    this->_UpdateSums()[i].begin() + (p+1) * layer->size(),
                                    this->_curWeightUpdates()[1 + N].begin() + j * layer->size());

                        updateWeightFn.weights       = helpers::getRawPointer(*emb);
                        updateWeightFn.weightUpdates = helpers::getRawPointer(this->_curWeightUpdates()[1 + N]) + j * layer->size();
                        updateWeightFn.weightDeltas  = helpers::getRawPointer(m_weightDeltas[1 + N]) + j * layer->size();

                        thrust::transform(
                            thrust::counting_iterator<int>(0),
                            thrust::counting_iterator<int>((int)emb->size()),
                            emb->begin(),
                            updateWeightFn
                            );
                      if ( layer->get_emb(w)->type() != std::string(typeid(TDevice).name()) ){
                          layer->get_emb(w)->replace(emb);
                      }
                      ++j;
                    }
                    continue;
                }

              	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers(device)[i].get());
                if (!layer)
                    continue;
                thrust::copy(this->_UpdateSums()[i].begin(), this->_UpdateSums()[i].end(), this->_curWeightUpdates()[i + N].begin());
                updateWeightFn.learningRate = m_learningRate;
                if (layer->learningRate() >= 0.0)
                    updateWeightFn.learningRate = layer->learningRate();
                //std::cout << "layer " << layer->name() << ": learning rate " << updateWeightFn.learningRate << std::endl;

                updateWeightFn.weights       = helpers::getRawPointer(layer->weights());
                updateWeightFn.weightUpdates = helpers::getRawPointer(this->_curWeightUpdates()[i + N]);
                updateWeightFn.weightDeltas  = helpers::getRawPointer(m_weightDeltas[i + N]);

                thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>((int)layer->weights().size()),
                    layer->weights().begin(),
                    updateWeightFn
                    );
            }
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
        // TODO create Deltas for all embedding
        // m_weightDeltas = this->_curWeightUpdates();
        m_weightDeltas.resize(this->_curWeightUpdates().size());
        for (int device = 0; device < this->_numDevice(); ++device){
            cudaSetDevice(device);
            int N = this->_layersize() * device;
            for (size_t i = 0; i < this->_layersize(); ++i){
                m_weightDeltas[i + N] = this->_curWeightUpdates()[i + N];
                thrust::fill(m_weightDeltas[i + N].begin(), m_weightDeltas[i + N].end(), 0);
            }
        }
    }

    template <typename TDevice>
    lmSteepestDescentOptimizer<TDevice>::~lmSteepestDescentOptimizer()
    {
        for ( int device = 0; device < this->_numDevice(); ++device){
            cudaSetDevice(device);
            for (size_t i = 1; i < this->_layersize()-1; ++i) {
                this->_curWeightUpdates()[device * this->_layersize() + i].clear();
                this->_curWeightUpdates()[device * this->_layersize() + i].shrink_to_fit();
            }
        }
        this->_curWeightUpdates().clear();
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
