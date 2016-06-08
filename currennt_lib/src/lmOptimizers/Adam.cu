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

#include "Adam.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../rnnlm/LookupLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../rapidjson/document.h"

#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>


namespace internal {
namespace {

    struct UpdateWeightFn
    {
        real_t beta1;
        real_t beta2;
        real_t eps;
        real_t alpha;
        real_t beta1t;
        real_t beta2t;

        real_t *m;
        real_t *v;



        const real_t *theta;//weights;
        const real_t *g; //weightUpdates;
        // real_t       *weightDeltas;

        __host__ __device__ real_t operator() (const int &i)
        {
            // calculate and store the weight delta
            // real_t delta = - learningRate * weightUpdates[weightIdx];
            // weightDeltas[weightIdx] = delta;
            m[i] = beta1 * (m[i]) + (1 - beta1) * g[i];
            v[i] = beta2 * (v[i]) + (1 - beta2) * g[i] * g[i];
            //*beta1t *= beta1;
            //*beta2t *= beta2;

            return theta[i] - ( alpha / (1 - beta1t)) * (m[i] / (sqrtf(v[i] / (1 - beta2t)) + eps));
            // calculate the new weight
            //return theta[i] + delta;

            //return newWeight;
        }
    };

} // anonymous namespace
} // namespace internal


namespace optimizers {

    template <typename TDevice>
    void Adam<TDevice>::_updateWeights()
    {
        internal::UpdateWeightFn updateWeightFn;

        updateWeightFn.beta1         = m_beta1;
        updateWeightFn.beta2         = m_beta2;
        updateWeightFn.eps           = m_eps;
        updateWeightFn.alpha         = m_learningRate;
        int parallelSequences = this->_neuralNetwork().postOutputLayer().parallelSequences();

        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
            if (i == 1){
                layers::LookupLayer<TDevice> *layer =  dynamic_cast<layers::LookupLayer<TDevice>*>( this->_neuralNetwork().layers()[i].get() );
                if (layer->type() != "lookup")
                    continue;
                if (layer->fixed())
                    continue;
                //update embeddings in lookup layer
                int w;
                for (int j = 0; j < layer->precedingLayer().intoutputs().size(); ++j){
                    w = layer->precedingLayer().intoutputs()[j];
                    real_vector* emb = layer->embeddings(w, j);  // if embeddings(w, i) is on the gpu memory, emb has the direct raw pointer of embeddings
                                                      // if it is on the cpu memory, emb is the temporal device vector(m_device_vectors.at(i)), thus we should feed back to original space by calling Embedding.replace()

                    updateWeightFn.beta1t        = m_beta1t_emb[w];
                    updateWeightFn.beta2t        = m_beta2t_emb[w];
                    updateWeightFn.m             = m_moment[0] + m_MLookupStart + w * layer->size(); //pointer
                    updateWeightFn.v             = m_second_moment[0] + m_MLookupStart + w * layer->size(); //pointer
                    updateWeightFn.theta         = helpers::getRawPointer(*emb);
                    updateWeightFn.g             = helpers::getRawPointer(this->_curWeightUpdates()[1]) + j * layer->size();

                    thrust::transform(
                        thrust::counting_iterator<int>(0),
                        thrust::counting_iterator<int>((int)emb->size()),
                        emb->begin(),
                        updateWeightFn
                        );
                  if ( layer->get_emb(w)->type() != std::string(typeid(TDevice).name()) ){
                      layer->get_emb(w)->replace(emb);
                  }
                  m_beta1t_emb[w] *= m_beta1t;
                  m_beta2t_emb[w] *= m_beta2t;
                }
                continue;
            }

          	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
            if (!layer)
                continue;

            /*
            thrust::transform(this->_curWeightUpdates()[i].begin(),
                              this->_curWeightUpdates()[i].end(),
                              thrust::constant_iterator<real_t>((real_t)(parallelSequences)),
                              this->_curWeightUpdates()[i].begin(),
                              thrust::divides<real_t>());
                              // */
            //std::cout << "layer " << layer->name() << ": learning rate " << updateWeightFn.learningRate << std::endl;

            updateWeightFn.beta1t        = m_beta1t;
            updateWeightFn.beta2t        = m_beta2t;
            updateWeightFn.m             = m_moment[0] + m_MStart[i]; //pointer
            updateWeightFn.v             = m_second_moment[0] + m_MStart[i]; //pointer
            updateWeightFn.theta         = helpers::getRawPointer(layer->weights());
            updateWeightFn.g             = helpers::getRawPointer(this->_curWeightUpdates()[i]);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>((int)layer->weights().size()),
                layer->weights().begin(),
                updateWeightFn
                );
        }
        if (m_t_ < m_tlimit) {
            m_beta1t *= m_beta1;
            m_beta2t *= m_beta2;
            ++m_t_;
        }
        else
            printf("m_beta**t: %d %f %f\n", m_t_, m_beta1t, m_beta2t);
    }

    /**
     * Stores the sum of updates into _UpdateSums()
     **/
    template <typename TDevice>
    void Adam<TDevice>::_SumUpdates(std::map<int, int> &emb_posi){

        int parallelSequences = this->_neuralNetwork().postOutputLayer().parallelSequences();

        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
              thrust::fill(this->_UpdateSums()[i].begin(), this->_UpdateSums()[i].end(), 0.0);
        }
        int next_posi = 0;
        int d;
        bool lookupFixed = false;
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
            layers::LookupLayer<TDevice> *layer =  dynamic_cast<layers::LookupLayer<TDevice>*>( this->_neuralNetwork().layers(device)[1].get() );
            if (layer->fixed()){
                lookupFixed = true;
                continue;
            }
            thrust::copy(this->_curWeightUpdates()[1 + N].begin(), this->_curWeightUpdates()[1 + N].end(), this->_allWeightUpdates()[1].begin());
            d = layer->size();
            for (int i = 0; i < layer->precedingLayer().intoutputs().size(); ++i){
                int w = layer->precedingLayer().intoutputs()[i];
                if (emb_posi.find(w) == emb_posi.end())
                    emb_posi[w] = next_posi++;
                // host
                thrust::transform(this->_UpdateSums()[1].begin() + emb_posi[w] * d,
                                  this->_UpdateSums()[1].begin() + (emb_posi[w] + 1) * d,
                                  this->_allWeightUpdates()[1].begin() + i * d,
                                  this->_UpdateSums()[1].begin() + emb_posi[w] * d,
                                  pls);
            }
        }
        // waiting calculating sum
        cudaDeviceSynchronize();
        for (int device = 0; device < this->_numDevice(); ++device){
            cudaSetDevice(device);
            int N = this->_layersize() * device;
            for (size_t i = 2; i < this->_neuralNetwork().layers().size()-1; ++i) {
                //average weights
                thrust::copy(this->_UpdateSums()[i].begin(),
                             this->_UpdateSums()[i].end(),
                             this->_curWeightUpdates()[i + N].begin());
                            //  /*
                thrust::transform(
                    this->_curWeightUpdates()[i + N].begin(),
                    this->_curWeightUpdates()[i + N].end(),
                    thrust::constant_iterator<real_t>((real_t)(parallelSequences * this->_numDevice())),
                    this->_curWeightUpdates()[i + N].begin(),
                    thrust::divides<real_t>()
                );
                    // */
            }
        }
        if (lookupFixed)
            return;
        int p;
        for (int device = 0; device < this->_numDevice(); ++device){
            cudaSetDevice(device);
            int N = this->_layersize() * device;
            int j = 0;
            for (auto w_p = emb_posi.begin(); w_p != emb_posi.end(); ++w_p){
                p = w_p->second;
                thrust::copy(this->_UpdateSums()[1].begin() + p * d,
                             this->_UpdateSums()[1].begin() + (p+1) * d,
                             this->_curWeightUpdates()[1 + N].begin() + j * d);
            }
        }


    }

    template <typename TDevice>
    void Adam<TDevice>::_updateWeightsMultiGpu()
    {
        internal::UpdateWeightFn updateWeightFn;

        updateWeightFn.beta1         = m_beta1;
        updateWeightFn.beta2         = m_beta2;
        updateWeightFn.eps           = m_eps;
        updateWeightFn.alpha         = m_learningRate;


        std::map<int, int> emb_posi;
        _SumUpdates(emb_posi);
        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
            for (int device = 0; device < this->_numDevice(); ++device){
                cudaSetDevice(device);
                int N = this->_layersize() * device;
                if (i == 1){
                    layers::LookupLayer<TDevice> *layer =  dynamic_cast<layers::LookupLayer<TDevice>*>( this->_neuralNetwork().layers(device)[i].get() );
                    int max_length = layer->parallelSequences() * layer->maxSeqLength();
                    if (layer->type() != "lookup") continue;
                    if (layer->fixed())
                        continue;
                    //update embeddings in lookup layer
                    int w, p;
                    int j = 0;
                    for (auto w_p = emb_posi.begin(); w_p != emb_posi.end(); ++w_p){
                        w = w_p->first;
                        p = w_p->second;
                        real_vector* emb = layer->embeddings(w, j);  // if embeddings(w, i) is on the gpu memory, emb has the direct raw pointer of embeddings
                                                          // if it is on the cpu memory, emb is the temporal device vector(m_device_vectors.at(i)), thus we should feed back to original space by calling Embedding.replace()

                        /* this should be done before here.
                           if this is done here, next thrust::transform will wait the end of this copying.
                        thrust::copy(this->_UpdateSums()[i].begin() + p * layer->size(),
                                    this->_UpdateSums()[i].begin() + (p+1) * layer->size(),
                                    this->_curWeightUpdates()[1 + N].begin() + j * layer->size());
                        */


                        updateWeightFn.beta1t        = m_beta1t_emb[w];
                        updateWeightFn.beta2t        = m_beta2t_emb[w];
                        updateWeightFn.m             = (m_moment[device] + m_MLookupStart + w * layer->size()); //pointer
                        updateWeightFn.v             = (m_second_moment[device] + m_MLookupStart + w * layer->size()); //pointer
                        updateWeightFn.theta         = helpers::getRawPointer(*emb);
                        updateWeightFn.g             = helpers::getRawPointer(this->_curWeightUpdates()[1 + N]) + j * layer->size();

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
                      if (j >= max_length) j = 0;
                      m_beta1t_emb[w] *= m_beta1t;
                      m_beta2t_emb[w] *= m_beta2t;
                    }
                    continue;
                }


                layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers(device)[i].get());
                if (!layer)
                    continue;
                 /* this should be done before here.
                    if this is done here, next thrust::transform will wait the end of this copying.
                thrust::copy(this->_UpdateSums()[i].begin(), this->_UpdateSums()[i].end(), this->_curWeightUpdates()[i + N].begin());
                */
                updateWeightFn.alpha = m_learningRate;
                if (layer->learningRate() >= 0.0)
                    updateWeightFn.alpha = layer->learningRate();
                //std::cout << "layer " << layer->name() << ": learning rate " << updateWeightFn.learningRate << std::endl;

                updateWeightFn.beta1t        = m_beta1t;
                updateWeightFn.beta2t        = m_beta2t;
                updateWeightFn.m             = m_moment[device] + m_MStart[i]; //pointer
                updateWeightFn.v             = m_second_moment[device] + m_MStart[i]; //pointer
                updateWeightFn.theta         = helpers::getRawPointer(layer->weights());
                updateWeightFn.g             = helpers::getRawPointer(this->_curWeightUpdates()[i + N]);

                thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>((int)layer->weights().size()),
                    layer->weights().begin(),
                    updateWeightFn
                    );
            }
        }
        if (m_t_ < m_tlimit) {
            m_beta1t *= m_beta1;
            m_beta2t *= m_beta2;
            m_t_ += 1;
        }
        else
            printf("m_beta**t: %d %f %f\n", m_t_, m_beta1t, m_beta2t);

    }


    template <typename TDevice>
    Adam<TDevice>::Adam(
        NeuralNetwork<TDevice> &neuralNetwork, data_sets::Corpus &trainingSet, data_sets::Corpus &validationSet,
        data_sets::Corpus &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery,
        real_t learningRate, int tmp_show, real_t beta1, real_t beta2, real_t eps)
        : lmOptimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery, tmp_show)
        , m_learningRate    (learningRate)
        , m_learningRateFirst(learningRate)
        , m_beta1        (beta1)
        , m_beta2        (beta2)
        , m_eps        (eps)
        , m_tlimit       (3000)
        , m_t_            (0)
    {
        // intialize the weight deltas vectors with zeros
        // TODO create Deltas for all embedding
        // m_weightDeltas = this->_curWeightUpdates();
        // m_weightDeltas.resize(this->_curWeightUpdates().size());
        size_t weightsize = 0;
        size_t wordnum = 0;
        size_t lookupsize = 0;
        m_MStart = std::vector<size_t>(this->_layersize(), 0);

        for (size_t i = 0; i < this->_layersize(); ++i){
            m_MStart[i] = weightsize;
            if (i == 1) {
                layers::LookupLayer<TDevice> *layer =  dynamic_cast<layers::LookupLayer<TDevice>*>( this->_neuralNetwork().layers()[i].get() );
                if (!layer) continue;
                lookupsize = layer->size() * layer->lookupSize();
                wordnum = layer->lookupSize();
                continue;
            }
          	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
            if (!layer) continue;
            weightsize += layer->weights().size();
            // m_weightDeltas[i + N] = this->_curWeightUpdates()[i + N];
            // thrust::fill(m_weightDeltas[i + N].begin(), m_weightDeltas[i + N].end(), 0);
        }
        // allocate momentum arrays
        m_MLookupStart = weightsize;

        m_momentum_arr.resize(this->_numDevice());
        m_2momentum_arr.resize(this->_numDevice());
        m_moment.resize(this->_numDevice());
        m_second_moment.resize(this->_numDevice());

        for (int device = 0; device < this->_numDevice(); ++device){
            cudaSetDevice(device);
            int N = this->_layersize() * device;

            // m_momentum_arr[device] = real_vector(weightsize + lookupsize, 0);
            m_momentum_arr[device].resize(weightsize + lookupsize);
            thrust::fill(m_momentum_arr[device].begin(), m_momentum_arr[device].end(), 0);
            // m_2momentum_arr[device] = real_vector(weightsize + lookupsize, 0);
            m_2momentum_arr[device].resize(weightsize + lookupsize);
            thrust::fill(m_2momentum_arr[device].begin(), m_2momentum_arr[device].end(), 0);

            m_moment[device] = helpers::getRawPointer(m_momentum_arr[device]);
            m_second_moment[device] = helpers::getRawPointer(m_2momentum_arr[device]);
        }
        m_beta1t = m_beta1;
        m_beta1t_emb = Cpu::real_vector(wordnum , m_beta1);
        m_beta2t = m_beta2;
        m_beta2t_emb = Cpu::real_vector(wordnum , m_beta2);

    }

    template <typename TDevice>
    Adam<TDevice>::~Adam()
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
    void Adam<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        lmOptimizer<TDevice>::exportState(jsonDoc);

        lmOptimizer<TDevice>::_exportWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", m_weightDeltas);
    }

    template <typename TDevice>
    void Adam<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        lmOptimizer<TDevice>::importState(jsonDoc);

        lmOptimizer<TDevice>::_importWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", &m_weightDeltas);
    }

    template <typename TDevice>
    void Adam<TDevice>::setLearningRateFirst(real_t learningRateFirst)
    {
        m_learningRateFirst = learningRateFirst;
    }


    // explicit template instantiations
    template class Adam<Cpu>;
    template class Adam<Gpu>;

} // namespace optimizers
