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

#include "lmOptimizer.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../layers/BinaryClassificationLayer.hpp"
#include "../layers/MulticlassClassificationLayer.hpp"
#include "../Configuration.hpp"
#include "../helpers/JsonClasses.hpp"

#include "../rnnlm/LookupLayer.hpp"

#include <limits>
#include <set>
#include <math.h>

#include <thrust/transform.h>
#include <boost/filesystem.hpp>


#ifdef _MYMPI

#include <mpi.h>

#endif

namespace internal{

    // return status, 0: successed 1:partially failed　2: completely failed
    int getMultiFraction(data_sets::Corpus &ds, int size, std::vector<boost::shared_ptr<data_sets::CorpusFraction>>* vec)
    {
        int ret = 0;
        boost::shared_ptr<data_sets::CorpusFraction> frac;
        for (int i = 0; i < size; ++i){
            if ((frac = ds.getNextFraction()))
              vec->at(i) = frac;
            else if (i == 0) return 2;
            else{
              vec->at(i) = vec->at(0);
              ret = 1;
            }
        }
        return ret;
    }

    void showProgress(int curEpoch, real_t error, real_t progress){
        printf("\n %5d | on training:\t %8.1f  progress: %f%%", curEpoch, (float)error, (float)progress * 100);
        fflush(stdout);
    }
    void refreshLine(int curEpoch){ printf("\r %5d | ", curEpoch); }

    template <typename T>
    struct _uDivide
    {
        T x;
        __host__ __device__ real_t operator() (const T& y)
        {
            return y / x;
        }
    };
}

namespace optimizers {

#ifdef _MYMPI
    template <typename TDevice>
    void lmOptimizer<TDevice>::_syncWeight()
    {
        // copying to host vector
        std::set<int> syncSet;
        cudaSetDevice(0);
        for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
            if (i == 1) {
              	layers::LookupLayer<TDevice> *_layer = dynamic_cast<layers::LookupLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
                if (_layer->fixed())
                    continue;
                //hoge hoge
                continue;
            }
          	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (!layer)
                continue;
            syncSet.insert(i);
            // copying
            thrust::copy(
                layer->weights().begin(),
                layer->weights().end(),
                m_allWeightUpdates[i].begin()
            );
        }

        // do all_reduce and calculate mean of weight.
        // result is stored in m_UpdateSums
        internal::_uDivide<real_t> devide_by_procs;
        devide_by_procs.x = (real_t)MPI::COMM_WORLD.Get_size();
        for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
            if (syncSet.find(i) == syncSet.end())
                continue;
            real_t* arr = thrust::raw_pointer_cast(m_allWeightUpdates[i].data());
            real_t* result = thrust::raw_pointer_cast(m_UpdateSums[i].data());
            // sum
            MPI::COMM_WORLD.Allreduce(
                (void*) arr,     // sendbuf
                (void*) result,  // recvbuf
                (int) m_allWeightUpdates[i].size(),
                MPI_FLOAT,
                MPI_SUM //,
                // MPI_COMM_WORLD
            );
            // calc mean of weight
            thrust::transform(
                m_UpdateSums[i].begin(),
                m_UpdateSums[i].end(),
                m_UpdateSums[i].begin(),
                devide_by_procs
            );
        }

        // copying result to device vector
        for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
            if (syncSet.find(i) == syncSet.end())
                continue;
            for (int device = 0; device < m_numDevice; ++device) {
                cudaSetDevice(device);
              	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers(device)[i].get());
                if(!layer)
                    continue;
                thrust:copy(
                    m_UpdateSums[i].begin(),
                    m_UpdateSums[i].end(),
                    layer->weights().begin()
                );
            }
        }
    }
#endif

    template <typename TDevice>
    void lmOptimizer<TDevice>::_resetWeightUpdates()
    {
        for (int device = 0; device < m_numDevice; ++device){
            cudaSetDevice(device);
            int N = device * m_layer_size;
            int dummy = 0;
            {
                layers::LookupLayer<TDevice> *layer = dynamic_cast<layers::LookupLayer<TDevice>*>(m_neuralNetwork.layers(device)[1].get());
                if (!layer || layer->fixed())
                    dummy = 1;
                else
                	thrust::fill(m_curWeightUpdates[N + 1].begin(), m_curWeightUpdates[N + 1].end(), 0.0);

            }

            for (size_t i = 2; i < m_neuralNetwork.layers().size()-1; ++i) {
                // int N = device * m_layer_size;
                layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers(device)[i].get());
                if (!layer)
                    continue;
              	thrust::fill(m_curWeightUpdates[N + i].begin(), m_curWeightUpdates[N + i].end(), 0.0);
            }

        }
    }
    template <typename TDevice>
    real_t lmOptimizer<TDevice>::_processDataSet(data_sets::Corpus &ds, bool calcWeightUpdates, real_t *classError)
    {
        // process all data set fractions
        real_t error = 0;
        *classError = (real_t) ds.totalTimesteps();

        int consume_sequences = 0;

        std::vector< boost::shared_ptr<data_sets::CorpusFraction> > fracs;
        fracs.resize(m_numDevice);
        bool firstFraction = true;
        int parallelSequences = m_neuralNetwork.postOutputLayer().parallelSequences();

        int loop_count = 0;
        int processed_count = 0;
        while (true) {

            if (processed_count == 0)
                firstFraction = true;
            int status = internal::getMultiFraction(ds, m_numDevice, &fracs);
            if (status == 2) break;
            // compute forward pass and calculate the error

            for (int i = 0; i < m_numDevice; ++i)
                m_neuralNetwork.loadSequences(*(fracs[i]), i);

            consume_sequences += m_numDevice * parallelSequences;
            m_neuralNetwork.computeForwardPass();

            if (!m_errorType) // log_prob
                for (int device = 0; device < m_numDevice; ++device)
                    error += m_neuralNetwork.calculateError(device);
            else  // entropy
                for (int device = 0; device < m_numDevice; ++device)
                    error += m_neuralNetwork.calculateError(device);

            if (dynamic_cast<layers::BinaryClassificationLayer<TDevice>*>(&m_neuralNetwork.postOutputLayer()))
                for (int device = 0; device < m_numDevice; ++device){
                    *classError -= (real_t)static_cast<layers::BinaryClassificationLayer<TDevice>&>(m_neuralNetwork.postOutputLayer(device)).countCorrectClassifications();
                }
            if (dynamic_cast<layers::MulticlassClassificationLayer<TDevice>*>(&m_neuralNetwork.postOutputLayer()))
                for (int device = 0; device < m_numDevice; ++device){
                    *classError -= (real_t)static_cast<layers::MulticlassClassificationLayer<TDevice>&>(m_neuralNetwork.postOutputLayer(device)).countCorrectClassifications();
                }


            if (calcWeightUpdates) {
                // weight noise:
                // std::vector<Cpu::real_vector> origWeights(m_neuralNetwork.layers().size());
                if (Configuration::instance().weightNoiseSigma() > 0) {
                    for (int device = 0; device < m_numDevice; ++device){
                        cudaSetDevice(device);
                        int N = device * m_layer_size;
                        for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
                            layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers(device)[i].get());
                            if (layer) {
                                // origWeights[N+i] = layer->weights();
                                thrust::copy(
                                    layer->weights().begin(),
                                    layer->weights().end(),
                                    origWeights[N + i].begin()
                                );
                                layer->injectWeightNoise(Configuration::instance().weightNoiseSigma());
                            }
                        }
                    }
                }
                // compute the backward pass and accumulate the weight updates to layer->weightUpdates()
                m_neuralNetwork.computeBackwardPass();



                // case lookup-layer (i = 1)
                for (int device = 0; device < m_numDevice; ++device){
                    cudaSetDevice(device);
                    int N = device * m_layer_size;
                    int dummy = 0;
                    {
                    // for (int device = 0; device < m_numDevice; ++device){
                        // cudaSetDevice(device);
                        // int N = device * m_layer_size;
                        layers::LookupLayer<TDevice> *layer = dynamic_cast<layers::LookupLayer<TDevice>*>(m_neuralNetwork.layers(device)[1].get());
                        if (!layer || layer->fixed())
                            dummy = 1;
                        else if (!firstFraction && (!Configuration::instance().hybridOnlineBatch() || m_numDevice > 1))
                            thrust::transform(layer->weightUpdates().begin(), layer->weightUpdates().end(), m_curWeightUpdates[N + 1].begin(), m_curWeightUpdates[N + 1].begin(), thrust::plus<real_t>());
                        else
                        	thrust::copy(layer->weightUpdates().begin(), layer->weightUpdates().end(), m_curWeightUpdates[N + 1].begin());

                    }

                // }


                    for (size_t i = 2; i < m_neuralNetwork.layers().size()-1; ++i) {
                        // int N = device * m_layer_size;
                        layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers(device)[i].get());
                        if (!layer)
                            continue;

                        // if (!firstFraction && !Configuration::instance().hybridOnlineBatch())
                        if (!firstFraction && (!Configuration::instance().hybridOnlineBatch() || m_numDevice > 1))
                            thrust::transform(layer->weightUpdates().begin(), layer->weightUpdates().end(), m_curWeightUpdates[N + i].begin(), m_curWeightUpdates[N + i].begin(), thrust::plus<real_t>());
                        else
                        	thrust::copy(layer->weightUpdates().begin(), layer->weightUpdates().end(), m_curWeightUpdates[N + i].begin());

                        // restore old weights before update in case of weight noise
                        if (Configuration::instance().weightNoiseSigma() > 0.0)
                            thrust::copy(origWeights[N + i].begin(), origWeights[N + i].end(), layer->weights().begin());

                    }

                }
                // update weights for hybrid online/batch learning
                if (Configuration::instance().hybridOnlineBatch()){
                    if (m_numDevice == 1) _updateWeights();
                    else if (processed_count == m_multi_sync_pace){
                        _updateWeightsMultiGpu();
                        processed_count = 0;
                        //_resetWeightUpdates();
                    }
                    else {
                        ++processed_count;

                    }
                }
            }

            firstFraction = false;
            if (status == 1) break;
            ++loop_count;
            if(m_tmp_show > 0 && consume_sequences % m_tmp_show ==0)
                internal::showProgress(m_curEpoch, error / consume_sequences, (float)consume_sequences / (float)ds.totalSequences());
            time_point now =  std::chrono::system_clock::now();

            auto spend_time = now - m_start_time;
            auto hour = std::chrono::duration_cast<std::chrono::hours>(spend_time).count();
            if(hour >= m_limit_hour){
                if(m_tmp_show > 0)
                    printf("\n");
                m_finished = true;
                break;
            }

        }

        // update weights for batch learning
        if (calcWeightUpdates && (!Configuration::instance().hybridOnlineBatch() || m_numDevice > 1)){
            if (m_numDevice == 1) _updateWeights();
            else                  _updateWeightsMultiGpu();
        }
  #ifdef _MYMPI
        _syncWeight();
  #endif

        // normalize the errors : default
        // if (m_errorType)
        error = error / (float)ds.totalSequences();
        // else  // perplexity
            // error = exp(error / ds.totalTimesteps());
        *classError /= (real_t)ds.totalTimesteps();
        if (m_tmp_show > 0)
            printf("\n %5d | ", m_curEpoch);
        return error;
    }

    template <typename TDevice>
    void lmOptimizer<TDevice>::_exportWeights(const helpers::JsonDocument &jsonDoc, const char *arrayName, const std::vector<real_vector> &weights)
    {
        rapidjson::Value weightsArray(rapidjson::kArrayType);
        weightsArray.Reserve((rapidjson::SizeType)weights.size(), jsonDoc->GetAllocator());

        for (size_t i = 0; i < weights.size(); ++i) {
            rapidjson::Value v(rapidjson::kArrayType);
            Cpu::real_vector w = weights[i];
            v.Reserve((rapidjson::SizeType)w.size(), jsonDoc->GetAllocator());
            for (size_t j = 0; j < w.size(); ++j)
                v.PushBack(w[j], jsonDoc->GetAllocator());
            weightsArray.PushBack(v, jsonDoc->GetAllocator());
        }

        jsonDoc->AddMember(arrayName, weightsArray, jsonDoc->GetAllocator());
    }

    template <typename TDevice>
    void lmOptimizer<TDevice>::_importWeights(const helpers::JsonDocument &jsonDoc, const char *arrayName, std::vector<real_vector> *weights)
    {
        if (!jsonDoc->HasMember(arrayName) || !(*jsonDoc)[arrayName].IsArray())
            throw std::runtime_error(std::string("Array '") + arrayName + "' is missing or has the wrong type");

        if ((*jsonDoc)[arrayName].Size() != (rapidjson::SizeType)weights->size())
            throw std::runtime_error(std::string("Array '") + arrayName + "' has a wrong size");

        int i = 0;
        for (rapidjson::Value::ConstValueIterator it = (*jsonDoc)[arrayName].Begin(); it != (*jsonDoc)[arrayName].End(); ++it) {
            if (!it->IsArray())
                throw std::runtime_error(std::string("Object in '") + arrayName + "' is not an array");
            if (it->Size() != (rapidjson::SizeType)(*weights)[i].size())
                throw std::runtime_error(std::string("Subarray in '") + arrayName + "' has a wrong size");

            Cpu::real_vector w;
            w.reserve(it->Size());
            for (rapidjson::Value::ConstValueIterator it2 = it->Begin(); it2 != it->End(); ++it2)
                w.push_back((real_t)it2->GetDouble());

            (*weights)[i] = w;

            ++i;
        }
    }

    template <typename TDevice>
    void lmOptimizer<TDevice>::_storeWeights()
    {
        for (size_t i = 1; i < m_neuralNetwork.layers().size() - 1; ++i) {
            layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (layer)
            	thrust::copy(layer->weights().begin(), layer->weights().end(), m_bestWeights[i].begin());
        }
    }

    template <typename TDevice>
    void lmOptimizer<TDevice>::_restoreWeights()
    {
        for (size_t i = 1; i < m_neuralNetwork.layers().size() - 1; ++i) {
        	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (layer)
            	thrust::copy(m_bestWeights[i].begin(), m_bestWeights[i].end(), layer->weights().begin());
        }
    }

    template <typename TDevice>
    NeuralNetwork<TDevice>& lmOptimizer<TDevice>::_neuralNetwork()
    {
        return m_neuralNetwork;
    }

    template <typename TDevice>
    std::vector<typename lmOptimizer<TDevice>::real_vector>& lmOptimizer<TDevice>::_curWeightUpdates()
    {
        return m_curWeightUpdates;
    }


    template <typename TDevice>
    std::vector<Cpu::real_vector>& lmOptimizer<TDevice>::_UpdateSums()
    {
          return m_UpdateSums;
      }

      template <typename TDevice>
      std::vector<Cpu::real_vector>& lmOptimizer<TDevice>::_allWeightUpdates()
    {
        return m_UpdateSums;
    }

    template <typename TDevice>
    int lmOptimizer<TDevice>::_layersize()
    {
        return m_layer_size;
    }

    template <typename TDevice>
    int lmOptimizer<TDevice>::_numDevice()
    {
        return m_numDevice;
    }

    template <typename TDevice>
    void lmOptimizer<TDevice>::use_entropyError()
    {
        m_errorType = 0;
    }

    template <typename TDevice>
    void lmOptimizer<TDevice>::saveEvery(std::string savedir)
    {
        m_savedir = savedir;
        m_saveEvery = true;
    }

    template <typename TDevice>
    lmOptimizer<TDevice>::lmOptimizer(NeuralNetwork<TDevice> &neuralNetwork, data_sets::Corpus &trainingSet,
                                   data_sets::Corpus &validationSet, data_sets::Corpus &testSet,
                                   int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery,
                                   int temp_show, int limit_hour, int multi_sync_pace)
        : m_neuralNetwork             (neuralNetwork)
        , m_trainingSet               (trainingSet)
        , m_validationSet             (validationSet)
        , m_testSet                   (testSet)
        , m_maxEpochs                 (maxEpochs)
        , m_maxEpochsNoBest           (maxEpochsNoBest)
        , m_validateEvery             (validateEvery)
        , m_testEvery                 (testEvery)
        , m_finished                  (false)
        , m_curEpoch                  (0)
        , m_epochsSinceLowestError    (0)
        , m_lowestValidationError     (std::numeric_limits<real_t>::max())
        , m_curTrainingError          (std::numeric_limits<real_t>::max())
        , m_curValidationError        (std::numeric_limits<real_t>::max())
        , m_curTestError              (std::numeric_limits<real_t>::max())
        , m_curValidationClassError   (0)
        , m_curTrainingClassError     (0)
        , m_curTestClassError         (0)
        , m_errorType                 (0)
        , m_savedir                   ("")
        , m_saveEvery                 (false)
        , m_tmp_show                  (temp_show)
        , m_limit_hour                (limit_hour)
        , m_multi_sync_pace           (multi_sync_pace)
    {
        // initialize the best weights vectors
	m_start_time = std::chrono::system_clock::now();
        m_numDevice = m_neuralNetwork.getNumDevice();
        m_bestWeights.resize(m_neuralNetwork.layers().size());
        m_allWeightUpdates.resize(m_neuralNetwork.layers().size());
        m_UpdateSums.resize(m_neuralNetwork.layers().size());
        m_curWeightUpdates.resize(m_neuralNetwork.layers().size() * m_numDevice);
        m_layer_size = m_neuralNetwork.layers().size();
        if (Configuration::instance().weightNoiseSigma() > 0)
            origWeights = std::vector<real_vector>( m_neuralNetwork.layers().size() * m_numDevice);
        for ( int device = 0; device < m_numDevice; ++device){
            cudaSetDevice(device);
            for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
            	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers(device)[i].get());
                if (layer){
                    if (device == 0){
                        m_bestWeights[i] = layer->weights();
                        m_allWeightUpdates[i] = Cpu::real_vector(layer->weights().size());
                        m_UpdateSums[i] = Cpu::real_vector(layer->weights().size());
                    }
                    m_curWeightUpdates[device * m_layer_size + i] = layer->weights();
                    if (Configuration::instance().weightNoiseSigma() > 0)
                        origWeights[device * m_layer_size + i] = layer->weights();
                }
                else{
                    if (i != 1) continue;
                  	layers::LookupLayer<TDevice> *_layer = dynamic_cast<layers::LookupLayer<TDevice>*>(m_neuralNetwork.layers(device)[i].get());
                    if (_layer){
                        if (device == 0){
                            m_bestWeights[i] = _layer->weightUpdates();
                            m_allWeightUpdates[i] = Cpu::real_vector(_layer->weightUpdates().size());
                            // allocate for embeddings of appeared word
                            m_UpdateSums[i] = Cpu::real_vector(_layer->weightUpdates().size() * m_numDevice);
                        }
                        m_curWeightUpdates[device * m_layer_size + i] = _layer->weightUpdates();
                    }
                }
            }
        }
        // m_UpdateSums = m_allWeightUpdates;


        // initialize the current weight updates vectors
        // m_curWeightUpdates = m_bestWeights;
    }

    template <typename TDevice>
    lmOptimizer<TDevice>::~lmOptimizer()
    {
        for ( int device = 0; device < m_numDevice; ++device){
            cudaSetDevice(device);
            for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
                m_curWeightUpdates[device * m_layer_size + i].clear();
                m_curWeightUpdates[device * m_layer_size + i].shrink_to_fit();
            }
        }
        m_curWeightUpdates.clear();
    }

    template <typename TDevice>
    bool lmOptimizer<TDevice>::set_syncPace(int pace)
    {
        m_multi_sync_pace = pace;
    }

    template <typename TDevice>
    bool lmOptimizer<TDevice>::finished() const
    {
        return m_finished;
    }

    template <typename TDevice>
    int lmOptimizer<TDevice>::currentEpoch() const
    {
        return m_curEpoch;
    }

    template <typename TDevice>
    real_t lmOptimizer<TDevice>::lowestValidationError() const
    {
        return m_lowestValidationError;
    }

    template <typename TDevice>
    int lmOptimizer<TDevice>::epochsSinceLowestValidationError() const
    {
        return m_epochsSinceLowestError;
    }

    template <typename TDevice>
    real_t lmOptimizer<TDevice>::curTrainingError() const
    {
        return m_curTrainingError;
    }

    template <typename TDevice>
    real_t lmOptimizer<TDevice>::curValidationError() const
    {
        return m_curValidationError;
    }

    template <typename TDevice>
    real_t lmOptimizer<TDevice>::curTestError() const
    {
        return m_curTestError;
    }

    template <typename TDevice>
    real_t lmOptimizer<TDevice>::curTrainingClassError() const
    {
        return m_curTrainingClassError;
    }

    template <typename TDevice>
    real_t lmOptimizer<TDevice>::curValidationClassError() const
    {
        return m_curValidationClassError;
    }

    template <typename TDevice>
    real_t lmOptimizer<TDevice>::curTestClassError() const
    {
        return m_curTestClassError;
    }

    template <typename TDevice>
    bool lmOptimizer<TDevice>::train()
    {
#ifndef _MYMPI
        if (!m_finished) {
            ++m_curEpoch;

            // train one epoch and update the weights
            m_curTrainingError = _processDataSet(m_trainingSet, true, &m_curTrainingClassError);
            if (m_saveEvery) {
                std::string Epoch = std::to_string(m_curEpoch);
                std::string dir = m_savedir + "/" + Epoch;
                boost::system::error_code err;
                const bool ret = boost::filesystem::create_directory(dir, err);
                m_neuralNetwork.exportWeightsBinary(dir);
            }
	    // m_start_time = std::chrono::system_clock::now();

            // calculate the validation error and store the weights if we a new lowest error
            if (!m_validationSet.empty() && m_curEpoch % m_validateEvery == 0) {
                m_curValidationError = _processDataSet(m_validationSet, false, &m_curValidationClassError);

                if (m_curValidationError < m_lowestValidationError) {
                    m_lowestValidationError  = m_curValidationError;
                    m_epochsSinceLowestError = 0;

                    _storeWeights();
                }
                else {
                    m_epochsSinceLowestError += m_validateEvery;
                }
            }
            else if (m_validationSet.empty()) {
                m_epochsSinceLowestError = 0;
                _storeWeights();
            }

            // calculate the test error
            if (!m_testSet.empty() && m_curEpoch % m_testEvery == 0)
                m_curTestError = _processDataSet(m_testSet, false, &m_curTestClassError);

            // check if we did not get a new lowest error for some training epochs
            // or if we reached the maximum number of training epochs
            if (m_epochsSinceLowestError >= m_maxEpochsNoBest || (m_maxEpochs >= 0 && m_curEpoch >= m_maxEpochs) || m_finished) {
                _restoreWeights();
                m_finished = true;
            }
        }

        return m_finished;
#endif

#ifdef _MYMPI
        if (!m_finished) {
            ++m_curEpoch;

            // train one epoch and update the weights
            m_curTrainingError = _processDataSet(m_trainingSet, true, &m_curTrainingClassError);

      	    m_start_time = std::chrono::system_clock::now();
            if (MPI::COMM_WORLD.Get_rank() == 0) {
                // calculate the validation error and store the weights if we a new lowest error
                if (!m_validationSet.empty() && m_curEpoch % m_validateEvery == 0) {
                    m_curValidationError = _processDataSet(m_validationSet, false, &m_curValidationClassError);

                    if (m_curValidationError < m_lowestValidationError) {
                        m_lowestValidationError  = m_curValidationError;
                        m_epochsSinceLowestError = 0;

                        _storeWeights();
                    }
                    else {
                        m_epochsSinceLowestError += m_validateEvery;
                    }
                }
                else if (m_validationSet.empty()) {
                    m_epochsSinceLowestError = 0;
                    _storeWeights();
                }

            // calculate the test error
                if (!m_testSet.empty() && m_curEpoch % m_testEvery == 0)
                    m_curTestError = _processDataSet(m_testSet, false, &m_curTestClassError);
                // check if we did not get a new lowest error for some training epochs
                // or if we reached the maximum number of training epochs
                if (m_epochsSinceLowestError >= m_maxEpochsNoBest || (m_maxEpochs >= 0 && m_curEpoch >= m_maxEpochs) || m_finished) {
                    _restoreWeights();
                    m_finished = true;
                }

            }
            MPI::COMM_WORLD.Bcast(
                  &m_finished,
                  1,
                  MPI::BOOL,
                  0
            );
        }

        return m_finished;
#endif
    }

    template <typename TDevice>
    void lmOptimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        jsonDoc->AddMember("optimizer_finished",                   m_finished,                jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_epoch",                  m_curEpoch,                jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_epochs_since_lowest_error",  m_epochsSinceLowestError,  jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_lowest_validation_error",    m_lowestValidationError,   jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_training_error",         m_curTrainingError,        jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_validation_error",       m_curValidationError,      jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_test_error",             m_curTestError,            jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_training_class_error",   m_curTrainingClassError,   jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_validation_class_error", m_curValidationClassError, jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_test_class_error",       m_curTestClassError,       jsonDoc->GetAllocator());

        _exportWeights(jsonDoc, "optimizer_best_weights", m_bestWeights);
    }

    template <typename TDevice>
    void lmOptimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        m_finished                = helpers::checkedJsonGet<bool  >(*jsonDoc, "optimizer_finished");
        m_curEpoch                = helpers::checkedJsonGet<int   >(*jsonDoc, "optimizer_cur_epoch");
        m_epochsSinceLowestError  = helpers::checkedJsonGet<int   >(*jsonDoc, "optimizer_epochs_since_lowest_error");
        m_lowestValidationError   = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_lowest_validation_error");
        m_curTrainingError        = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_training_error");
        m_curValidationError      = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_validation_error");
        m_curTestError            = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_test_error");
        m_curTrainingClassError   = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_training_class_error");
        m_curValidationClassError = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_validation_class_error");
        m_curTestClassError       = helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_test_class_error");

        _importWeights(jsonDoc, "optimizer_best_weights", &m_bestWeights);
    }

    template <typename TDevice>
    void lmOptimizer<TDevice>::setLimitHour(int limit_hour)
    {
        m_limit_hour = limit_hour;
    }

    // explicit template instantiations
    template class lmOptimizer<Cpu>;
    template class lmOptimizer<Gpu>;

} // namespace optimizers
