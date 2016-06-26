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

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "Types.hpp"
#include "layers/InputLayer.hpp"
#include "rnnlm/intInputLayer.hpp"
#include "layers/TrainableLayer.hpp"
#include "rnnlm/LookupLayer.hpp"
#include "layers/PostOutputLayer.hpp"
#include "data_sets/DataSet.hpp"
#include "helpers/JsonClassesForward.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <boost/shared_ptr.hpp>

#include <vector>
#include <memory>
#include <unordered_map>


/*****************************************************************************************************************//**
 * Represents the neural network
 *
 * @param TDevice The computation device
 *********************************************************************************************************************/
template <typename TDevice>
class NeuralNetwork
{
    typedef typename TDevice::real_vector real_vector;
private:
    std::vector< std::vector<boost::shared_ptr<layers::Layer<TDevice> > > > m_layers;
    int m_lookup_layer_idx = -1;

public:
    /**
     * Creates the neural network from the process configuration
     *
     * @param jsonDoc           The JSON document containing the network configuration
     * @param parallelSequences The maximum number of sequences that shall be computed in parallel
     * @param maxSeqLength      The maximum length of a sequence
     */
    NeuralNetwork(const helpers::JsonDocument &jsonDoc, int parallelSequences, int maxSeqLength,
                  int inputSizeOverride = -1, int outputSizeOverride = -1, int vocab_size = -1, int numberOfDevice = 1);

    /**
     * Destructs the neural network
     */
    ~NeuralNetwork();

    /**
     * Returns the layers
     *
     * @return The layers
     */
    const std::vector<boost::shared_ptr<layers::Layer<TDevice> > >& layers(const int device = 0) const;

    /**
     * Returns the input layer
     *
     * @return The input layer
     */
    layers::InputLayer<TDevice>& inputLayer(const int device = 0);
    layers::intInputLayer<TDevice>& intinputLayer(const int device = 0);

    /**
     * Returns the output layer
     *
     * @return The output layer
     */
    layers::TrainableLayer<TDevice>& outputLayer(const int device = 0);

    /**
     * Returns the post output layer
     *
     * @return The post output layer
     */
    layers::PostOutputLayer<TDevice>& postOutputLayer(const int device = 0);

    layers::LookupLayer<TDevice>& lookupLayer(const int device = 0);
    /**
     * Loads sequences to the device
     *
     * @param fraction The data set fraction containing the sequences
     */
    void loadSequences(const data_sets::DataSetFraction &fraction, const int device = 0);

    /**
     * Computes the forward pass
     */
    void computeForwardPass();

    /**
     * Computes the backward pass, including the weight updates
     *
     * The forward pass must be computed first!
     */
    void computeBackwardPass();

    /**
     * Calculates the error at the output layer
     *
     * The forward pass must be computed first!
     *
     * @return The computed error
     */
    real_t calculateError(int end = -1, const int device = 0) const;
    // entropy ver.
    real_t calculateEntropy(const int device = 0) const;


    void setWordDict(std::unordered_map<std::string, int> *wdic);

    /*
        load pretrained embeddings from w2v-style txt-file
    */
    void loadEmbeddings(const std::string& filename);
    /**
     * Stores the description of the layers in a JSON tree
     *
     * @param jsonDoc The JSON document
     */
    void exportLayers(const helpers::JsonDocument& jsonDoc) const;

    /**
     * Stores the weights in a JSON tree
     *
     * @param jsonDoc The JSON document
     */
    void exportWeights(const helpers::JsonDocument& jsonDoc) const;

    void exportWeightsBinary(const std::string &dirname) const;
    void importWeightsBinary(const std::string &dirname, std::unordered_map<std::string, int> *m = NULL);
    /**
     * Returns the outputs of the processed fraction
     *
     * ...[1][2][3] contains the activation of the 4th output neuron at the 3nd timestep
     * of the 2nd parallel sequence.
     *
     * @return Outputs of the processed fraction
     */
    std::vector<std::vector<std::vector<real_t> > > getOutputs(const int device = 0);
    real_vector& last_layer();

    /**
     * Returns the number of using gpu device.
     */
    int getNumDevice();
    void fixLookup();
};


#endif // NEURALNETWORK_HPP
