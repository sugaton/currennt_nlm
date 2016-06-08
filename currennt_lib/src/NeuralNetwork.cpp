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

#include "NeuralNetwork.hpp"
#include "Configuration.hpp"
#include "LayerFactory.hpp"
#include "layers/InputLayer.hpp"
#include "rnnlm/intInputLayer.hpp"
#include "rnnlm/LookupLayer.hpp"
#include "layers/PostOutputLayer.hpp"
#include "helpers/JsonClasses.hpp"

#include <vector>
#include <map>
#include <stdexcept>
#include <cassert>

#include <boost/foreach.hpp>


template <typename TDevice>
NeuralNetwork<TDevice>::NeuralNetwork(const helpers::JsonDocument &jsonDoc, int parallelSequences, int maxSeqLength,
                                      int inputSizeOverride, int outputSizeOverride, int vocab_size, int numberOfDevice)
                                    //   int inputSizeOverride = -1, int outputSizeOverride = -1)
{
    bool b_intinput = false;
    printf("\nmaxSeqLength:%d parallelSequences: %d outputSizeOverride: %d\n", maxSeqLength, parallelSequences, outputSizeOverride);
    try {
        // check the layers and weight sections
        if (!jsonDoc->HasMember("layers"))
            throw std::runtime_error("Missing section 'layers'");
        rapidjson::Value &layersSection  = (*jsonDoc)["layers"];

        if (!layersSection.IsArray())
            throw std::runtime_error("Section 'layers' is not an array");

        helpers::JsonValue weightsSection;
        if (jsonDoc->HasMember("weights")) {
            if (!(*jsonDoc)["weights"].IsObject())
                throw std::runtime_error("Section 'weights' is not an object");

            weightsSection = helpers::JsonValue(&(*jsonDoc)["weights"]);
        }
        m_layers.resize(numberOfDevice);
        // extract the layers
        for (rapidjson::Value::ValueIterator layerChild = layersSection.Begin(); layerChild != layersSection.End(); ++layerChild) {
            // check the layer child type
            if (!layerChild->IsObject())
                throw std::runtime_error("A layer section in the 'layers' array is not an object");

            // extract the layer type and create the layer
            if (!layerChild->HasMember("type"))
                throw std::runtime_error("Missing value 'type' in layer description");

            std::string layerType = (*layerChild)["type"].GetString();

            printf("\t creating network %s..\n", (*layerChild)["name"].GetString());
            // override input/output sizes
            if (inputSizeOverride > 0 && layerType == "input") {
                (*layerChild)["size"].SetInt(inputSizeOverride);
            }
// /*  Does not work yet, need another way to identify a) postoutput layer (last!) and then the corresponging output layer and type!
            if (outputSizeOverride > 0 && strcmp( (*layerChild)["name"].GetString(), "output" )==0) {
                (*layerChild)["size"].SetInt(outputSizeOverride);
            }
            if (outputSizeOverride > 0 && ( strcmp( (*layerChild)["name"].GetString() , "postoutput" ) == 0 || strcmp( (*layerChild)["type"].GetString() , "multiclass_classification" ) == 0 ) ) {
                (*layerChild)["size"].SetInt(outputSizeOverride);
            }
            // lookup-layer need the number of word
            if (strcmp( (*layerChild)["name"].GetString() , "lookup" ) == 0 ) {
                // (*layerChild)["w_size"].SetInt(outputSizeOverride);
                (*layerChild).AddMember("w_size", vocab_size, jsonDoc->GetAllocator());
                if (jsonDoc->HasMember("max_lookup_size"))
                    (*layerChild).AddMember("max_gpusize", (*jsonDoc)["max_lookup_size"].GetInt(), jsonDoc->GetAllocator());
            }
// */
            for (size_t device = 0; device < m_layers.size(); ++device){
                try {
                    cudaSetDevice(device);
                    layers::Layer<TDevice> *layer;

                    if (m_layers.at(device).empty())
                        layer = LayerFactory<TDevice>::createLayer(layerType, &*layerChild, weightsSection, parallelSequences, maxSeqLength);
                    else
                        layer = LayerFactory<TDevice>::createLayer(layerType, &*layerChild, weightsSection, parallelSequences, maxSeqLength, m_layers.at(device).back().get());

                    m_layers.at(device).push_back(boost::shared_ptr<layers::Layer<TDevice> >(layer));
        void saveEvery(std::string savedir);
                }
                catch (const std::exception &e) {
                    throw std::runtime_error(std::string("Could not create layer: ") + e.what());
                }
            }
        }

        // check if we have at least one input, one output and one post output layer
        if (m_layers[0].size() < 3)
            throw std::runtime_error("Not enough layers defined");

        // check if only the first layer is an input layer
        if (!dynamic_cast<layers::InputLayer<TDevice>*>(m_layers[0].front().get()))
            if (!dynamic_cast<layers::intInputLayer<TDevice>*>(m_layers[0].front().get()))
                throw std::runtime_error("The first layer is not an input layer");
            else b_intinput = true;

        for (size_t i = 1; i < m_layers[0].size(); ++i) {
            if (dynamic_cast<layers::InputLayer<TDevice>*>(m_layers[0][i].get()) || dynamic_cast<layers::intInputLayer<TDevice>*>(m_layers[0][i].get()))
                throw std::runtime_error("Multiple input layers defined");
        }

        // check if only the last layer is a post output layer
        if (!dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers[0].back().get()))
            throw std::runtime_error("The last layer is not a post output layer");

        for (size_t i = 0; i < m_layers.size()-1; ++i) {
            if (dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers[0][i].get()))
                throw std::runtime_error("Multiple post output layers defined");
        }

        // check if two layers have the same name
        for (size_t i = 0; i < m_layers[0].size(); ++i) {
            for (size_t j = 0; j < m_layers[0].size(); ++j) {
                if (i != j && m_layers[0][i]->name() == m_layers[0][j]->name())
                    throw std::runtime_error(std::string("Different layers have the same name '") + m_layers[0][i]->name() + "'");
            }
        }
    }
    catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid network file: ") + e.what());
    }
}

template <typename TDevice>
NeuralNetwork<TDevice>::~NeuralNetwork()
{
}


template <typename TDevice>
void NeuralNetwork<TDevice>::setWordDict(std::unordered_map<std::string, int> *wdic)
{
    for (size_t device = 0; device < m_layers.size(); ++device){
    	layers::LookupLayer<TDevice> *lookup = dynamic_cast<layers::LookupLayer<TDevice>*>(m_layers[device][1].get());
        // need to Cast, cause cuda code dose not support unordered_map.
        std::map<std::string, int> tmp_map(wdic->begin(), wdic->end());
        if (lookup)
            lookup->setWordDict(&tmp_map);
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::loadEmbeddings(const std::string& filename)
{
    for (size_t device = 0; device < m_layers.size(); ++device){
        cudaSetDevice(device);
    	layers::LookupLayer<TDevice> *lookup = dynamic_cast<layers::LookupLayer<TDevice>*>(m_layers[device][1].get());
        // need to Cast, cause cuda code dose not support unordered_map.
        if (lookup)
            lookup->loadEmbeddings(filename);
    }
}
// TODO is it ok to return m_layers[0] (device 0 layers)?
template <typename TDevice>
const std::vector<boost::shared_ptr<layers::Layer<TDevice> > >& NeuralNetwork<TDevice>::layers(const int device) const
{
    cudaSetDevice(device);
    return m_layers.at(device);
}

template <typename TDevice>
layers::InputLayer<TDevice>& NeuralNetwork<TDevice>::inputLayer(const int device)
{
    cudaSetDevice(device);
    return static_cast<layers::InputLayer<TDevice>&>(*m_layers.at(device).front());
}

template <typename TDevice>
layers::intInputLayer<TDevice>& NeuralNetwork<TDevice>::intinputLayer(const int device)
{
    cudaSetDevice(device);
    return static_cast<layers::intInputLayer<TDevice>&>(*m_layers.at(device).front());
}

template <typename TDevice>
layers::TrainableLayer<TDevice>& NeuralNetwork<TDevice>::outputLayer(const int device)
{
    cudaSetDevice(device);
    return static_cast<layers::TrainableLayer<TDevice>&>(*m_layers[device][m_layers.at(0).size()-2]);
}

template <typename TDevice>
layers::PostOutputLayer<TDevice>& NeuralNetwork<TDevice>::postOutputLayer(const int device)
{
    cudaSetDevice(device);
    return static_cast<layers::PostOutputLayer<TDevice>&>(*m_layers.at(device).back());
}

template <typename TDevice>
void NeuralNetwork<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction, const int device)
{
    cudaSetDevice(device);
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers[device]){
        layer->loadSequences(fraction);
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::computeForwardPass()
{
    // for (int i = 0; i < m_layers.at(0).size(); ++i){
    for (size_t device = 0; device < m_layers.size(); ++device){
        cudaSetDevice(device);
        BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers[device]) {
        // for (size_t device = 0; device < m_layers.size(); ++device){
        //     cudaSetDevice(device);
            // boost::shared_ptr<layers::Layer<TDevice> > &layer = m_layers.at(device).at(i);
            layer->computeForwardPass();
        }
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::computeBackwardPass()
{
    for (size_t device = 0; device < m_layers.size(); ++device){
        cudaSetDevice(device);
        BOOST_REVERSE_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers[device]) {
    // for (int i = 0; i < m_layers.at(0).size(); ++i){
        // for (size_t device = 0; device < m_layers.size(); ++device){
            // cudaSetDevice(device);
            // boost::shared_ptr<layers::Layer<TDevice> > &layer = m_layers.at(device).at(i);
            layer->computeBackwardPass();
        //std::cout << "output errors " << layer->name() << std::endl;
        //thrust::copy(layer->outputErrors().begin(), layer->outputErrors().end(), std::ostream_iterator<real_t>(std::cout, ";"));
        //std::cout << std::endl;
        }
    }
}

template <typename TDevice>
real_t NeuralNetwork<TDevice>::calculateError(const int device) const
{
    cudaSetDevice(device);
    return static_cast<layers::PostOutputLayer<TDevice>&>(*m_layers[device].back()).calculateError();
}

template <typename TDevice>
real_t NeuralNetwork<TDevice>::calculateEntropy(const int device) const
{
    cudaSetDevice(device);
    return static_cast<layers::PostOutputLayer<TDevice>&>(*m_layers[device].back()).calculateEntropy();
}

template <typename TDevice>
void NeuralNetwork<TDevice>::exportLayers(const helpers::JsonDocument& jsonDoc) const
{
    cudaSetDevice(0);
    if (!jsonDoc->IsObject())
        throw std::runtime_error("JSON document root must be an object");

    // create the layers array
    rapidjson::Value layersArray(rapidjson::kArrayType);

    // create the layer objects
    for (size_t i = 0; i < m_layers.size(); ++i)
        m_layers[0][i]->exportLayer(&layersArray, &jsonDoc->GetAllocator());

    // if the section already exists, we delete it first
    if (jsonDoc->HasMember("layers"))
        jsonDoc->RemoveMember("layers");

    // add the section to the JSON document
    jsonDoc->AddMember("layers", layersArray, jsonDoc->GetAllocator());

	layers::LookupLayer<TDevice> *lookup = dynamic_cast<layers::LookupLayer<TDevice>*>(m_layers[0][1].get());
    if (lookup)
        lookup->exportDict(jsonDoc, &jsonDoc->GetAllocator());
}


template <typename TDevice>
void NeuralNetwork<TDevice>::exportWeights(const helpers::JsonDocument& jsonDoc) const
{
    cudaSetDevice(0);
    if (!jsonDoc->IsObject())
        throw std::runtime_error("JSON document root must be an object");

    // create the weights object
    rapidjson::Value weightsObject(rapidjson::kObjectType);

    // create the weight objects
    BOOST_FOREACH (const boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers[0]) {
    	layers::TrainableLayer<TDevice> *trainableLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
        if (trainableLayer)
            trainableLayer->exportWeights(&weightsObject, &jsonDoc->GetAllocator());
        else{
        	layers::LookupLayer<TDevice> *lookup = dynamic_cast<layers::LookupLayer<TDevice>*>(layer.get());
            if (lookup)
                lookup->exportWeights(&weightsObject, &jsonDoc->GetAllocator());
        }
    }

    // if the section already exists, we delete it first
    if (jsonDoc->HasMember("weights"))
        jsonDoc->RemoveMember("weights");

    // add the section to the JSON document
    jsonDoc->AddMember("weights", weightsObject, jsonDoc->GetAllocator());
}

template <typename TDevice>
void NeuralNetwork<TDevice>::exportWeightsBinary(const std::string &dirname) const
{
    cudaSetDevice(0);

    // create the weight objects
    BOOST_FOREACH (const boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers[0]) {
    	layers::TrainableLayer<TDevice> *trainableLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
        if (trainableLayer)
            trainableLayer->exportWeightsBinary(dirname);
        else{
        	layers::LookupLayer<TDevice> *lookup = dynamic_cast<layers::LookupLayer<TDevice>*>(layer.get());
            if (lookup)
                lookup->exportWeightsBinary(dirname);
        }
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::importWeightsBinary(const std::string &dirname)
{
    // cudaSetDevice(0);

    // create the weight objects
    for (size_t device = 0; device < m_layers.size(); ++device){
        cudaSetDevice(device);
        BOOST_FOREACH (const boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers[device]) {
        	layers::TrainableLayer<TDevice> *trainableLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
            if (trainableLayer)
                trainableLayer->importWeightsBinary(dirname);
            else{
            	layers::LookupLayer<TDevice> *lookup = dynamic_cast<layers::LookupLayer<TDevice>*>(layer.get());
                if (lookup)
                    lookup->importWeightsBinary(dirname);
            }
        }
    }
}

template <typename TDevice>
std::vector<std::vector<std::vector<real_t> > > NeuralNetwork<TDevice>::getOutputs(const int device)
{
    layers::TrainableLayer<TDevice> &ol = outputLayer(device);

    std::vector<std::vector<std::vector<real_t> > > outputs;
    for (int patIdx = 0; patIdx < (int)ol.patTypes().size(); ++patIdx) {
        switch (ol.patTypes()[patIdx]) {
        case PATTYPE_FIRST:
            outputs.resize(outputs.size() + 1);

        case PATTYPE_NORMAL:
        case PATTYPE_LAST: {{
            Cpu::real_vector pattern(ol.outputs().begin() + patIdx * ol.size(), ol.outputs().begin() + (patIdx+1) * ol.size());
            int psIdx = patIdx % ol.parallelSequences();
            outputs[psIdx].push_back(std::vector<real_t>(pattern.begin(), pattern.end()));
            break;
        }}

        default:
            break;
        }
    }

    return outputs;
}

template <typename TDevice>
typename NeuralNetwork<TDevice>::real_vector& NeuralNetwork<TDevice>::last_layer()
{
    layers::TrainableLayer<TDevice> &ol = static_cast<layers::TrainableLayer<TDevice>&>(*m_layers[0][m_layers.at(0).size()-3]);
    return ol.outputs();
}

template <typename TDevice>
int NeuralNetwork<TDevice>::getNumDevice()
{
    return (int)m_layers.size();
}


template <typename TDevice>
void NeuralNetwork<TDevice>::fixLookup()
{
    for (size_t device = 0; device < m_layers.size(); ++device){
        cudaSetDevice(device);
    	layers::LookupLayer<TDevice> *lookup = dynamic_cast<layers::LookupLayer<TDevice>*>(m_layers[device][1].get());
        // need to Cast, cause cuda code dose not support unordered_map.
        if (lookup)
            lookup->fixEmb();
    }
}

// explicit template instantiations
template class NeuralNetwork<Cpu>;
template class NeuralNetwork<Gpu>;
