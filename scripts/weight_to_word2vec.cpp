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

/***
    main_nlm.cpp
    this file includes main function of currennt for rnnlm
 ***/


#include "../currennt_lib/src/helpers/Matrix.hpp"
#include "../currennt_lib/src/Configuration.hpp"
#include "../currennt_lib/src/NeuralNetwork.hpp"
#include "../currennt_lib/src/layers/LstmLayer.hpp"
#include "../currennt_lib/src/layers/BinaryClassificationLayer.hpp"
#include "../currennt_lib/src/layers/MulticlassClassificationLayer.hpp"

#include "../currennt_lib/src/rnnlm/intInputLayer.hpp"
#include "../currennt_lib/src/rnnlm/LookupLayer.hpp"
#include "../currennt_lib/src/lmOptimizers/lmSteepestDescentOptimizer.hpp"
// #include "../currennt_lib/src/optimizers/SteepestDescentOptimizer.hpp"

#include "../currennt_lib/src/helpers/JsonClasses.hpp"
#include "../currennt_lib/src/rapidjson/prettywriter.h"
#include "../currennt_lib/src/rapidjson/filestream.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <stdarg.h>
#include <sstream>
#include <cstdlib>
#include <iomanip>


void swap32 (uint32_t *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}

void swap16 (uint16_t *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 1 ); *( q + 1 ) = temp;
}

void swapFloat(float *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}

enum data_set_type
{
    DATA_SET_TRAINING,
    DATA_SET_VALIDATION,
    DATA_SET_TEST,
    DATA_SET_FEEDFORWARD
};

// helper functions (implementation below)
void readJsonFile(rapidjson::Document *doc, const std::string &filename);
void loadDict(rapidjson::Document *doc, std::unordered_map<std::string, int> *p_dict);
void outputWeight(rapidjson::Document *doc, rapidjson::Document *doc2, std::unordered_map<std::string, int> *p_dict, std::string outputfilename);
boost::shared_ptr<data_sets::Corpus> loadDataSet(data_set_type dsType, const int max_vocab, std::unordered_map<std::string, int>* p_map=NULL, int constructDict = 0);
template <typename TDevice> void printLayers(const NeuralNetwork<TDevice> &nn);
template <typename TDevice> void printOptimizer(const optimizers::lmOptimizer<TDevice> &optimizer);
template <typename TDevice> void saveNetwork(const NeuralNetwork<TDevice> &nn, const std::string &filename);
void createModifiedTrainingSet(data_sets::Corpus *trainingSet, int parallelSequences, bool outputsToClasses, boost::mutex &swapTrainingSetsMutex);
template <typename TDevice> void saveState(const NeuralNetwork<TDevice> &nn, const optimizers::lmOptimizer<TDevice> &optimizer, const std::string &infoRows);
template <typename TDevice> void restoreState(NeuralNetwork<TDevice> *nn, optimizers::lmOptimizer<TDevice> *optimizer, std::string *infoRows);
std::string printfRow(const char *format, ...);


// main function
int main(int argc, char const *argv[])
{
    try {
        // read the neural network description file
        std::string networkFile = argv[1];
        std::string weightFile = argv[2];
        std::string outputFile = argv[3];
        printf("Reading network from '%s'... ", networkFile.c_str());
        fflush(stdout);
        rapidjson::Document layerDoc;
        rapidjson::Document weightDoc;
        readJsonFile(&layerDoc, networkFile);
        readJsonFile(&weightDoc, weightFile);
        printf("done.\n");
        printf("\n");

        printf("loadDict..." );
        fflush(stdout);
        std::unordered_map<std::string, int> _wordDict;
        loadDict(&weightDoc, &_wordDict);
        printf("done.\n");
        printf("\n");

        printf("output parameter to %s...", outputFile.c_str());
        fflush(stdout);
        outputWeight(&layerDoc, &weightDoc, &_wordDict, outputFile);
        printf("finished.\n");
    }
    catch (const std::exception &e) {
        printf("FAILED: %s\n", e.what());
        return 2;
    }

    return 0;
}


void readJsonFile(rapidjson::Document *doc, const std::string &filename)
{
    // open the file
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Cannot open file");

    // calculate the file size in bytes
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    // read the file into a buffer
    char *buffer = new char[size + 1];
    ifs.read(buffer, size);
    buffer[size] = '\0';

    std::string docStr(buffer);
    delete buffer;

    // extract the JSON tree
    if (doc->Parse<0>(docStr.c_str()).HasParseError())
        throw std::runtime_error(std::string("Parse error: ") + doc->GetParseError());
}

void loadDict(rapidjson::Document *doc, std::unordered_map<std::string, int> *p_dict)
{
    if (! doc->HasMember("word_dict") )
        return;
    const rapidjson::Value &wordlist = (*doc)["word_dict"];
    int c = 0;
    for (rapidjson::Value::ConstValueIterator wordObject = wordlist.Begin(); wordObject != wordlist.End(); ++wordObject) {
        if (! wordObject->HasMember("name"))
            throw std::runtime_error("A member of word_dict section doesn't have name property.");
        std::string word = (*wordObject)["name"].GetString();
        if (! wordObject->HasMember("id"))
            throw std::runtime_error("A member of word_dict section doesn't have id property.");
        int id = (*wordObject)["id"].GetInt();
        (*p_dict)[word] = id;
    }
    printf("%lld words are loaded from json file.", (long long int) p_dict->size());

}

void outputWeight(rapidjson::Document *jsonDoc, rapidjson::Document *jsonDoc2, std::unordered_map<std::string, int> *p_dict, std::string outputfilename)
{

    if (!jsonDoc->HasMember("layers"))
        throw std::runtime_error("Missing section 'layers'");
    rapidjson::Value &layersSection  = (*jsonDoc)["layers"];

    if (! jsonDoc2->HasMember("weights") )
        return;
    helpers::JsonValue weightsSection;
    if (jsonDoc2->HasMember("weights")) {
        //if (!(*jsonDoc)["weights"].IsObject())
            //throw std::runtime_error("Section 'weights' is not an object");

        weightsSection = helpers::JsonValue(&(*jsonDoc2)["weights"]);
    }
    std::ofstream ofs(outputfilename);
    bool found = false;
    int beforesize=0;
    for (rapidjson::Value::ValueIterator layerChild = layersSection.Begin(); layerChild != layersSection.End(); ++layerChild) {
        if (strcmp( (*layerChild)["name"].GetString(), "output" ) != 0 ){
            beforesize = (*layerChild)["size"].GetInt();
            continue;
        }
        int esize = beforesize;
        printf("found output layer! size : %d\n", esize);
        found = true;
        if (!weightsSection->HasMember("output"))
            throw std::runtime_error("softmax weight doesn't exist");
        const rapidjson::Value &weightsChild = (*weightsSection)["output"];
        const rapidjson::Value &inputWeightsChild = weightsChild["input"];
        const rapidjson::Value &biasWeightsChild  = weightsChild["bias"];

        Cpu::real_vector weights;
        weights.reserve(inputWeightsChild.Size());
        for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); it != inputWeightsChild.End(); ++it)
            weights.push_back(static_cast<real_t>(it->GetDouble()));

        // write "WordNum dimension(size)"
        ofs << p_dict->size() << " " << esize << std::endl;
        if (weights.size() < esize * p_dict->size()){
            std::unordered_map<int, std::string> reversed;
            for (auto it = p_dict->begin(); it != p_dict->end(); ++it){
                reversed[it->second] = it->first;
            }
            for (int id = 0; id < weights.size() / esize; ++id) {
                // writting word
                ofs << reversed[id] << " ";
                int startId = id * esize;
                // writting vector
                for (int i = 0; i < esize; ++i)
                    ofs << weights[startId + i] << " ";
                ofs << std::endl;
            }
        }
        else {
           for (auto it = p_dict->begin(); it != p_dict->end(); ++it){
               std::string word = it->first;
               int startId = it->second * esize;
               ofs << word << " ";
               for (int i = 0; i < esize; ++i)
                   ofs << weights[startId + i] << " ";
               ofs << std::endl;
            }
         }
    }
    if(!found)
        throw std::runtime_error("softmax layer's information doesn't exist in layer section");
}

boost::shared_ptr<data_sets::Corpus> loadDataSet(data_set_type dsType, const int max_vocab, std::unordered_map<std::string, int>* p_map, int constructDict)
{
    std::string type;
    std::vector<std::string> filenames;
    real_t fraction = 1;
    bool fracShuf   = false;
    bool seqShuf    = false;
    real_t noiseDev = 0;
    std::string cachePath = "";
    int truncSeqLength = -1;

    cachePath = Configuration::instance().cachePath();
    switch (dsType) {
    case DATA_SET_TRAINING:
        type     = "training set";
        filenames = Configuration::instance().trainingFiles();
        fraction = Configuration::instance().trainingFraction();
        fracShuf = Configuration::instance().shuffleFractions();
        seqShuf  = Configuration::instance().shuffleSequences();
        noiseDev = Configuration::instance().inputNoiseSigma();
        truncSeqLength = Configuration::instance().truncateSeqLength();
        break;

    case DATA_SET_VALIDATION:
        type     = "validation set";
        filenames = Configuration::instance().validationFiles();
        fraction = Configuration::instance().validationFraction();
        cachePath = Configuration::instance().cachePath();
        truncSeqLength = Configuration::instance().truncateSeqLength();
        break;

    case DATA_SET_TEST:
        type     = "test set";
        filenames = Configuration::instance().testFiles();
        fraction = Configuration::instance().testFraction();
        truncSeqLength = Configuration::instance().truncateSeqLength();
        break;

    default:
        type     = "feed forward input set";
        filenames = Configuration::instance().feedForwardInputFiles();
        noiseDev = Configuration::instance().inputNoiseSigma();
        truncSeqLength = Configuration::instance().truncateSeqLength();
        break;
    }

    printf("Loading %s ", type.c_str());
    for (std::vector<std::string>::const_iterator fn_itr = filenames.begin();
         fn_itr != filenames.end(); ++fn_itr)
    {
        printf("'%s' ", fn_itr->c_str());
    }
    printf("...");
    fflush(stdout);
    printf("maxvocab: %d\n", max_vocab);
    //std::cout << "truncating to " << truncSeqLength << std::endl;
    boost::shared_ptr<data_sets::Corpus> ds = boost::make_shared<data_sets::Corpus>(
        filenames,
        Configuration::instance().parallelSequences(), fraction, truncSeqLength,
        fracShuf, seqShuf, noiseDev, cachePath, p_map, constructDict, max_vocab);

    printf("done.\n");
    printf("Loaded fraction:  %d%%\n",   (int)(fraction*100));
    printf("Sequences:        %lld\n",     ds->totalSequences());
    printf("Sequence lengths: %d..%d\n", ds->minSeqLength(), ds->maxSeqLength());
    printf("Total timesteps:  %lld\n",     ds->totalTimesteps());
    printf("\n");

    return ds;
}


template <typename TDevice>
void printLayers(const NeuralNetwork<TDevice> &nn)
{
    int weights = 0;

    for (int i = 0; i < (int)nn.layers().size(); ++i) {
        printf("(%d) %s ", i, nn.layers()[i]->type().c_str());
        printf("[size: %d", nn.layers()[i]->size());

        const layers::TrainableLayer<TDevice>* tl = dynamic_cast<const layers::TrainableLayer<TDevice>*>(nn.layers()[i].get());
        if (tl) {
            printf(", bias: %.1lf, weights: %d", (double)tl->bias(), (int)tl->weights().size());
            weights += (int)tl->weights().size();
        }

        printf("]\n");
    }

    printf("Total weights: %d\n", weights);
}


template <typename TDevice>
void printOptimizer(const Configuration &config, const optimizers::lmOptimizer<TDevice> &optimizer)
{
    if (dynamic_cast<const optimizers::lmSteepestDescentOptimizer<TDevice>*>(&optimizer)) {
        printf("Optimizer type: Steepest descent with momentum\n");
        printf("Max training epochs:       %d\n", config.maxEpochs());
        printf("Max epochs until new best: %d\n", config.maxEpochsNoBest());
        printf("Validation error every:    %d\n", config.validateEvery());
        printf("Test error every:          %d\n", config.testEvery());
        printf("Learning rate:             %g\n", (double)config.learningRate());
        printf("Momentum:                  %g\n", (double)config.momentum());
        printf("\n");
    }
}


template <typename TDevice>
void saveNetwork(const NeuralNetwork<TDevice> &nn, const std::string &filename)
{
    rapidjson::Document jsonDoc;
    jsonDoc.SetObject();
    nn.exportLayers (&jsonDoc);
    nn.exportWeights(&jsonDoc);

    FILE *file = fopen(filename.c_str(), "w");
    if (!file)
        throw std::runtime_error("Cannot open file");

    rapidjson::FileStream os(file);
    rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
    jsonDoc.Accept(writer);

    fclose(file);
}


template <typename TDevice>
void saveState(const NeuralNetwork<TDevice> &nn, const optimizers::lmOptimizer<TDevice> &optimizer, const std::string &infoRows)
{
    // create the JSON document
    rapidjson::Document jsonDoc;
    jsonDoc.SetObject();

    // add the configuration options
    jsonDoc.AddMember("configuration", Configuration::instance().serializedOptions().c_str(), jsonDoc.GetAllocator());

    // add the info rows
    std::string tmp = boost::replace_all_copy(infoRows, "\n", ";;;");
    jsonDoc.AddMember("info_rows", tmp.c_str(), jsonDoc.GetAllocator());

    // add the network structure and weights
    nn.exportLayers (&jsonDoc);
    nn.exportWeights(&jsonDoc);

    // add the state of the optimizer
    optimizer.exportState(&jsonDoc);

    // open the file
    std::stringstream autosaveFilename;
    std::string prefix = Configuration::instance().autosavePrefix();
    autosaveFilename << prefix;
    if (!prefix.empty())
        autosaveFilename << '_';
    autosaveFilename << "epoch";
    autosaveFilename << std::setfill('0') << std::setw(3) << optimizer.currentEpoch();
    autosaveFilename << ".autosave";
    std::string autosaveFilename_str = autosaveFilename.str();
    FILE *file = fopen(autosaveFilename_str.c_str(), "w");
    if (!file)
        throw std::runtime_error("Cannot open file");

    // write the file
    rapidjson::FileStream os(file);
    rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
    jsonDoc.Accept(writer);
    fclose(file);
}


template <typename TDevice>
void restoreState(NeuralNetwork<TDevice> *nn, optimizers::lmOptimizer<TDevice> *optimizer, std::string *infoRows)
{
    rapidjson::Document jsonDoc;
    readJsonFile(&jsonDoc, Configuration::instance().continueFile());

    // extract info rows
    if (!jsonDoc.HasMember("info_rows"))
        throw std::runtime_error("Missing value 'info_rows'");
    *infoRows = jsonDoc["info_rows"].GetString();
    boost::replace_all(*infoRows, ";;;", "\n");

    // extract the state of the optimizer
    optimizer->importState(jsonDoc);
}


std::string printfRow(const char *format, ...)
{
    // write to temporary buffer
    char buffer[100];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // print on stdout
    std::cout << buffer;
    fflush(stdout);

    // return the same string
    return std::string(buffer);
}
