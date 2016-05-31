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


#include "../../currennt_lib/src/helpers/Matrix.hpp"
#include "../../currennt_lib/src/Configuration.hpp"
#include "../../currennt_lib/src/NeuralNetwork.hpp"
#include "../../currennt_lib/src/layers/LstmLayer.hpp"
#include "../../currennt_lib/src/layers/BinaryClassificationLayer.hpp"
#include "../../currennt_lib/src/layers/MulticlassClassificationLayer.hpp"

#include "../../currennt_lib/src/rnnlm/intInputLayer.hpp"
#include "../../currennt_lib/src/rnnlm/LookupLayer.hpp"
#include "../../currennt_lib/src/lmOptimizers/lmSteepestDescentOptimizer.hpp"
#include "../../currennt_lib/src/lmOptimizers/Adam.hpp"
// #include "../../currennt_lib/src/optimizers/SteepestDescentOptimizer.hpp"

#include "../../currennt_lib/src/helpers/JsonClasses.hpp"
#include "../../currennt_lib/src/rapidjson/prettywriter.h"
#include "../../currennt_lib/src/rapidjson/filestream.h"

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

#include <cereal/types/unordered_map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/binary.hpp>

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <stdarg.h>
#include <sstream>
#include <cstdlib>
#include <math.h>
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
boost::shared_ptr<data_sets::Corpus> loadDataSet(data_set_type dsType, const int max_vocab, std::unordered_map<std::string, int>* p_map=NULL, int constructDict = 0);
template <typename TDevice> void printLayers(const NeuralNetwork<TDevice> &nn);
template <typename TDevice> void printOptimizer(const optimizers::lmOptimizer<TDevice> &optimizer);
template <typename TDevice> void saveNetwork(const NeuralNetwork<TDevice> &nn, const std::string &filename);
void createModifiedTrainingSet(data_sets::Corpus *trainingSet, int parallelSequences, bool outputsToClasses, boost::mutex &swapTrainingSetsMutex);
template <typename TDevice> void saveState(const NeuralNetwork<TDevice> &nn, const optimizers::lmOptimizer<TDevice> &optimizer, const std::string &infoRows);
template <typename TDevice> void restoreState(NeuralNetwork<TDevice> *nn, optimizers::lmOptimizer<TDevice> *optimizer, std::string *infoRows);
std::string mpiprintfRow(const char *format, ...);
void exportDictBinary(const std::unordered_map<std::string, int> &m, std::string &fname);
void importDictBinary(std::unordered_map<std::string, int> &m, std::string &fname);
void mpiprintf(const char *format, ...);

// main function
template <typename TDevice>
int trainerMain(const Configuration &config)
{
    try {
        int rank = MPI::COMM_WORLD.Get_rank();
        mpiprintf("max_vocab_size: %d\n", config.max_vocab_size());
        // read the neural network description file
        std::string networkFile = config.continueFile().empty() ? config.networkFile() : config.continueFile();
        mpiprintf("Reading network from '%s'... ", networkFile.c_str());
        fflush(stdout);
        rapidjson::Document netDoc;
        std::string importdir = config.importDir();
        std::string savedir = config.exportDir();
        readJsonFile(&netDoc, networkFile);
        mpiprintf("done.\n");
        mpiprintf("\n");
        std::unordered_map<std::string, int> _wordDict;
        if (importdir != "") {
            std::string fname = importdir + "/wdict.cereal";
            importDictBinary(_wordDict, fname);
        }

        // loadDict(&netDoc, &_wordDict);

        // load data sets
        boost::shared_ptr<data_sets::Corpus> trainingSet    = boost::make_shared<data_sets::Corpus>();
        boost::shared_ptr<data_sets::Corpus> validationSet  = boost::make_shared<data_sets::Corpus>();
        boost::shared_ptr<data_sets::Corpus> testSet        = boost::make_shared<data_sets::Corpus>();
        boost::shared_ptr<data_sets::Corpus> feedForwardSet = boost::make_shared<data_sets::Corpus>();

        // not efficient
        if (_wordDict.empty()){
            trainingSet = loadDataSet(DATA_SET_TRAINING, config.max_vocab_size());
            _wordDict = *(trainingSet->dict()); // copy from trainingSet
        }
        else{
            // TODO add option fixed_dict();
            //    int iffix = config.fixed_dict()
            //    trainingSet = loadDataSet(DATA_SET_TRAINING, config.max_vocab_size(), &_wordDict, iffix);
            trainingSet = loadDataSet(DATA_SET_TRAINING, config.max_vocab_size(), &_wordDict, 0);
            _wordDict = *(trainingSet->dict());
        }

        if (rank == 0) {
            // use same wordDict as used in trainingSet
            if (!config.validationFiles().empty())
                validationSet = loadDataSet(DATA_SET_VALIDATION, config.max_vocab_size(), &_wordDict);

            if (!config.testFiles().empty())
                testSet = loadDataSet(DATA_SET_TEST, config.max_vocab_size(), &_wordDict);
        }

        // calculate the maximum sequence length
        int maxSeqLength;
        maxSeqLength = std::max(trainingSet->maxSeqLength(), std::max(validationSet->maxSeqLength(), testSet->maxSeqLength()));

        int parallelSequences = config.parallelSequences();

        // modify input and output size in netDoc to match the training set size
        // trainingSet->inputPatternSize
        // trainingSet->outputPatternSize

        // create the neural network
        mpiprintf("Creating the neural network... ");
        fflush(stdout);
        if (config.max_lookup_size() != -1)
            netDoc.AddMember("max_lookup_size", config.max_lookup_size(), netDoc.GetAllocator());

        int inputSize = -1;
        int outputSize = -1;
        inputSize = trainingSet->inputPatternSize();
        outputSize = trainingSet->outputPatternSize();
        int vocab_size = (int)_wordDict.size();
        // if (config.max_vocab_size() != -1 && config.max_vocab_size() < outputSize)
        //     outputSize = config.max_vocab_size();

        NeuralNetwork<TDevice> neuralNetwork(netDoc, parallelSequences, maxSeqLength, inputSize, outputSize, vocab_size, config.devices());
        neuralNetwork.setWordDict(&_wordDict);
        if (importdir != "")
            neuralNetwork.importWeightsBinary(importdir);
        if (config.pretrainedEmbeddings() != "")
            neuralNetwork.loadEmbeddings(config.pretrainedEmbeddings());
        if (config.fixedLookup())
            neuralNetwork.fixLookup();

        if (!trainingSet->empty() && trainingSet->outputPatternSize() != neuralNetwork.postOutputLayer().size())
            throw std::runtime_error("Post output layer size != target pattern size of the training set");
        if (!validationSet->empty() && validationSet->outputPatternSize() != neuralNetwork.postOutputLayer().size())
            throw std::runtime_error("Post output layer size != target pattern size of the validation set");
        if (!testSet->empty() && testSet->outputPatternSize() != neuralNetwork.postOutputLayer().size())
            throw std::runtime_error("Post output layer size != target pattern size of the test set");

            mpiprintf("done.\n");
            mpiprintf("Layers:\n");
        printLayers(neuralNetwork);
        mpiprintf("\n");

        // check if this is a classification task
        bool classificationTask = false;
        if (dynamic_cast<layers::BinaryClassificationLayer<TDevice>*>(&neuralNetwork.postOutputLayer())
            || dynamic_cast<layers::MulticlassClassificationLayer<TDevice>*>(&neuralNetwork.postOutputLayer())) {
                classificationTask = true;
        }

        mpiprintf("\n");

        // create the optimizer
        if (config.trainingMode()) {
            mpiprintf("Creating the optimizer... ");
            fflush(stdout);
            boost::scoped_ptr<optimizers::lmOptimizer<TDevice> > optimizer;

            // /*
            switch (config.optimizer()) {
            case Configuration::OPTIMIZER_STEEPESTDESCENT:
                optimizers::lmSteepestDescentOptimizer<TDevice> *sdo;
                sdo = new optimizers::lmSteepestDescentOptimizer<TDevice>(
                    neuralNetwork, *trainingSet, *validationSet, *testSet,
                    config.maxEpochs(), config.maxEpochsNoBest(), config.validateEvery(), config.testEvery(),
                    config.learningRate(), config.momentum(), config.temp_show()
                    );
                optimizer.reset(sdo);
                break;

            case Configuration::OPTIMIZER_ADAM:
                optimizers::Adam<TDevice> *adm;
                adm = new optimizers::Adam<TDevice>(
                        neuralNetwork, *trainingSet, *validationSet, *testSet,
                        config.maxEpochs(), config.maxEpochsNoBest(), config.validateEvery(), config.testEvery(),
                        config.learningRate()
                        );
                optimizer.reset(adm);
                break;

            default:
                throw std::runtime_error("Unknown optimizer type");
            }
            // */

            mpiprintf("done.\n");
            printOptimizer(config, *optimizer);

            std::string infoRows;
            // continue from autosave?

            if (!config.continueFile().empty()) {
                mpiprintf("Restoring state from '%s'... ", config.continueFile().c_str());
                fflush(stdout);
                restoreState(&neuralNetwork, &*optimizer, &infoRows);
                mpiprintf("done.\n\n");
            }

            // train the network
            mpiprintf("Starting training...\n");
            mpiprintf("\n");

            mpiprintf(" Epoch | Duration |  Training error  | Validation error |    Test error    | New best \n");
            mpiprintf("-------+----------+------------------+------------------+------------------+----------\n");
            std::cout << infoRows;

            bool finished = false;
            while (!finished) {
                const char *errFormat = (classificationTask ? "%6.2lf%%%10.3lf |" : "%17.3lf |");
                const char *errSpace  = "                  |";

                // train for one epoch and measure the time
                infoRows += mpiprintfRow(" %5d | ", optimizer->currentEpoch() + 1);

                boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::local_time();
                finished = optimizer->train();
                boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
                double duration = (double)(endTime - startTime).total_milliseconds() / 1000.0;

                infoRows += mpiprintfRow("%8.1lf |", duration);
                if (classificationTask)
                    infoRows += mpiprintfRow(errFormat, (double)optimizer->curTrainingClassError()*100.0, (double)optimizer->curTrainingError());
                else
                    infoRows += mpiprintfRow(errFormat, (double)optimizer->curTrainingError());

                if (!validationSet->empty() && optimizer->currentEpoch() % config.validateEvery() == 0) {
                    if (classificationTask)
                        infoRows += mpiprintfRow(errFormat, (double)optimizer->curValidationClassError()*100.0, (double)optimizer->curValidationError());
                    else
                        infoRows += mpiprintfRow(errFormat, (double)optimizer->curValidationError());
                }
                else
                    infoRows += mpiprintfRow("%s", errSpace);

                if (!testSet->empty() && optimizer->currentEpoch() % config.testEvery() == 0) {
                    if (classificationTask)
                        infoRows += mpiprintfRow(errFormat, (double)optimizer->curTestClassError()*100.0, (double)optimizer->curTestError());
                    else
                        infoRows += mpiprintfRow(errFormat, (double)optimizer->curTestError());
                }
                else
                    infoRows += mpiprintfRow("%s", errSpace);

                if (!validationSet->empty() && optimizer->currentEpoch() % config.validateEvery() == 0) {
                    if (optimizer->epochsSinceLowestValidationError() == 0) {
                        ///infoRows +=  mpiprintfRow("  yes   \n");
                        if (config.autosaveBest()) {
                            std::stringstream saveFileS;
                            if (config.autosavePrefix().empty()) {
                                size_t pos = config.networkFile().find_last_of('.');
                                if (pos != std::string::npos && pos > 0)
                                    saveFileS << config.networkFile().substr(0, pos);
                                else
                                    saveFileS << config.networkFile();
                            }
                            else{
                                saveFileS << config.autosavePrefix();
            			    }
                            saveFileS << ".best.jsn";
                            //saveNetwork(neuralNetwork, saveFileS.str());
                            neuralNetwork.exportWeightsBinary(savedir);
                        }
                        infoRows += mpiprintfRow(" yes \n");
                    }
                    else
                        infoRows += mpiprintfRow("  no    \n");
                }
                else
                    infoRows += mpiprintfRow("        \n");

                // autosave
                if (config.autosave())
                    saveState(neuralNetwork, *optimizer, infoRows);
            }

            mpiprintf("\n");

            if (optimizer->epochsSinceLowestValidationError() == config.maxEpochsNoBest())
                mpiprintf("No new lowest error since %d epochs. Training stopped.\n", config.maxEpochsNoBest());
            else
                mpiprintf("Maximum number of training epochs reached. Training stopped.\n");

            if (!validationSet->empty())
                mpiprintf("Lowest validation error: %lf\n", optimizer->lowestValidationError());
            else
                mpiprintf("Final training set error: %lf\n", optimizer->curTrainingError());
                mpiprintf("\n");

            // save the trained network to the output file
            //  mpiprintf("Storing the trained network in '%s'... ", config.trainedNetworkFile().c_str());
            mpiprintf("Storing the trained network in '%s'... ", savedir.c_str());
            // saveNetwork(neuralNetwork, config.trainedNetworkFile());
            neuralNetwork.exportWeightsBinary(savedir);
            std::string ofname = savedir + "/wdict.cereal";
            exportDictBinary(_wordDict, ofname);
            mpiprintf("done.\n");

            std::cout << "Removing cache file(s) ..." << std::endl;
            if (trainingSet != boost::shared_ptr<data_sets::Corpus>())
                boost::filesystem::remove(trainingSet->cacheFileName());
            if (validationSet != boost::shared_ptr<data_sets::Corpus>())
                boost::filesystem::remove(validationSet->cacheFileName());
            if (testSet != boost::shared_ptr<data_sets::Corpus>())
                boost::filesystem::remove(testSet->cacheFileName());
        }
    }
    catch (const std::exception &e) {
        mpiprintf("FAILED: %s\n", e.what());
        return 2;
    }

    return 0;
}


int main(int argc, const char *argv[])
{
    // Configuration config(argc, argv);
    MPI::Init(argc, argv);
    int ret;
    // load the configuration
    char** config_arg[3][20];
    config_arg[0] = "mpi_learn";
    config_arg[1] = "--option_file";
    config_arg[2] = "config.cfg";
    Configuration config(3, config_arg);

    // run the execution device specific main function
    if (config.useCuda()) {
        int count;
        cudaError_t err;
        if (config.listDevices()) {
            if ((err = cudaGetDeviceCount(&count)) != cudaSuccess) {
                std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
                return err;
            }
            std::cout << count << " devices found" << std::endl;
            cudaDeviceProp prop;
            for (int i = 0; i < count; ++i) {
                if ((err = cudaGetDeviceProperties(&prop, i)) != cudaSuccess) {
                    std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
                    return err;
                }
                std::cout << i << ": " << prop.name << std::endl;
            }
            return 0;
        }
        int device = 0;
        char* dev = std::getenv("CURRENNT_CUDA_DEVICE");
        if (dev != NULL) {
            device = std::atoi(dev);
        }
        cudaDeviceProp prop;
        if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) {
            std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        std::cout << "Using device #" << device << " (" << prop.name << ")" << std::endl;
        if ((err = cudaSetDevice(device)) != cudaSuccess) {
            std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        ret = trainerMain<Gpu>(config);
    }
    else
        ret = trainerMain<Cpu>(config);
    MPI::Finalize();
    return ret;
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
    mpiprintf("%lld words are loaded from json file.", (long long int) p_dict->size());

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

    mpiprintf("Loading %s ", type.c_str());
    for (std::vector<std::string>::const_iterator fn_itr = filenames.begin();
         fn_itr != filenames.end(); ++fn_itr)
    {
        mpiprintf("'%s' ", fn_itr->c_str());
    }
    mpiprintf("...");
    fflush(stdout);
    mpiprintf("maxvocab: %d\n", max_vocab);
    //std::cout << "truncating to " << truncSeqLength << std::endl;
    boost::shared_ptr<data_sets::Corpus> ds = boost::make_shared<data_sets::Corpus>(
        filenames,
        Configuration::instance().parallelSequences(), fraction, truncSeqLength,
        fracShuf, seqShuf, noiseDev, cachePath, p_map, constructDict, max_vocab);

        mpiprintf("done.\n");
        mpiprintf("Loaded fraction:  %d%%\n",   (int)(fraction*100));
        mpiprintf("Sequences:        %lld\n",     ds->totalSequences());
        mpiprintf("Sequence lengths: %d..%d\n", ds->minSeqLength(), ds->maxSeqLength());
        mpiprintf("Total timesteps:  %lld\n",     ds->totalTimesteps());
        mpiprintf("\n");

    return ds;
}


template <typename TDevice>
void printLayers(const NeuralNetwork<TDevice> &nn)
{
    int weights = 0;

    for (int i = 0; i < (int)nn.layers().size(); ++i) {
        mpiprintf("(%d) %s ", i, nn.layers()[i]->type().c_str());
        mpiprintf("[size: %d", nn.layers()[i]->size());

        const layers::TrainableLayer<TDevice>* tl = dynamic_cast<const layers::TrainableLayer<TDevice>*>(nn.layers()[i].get());
        if (tl) {
            mpiprintf(", bias: %.1lf, weights: %d", (double)tl->bias(), (int)tl->weights().size());
            weights += (int)tl->weights().size();
        }

        mpiprintf("]\n");
    }

    mpiprintf("Total weights: %d\n", weights);
}


template <typename TDevice>
void printOptimizer(const Configuration &config, const optimizers::lmOptimizer<TDevice> &optimizer)
{
    if (dynamic_cast<const optimizers::lmSteepestDescentOptimizer<TDevice>*>(&optimizer)) {
        mpiprintf("Optimizer type: Steepest descent with momentum\n");
        mpiprintf("Max training epochs:       %d\n", config.maxEpochs());
        mpiprintf("Max epochs until new best: %d\n", config.maxEpochsNoBest());
        mpiprintf("Validation error every:    %d\n", config.validateEvery());
        mpiprintf("Test error every:          %d\n", config.testEvery());
        mpiprintf("Learning rate:             %g\n", (double)config.learningRate());
        mpiprintf("Momentum:                  %g\n", (double)config.momentum());
        mpiprintf("\n");
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
/*
template <typename TDevice>
void saveNetworkBinary(const NeuralNetwork<TDevice> &nn, const std::string &dirname, const std::string &filename)
{
    nn.exportWeightsBinary(dirname);

}
*/

void exportDictBinary(const std::unordered_map<std::string, int> &m, std::string &fname)
{
    std::ofstream ofs(fname, std::ios::binary);
    cereal::BinaryOutputArchive archive(ofs);
    archive(m);
}

void importDictBinary(std::unordered_map<std::string, int> &m, std::string &fname)
{
    std::ifstream ifs(fname, std::ios::binary);
    cereal::BinaryInputArchive archive(ifs);
    archive(m);
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


std::string mpiprintfRow(const char *format, ...)
{
    if (rank != 0) return;
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


void mpiprintf(const char *format, ...)
{
    if (rank != 0) return;
    char buffer[100];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    std::cout << buffer;
    fflush(stdout);
}
