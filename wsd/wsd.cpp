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
#include "../currennt_lib/src/helpers/Matrix.hpp"
#include "../currennt_lib/src/Configuration.hpp"
#include "../currennt_lib/src/NeuralNetwork.hpp"
#include "../currennt_lib/src/layers/LstmLayer.hpp"
#include "../currennt_lib/src/layers/BinaryClassificationLayer.hpp"
#include "../currennt_lib/src/layers/MulticlassClassificationLayer.hpp"

#include "../currennt_lib/src/corpus/CorpusFraction.hpp"

#include "../currennt_lib/src/rnnlm/intInputLayer.hpp"
#include "../currennt_lib/src/rnnlm/LookupLayer.hpp"
#include "../currennt_lib/src/lmOptimizers/lmSteepestDescentOptimizer.hpp"
// #include "../../currennt_lib/src/optimizers/SteepestDescentOptimizer.hpp"

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
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include <fstream>
#include <stdexcept>
#include <tuple>
#include <algorithm>
#include <algorithm>
#include <stdarg.h>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <cfloat>
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

enum POS_type_t
{
    NOUN,
    VERB,
    ADJ,
    ADV,
    POS_OTHER
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
std::string printfRow(const char *format, ...);
void loadLexemes(const std::string& filename,
                 std::unordered_map< std::string, std::vector<std::string> >& word_synsets,
                 std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>>& lexeme_emb);
void readLine(std::string line, std::vector<std::string>& words, std::vector<POS_type_t>& pos);
void makeFrac(const std::vector<std::string>& words,
              const std::unordered_map<std::string, int>& dic,
              boost::shared_ptr<data_sets::CorpusFraction> frac,
              int size);

std::string wsd(const std::string& word,
                const POS_type_t& pos,
                const std::vector<std::string>& synsets,
                Cpu::real_vector& output,
                const std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>>& lexeme_emb);

// main function
template <typename TDevice>
int trainerMain(const Configuration &config)
{
    try {
        printf("max_vocab_size: %d\n", config.max_vocab_size());
        // read the neural network description file
        std::string networkFile = config.continueFile().empty() ? config.networkFile() : config.continueFile();
        printf("Reading network from '%s'... ", networkFile.c_str());
        fflush(stdout);
        rapidjson::Document netDoc;
        readJsonFile(&netDoc, networkFile);
        printf("done.\n");
        printf("\n");
        std::unordered_map<std::string, int> _wordDict;
        loadDict(&netDoc, &_wordDict);

        // load data sets
        boost::shared_ptr<data_sets::Corpus> trainingSet    = boost::make_shared<data_sets::Corpus>();
        boost::shared_ptr<data_sets::Corpus> validationSet  = boost::make_shared<data_sets::Corpus>();
        boost::shared_ptr<data_sets::Corpus> testSet        = boost::make_shared<data_sets::Corpus>();
        boost::shared_ptr<data_sets::Corpus> feedForwardSet = boost::make_shared<data_sets::Corpus>();


        testSet = loadDataSet(DATA_SET_TEST, config.max_vocab_size(), &_wordDict, 0);
        // calculate the maximum sequence length
        int maxSeqLength;
        maxSeqLength = testSet->maxSeqLength();

        int parallelSequences = config.parallelSequences();
        std::unordered_map<std::string, std::vector<std::string>> word_synsets;
        std::unordered_map<std::string, std::unique_ptr<Cpu::real_vector>> lexeme_emb;
        if (config.lexeme_file() == "")
            throw std::runtime_error("specify lexeme_parameters");
        loadLexemes(config.lexeme_file(), word_synsets, lexeme_emb);
        // modify input and output size in netDoc to match the training set size
        // trainingSet->inputPatternSize
        // trainingSet->outputPatternSize

        // create the neural network
        printf("Creating the neural network... ");
        fflush(stdout);
        if (config.max_lookup_size() != -1)
            netDoc.AddMember("max_lookup_size", config.max_lookup_size(), netDoc.GetAllocator());

        int inputSize = -1;
        int outputSize = -1;
        inputSize = testSet->inputPatternSize();
        outputSize = testSet->outputPatternSize();
        int vocab_size = (int)_wordDict.size();
        // if (config.max_vocab_size() != -1 && config.max_vocab_size() < outputSize)
        //     outputSize = config.max_vocab_size();

        NeuralNetwork<TDevice> neuralNetwork(netDoc, parallelSequences, maxSeqLength, inputSize, outputSize, vocab_size, config.devices());
        neuralNetwork.setWordDict(&_wordDict);
        if (config.pretrainedEmbeddings() != "")
            neuralNetwork.loadEmbeddings(config.pretrainedEmbeddings());

        printf("done.\n");
        printf("Layers:\n");
        printLayers(neuralNetwork);
        printf("\n");

        // check if this is a classification task
        bool classificationTask = false;
        if (dynamic_cast<layers::BinaryClassificationLayer<TDevice>*>(&neuralNetwork.postOutputLayer())
            || dynamic_cast<layers::MulticlassClassificationLayer<TDevice>*>(&neuralNetwork.postOutputLayer())) {
                classificationTask = true;
        }

        printf("\n");

        // process all data set fractions
        int fracIdx = 0;
        printf("open test-file and output-result-file\n");
        std::ifstream ifs(Configuration::instance().testFiles()[0]);
        std::ofstream ofs(Configuration::instance().wsdResult());
        std::string line, word, wsd_result;
        POS_type_t _p;
        std::vector<std::string> words;
        std::vector<POS_type_t> pos;
        // std::vector<int> POS;
        int d = (int)(lexeme_emb.begin()->second)->size();
        Cpu::real_vector output_;
        output_.reserve(d);
        typename TDevice::real_vector outputs;
        printf("start wsd\n");
        while (std::getline(ifs, line)) {
            // printf("Computing outputs for data fraction %d...", ++fracIdx);
            // fflush(stdout);
            boost::shared_ptr<data_sets::CorpusFraction> frac = testSet->getNewFrac();
            readLine(line, words, pos);
            // printf("make Fraction\n");
            makeFrac(words, _wordDict, frac, outputSize);
            // compute the forward pass for the current data fraction and extract the outputs
            // printf("compute forward\n");
            neuralNetwork.loadSequences(*frac);
            neuralNetwork.computeForwardPass();
            outputs = neuralNetwork.last_layer();
            // output_ = outputs; //copying
            // thrust::copy(outputs.begin(), outputs.size(), output_.begin());

            // write one output file per sequence
            // printf("wsd\n");
            // printf("outputsize:%d, dimension:%d, hiddenlayer_size:%d\n", outputs.size(), d, output_.size());
            for (int i = 0; i < words.size(); ++i) {
                thrust::copy(outputs.begin() + i * d, outputs.begin() + (i+1) * d, output_.begin());

                word = words.at(i);
                _p = pos.at(i);
                if (word_synsets.find(word) == word_synsets.end())
                    wsd_result = word;
                else if (_p == POS_OTHER)
                    wsd_result = word;
                else
                    wsd_result = wsd(word, _p, word_synsets[word], output_, lexeme_emb);
                std::cout << "word: " << word << " || result: " << wsd_result << std::endl;
                ofs << wsd_result << " ";
            }
            ofs << std::endl;

            // printf(" done.\n");
        }
        // boost::filesystem::remove(feedForwardSet->cacheFileName());
    }
    catch (const std::exception &e) {
        printf("FAILED: %s\n", e.what());
        return 2;
    }

    return 0;
}


int main(int argc, const char *argv[])
{
    // load the configuration
    Configuration config(argc, argv);

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
        return trainerMain<Gpu>(config);
    }
    else
        return trainerMain<Cpu>(config);
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
        filenames = Configuration::instance().testFiles();
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

// return if char c is conained by string s
bool ifContain(std::string s, std::string c) {
    return (s.find(c) != std::string::npos);
}

// store spliting result to target
void strSplit(std::string& s, std::vector<std::string> *target, char delim) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim))
        target->push_back(item);
}

bool ifLexeme(std::string word) {
    if ( !ifContain(word, "%") )
        return false;
    std::vector<std::string> v;
    strSplit(word, &v, '-');
    for(std::string s : v)
        if (ifContain(s, "%"))
            return true;
    return false;
}

void getWordSynsetOfLexeme(std::string word, std::vector<std::string> *ret) {
    ret->resize(2);
    std::vector<std::string> v;
    strSplit(word, &v, '-');
    std::string w, syn;
    if (v.size() == 2){
        w = v.at(0);
        syn = v.at(1);
    }
    else if (v.size() > 2) {
        w = v.at(0) + v.at(1);
        syn = "";
        for (int i = 2; i < v.size(); ++i)
            syn += v.at(i);
    }
    ret->at(0) = w;
    ret->at(1) = syn;
}

void loadLexemes(const std::string& filename,
                 std::unordered_map< std::string, std::vector<std::string> >& word_synsets,
                 std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>>& lexeme_emb)
{

    std::ifstream ifs(filename);
    std::vector<std::string> Nd;
    std::string line, item, lexeme, word, syn;
    int d;
    char delim = ' ';
    std::getline(ifs, line);
    strSplit(line, &Nd, delim);
    d = std::stoi(Nd[1]);
    std::vector<std::string> word_syn;
    printf("loading lexeme embeddings... \n");
    while(std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::getline(ss, lexeme, delim);
        if (!ifLexeme(lexeme)) {
            printf("not lexeme:%s\n", lexeme.c_str());
            continue;
        }

        std::unique_ptr<Cpu::real_vector> array(new Cpu::real_vector());
        array->reserve(d);
        // lexeme_emb[lexeme] = std::unique_ptr<Cpu::real_vector>();
        // lexeme_emb[lexeme]->reserve(d);

        getWordSynsetOfLexeme(lexeme, &word_syn);
        word = word_syn[0];
        syn = word_syn[1];

        if (word_synsets.find(word) == word_synsets.end()) {
            word_synsets[word] = std::vector<std::string>();
        }
        word_synsets[word].push_back(lexeme);


        while(std::getline(ss, item, delim))
            array->push_back(std::stof(item));
            // lexeme_emb[lexeme]->push_back(std::stof(item));
        lexeme_emb[lexeme] = std::move(array);

    }
    printf("done. \n\n" );
}

POS_type_t getPOStype(std::string postype)
{
    if (postype == "NOUN")
        return NOUN;
    else if (postype == "VERB")
        return VERB;
    else if (postype == "ADJ")
        return ADJ;
    else if (postype == "ADV")
        return ADV;
    else
        return POS_OTHER;
}
void readLine(std::string line, std::vector<std::string>& words, std::vector<POS_type_t>& pos)
{
    words.clear();
    std::stringstream ss(line);
    std::string word, item;
    POS_type_t postag;
    while (std::getline(ss, word, ' ')) {
        postag = POS_OTHER;
        std::stringstream lempos(word);
        std::getline(lempos, item, '@');
        words.push_back(item);
        std::getline(lempos, item, '@');
        postag = getPOStype(item);
        pos.push_back(postag);
    }
}

void makeInput(const std::vector<std::string>& words,
              const std::unordered_map<std::string, int>& dic,
              Cpu::int_vector *input)
{
    input->reserve(words.size());
    std::string unk = "<UNK>";
    int unk_int = dic.find(unk)->second;
    for (auto w : words) {
        auto it = dic.find(w);
        if (it == dic.end())
            input->push_back(unk_int);
        else
            input->push_back(it->second);
    }
}
void makeFrac(const std::vector<std::string>& words,
              const std::unordered_map<std::string, int>& dic,
              boost::shared_ptr<data_sets::CorpusFraction> frac,
              int size)
{
    int context_left = Configuration::instance().inputLeftContext();
    int context_right = Configuration::instance().inputRightContext();
    int context_length = context_left + context_right + 1;
    int output_lag = Configuration::instance().outputTimeLag();

    printf("setting fraction\n");
    frac->use_intInput();
    frac->set_inputPatternSize(context_length);
    // frac->m_outputPatternSize = m_outputPatternSize;
    frac->set_outputPatternSize(size);
    // frac->m_maxSeqLength      = std::numeric_limits<int>::min();
    frac->set_maxSeqLength(words.size());
    // frac->m_minSeqLength      = std::numeric_limits<int>::max();
    frac->set_minSeqLength(words.size());

    // fill fraction sequence info
            // frac->m_minSeqLength = std::min(frac->m_minSeqLength, m_sequences[seqIdx].length);

    data_sets::CorpusFraction::seq_info_t seqInfo;
    seqInfo.originalSeqIdx = 1;
    seqInfo.length         = words.size();
    seqInfo.seqTag         = "";

    // frac->m_seqInfo.push_back(seqInfo);
    frac->set_seqInfo(seqInfo);

    // allocate memory for the fraction
    frac->intinput()->resize(frac->maxSeqLength() * frac->inputPatternSize(), 0);
    // resize of targetclass and patTypes
    frac->vectorResize(1, 0, -1);

    // inputs
    printf("make input\n");
    Cpu::int_vector input = Cpu::int_vector();
    makeInput(words, dic, &input);

    printf("place vector to memory\n");
    for (int timestep = 0; timestep < words.size(); ++timestep) {
        int offset_out = 0;
        for (int offset_in = -context_left; offset_in <= context_right; ++offset_in) {
            int srcStart = (timestep + offset_in);
            // duplicate first time step if needed
            if (srcStart < 0)
                srcStart = 0;
            // duplicate last time step if needed
            else if (srcStart > (words.size() - 1))
                srcStart = (words.size() - 1);
            int tgtStart = frac->inputPatternSize() * (timestep) + offset_out;
            //std::cout << "copy from " << srcStart << " to " << tgtStart << " size " << m_inputPatternSize << std::endl;
            thrust::copy_n(input.begin() + srcStart, 1, frac->intinput()->begin() + tgtStart);
            ++offset_out;
        }
    }
        /*std::cout << "original inputs: ";
        thrust::copy(inputs.begin(), inputs.end(), std::ostream_iterator<real_t>(std::cout, ";"));
        std::cout << std::endl;*/

        //  target classes
        //  only classification
    /*
    Cpu::int_vector targetClasses = _loadTargetClassesFromCache(seq);
    for (int timestep = 0; timestep < words.size(); ++timestep) {
        int tgt = 0; // default class (make configurable?)
        if (timestep >= output_lag)
            tgt = targetClasses[timestep - output_lag];
        // frac->m_targetClasses[timestep * m_parallelSequences + i] = tgt;
        frac->setTargetClasses(timestep * m_parallelSequences + i, tgt);
    }
    // pattern types
    for (int timestep = 0; timestep < words.size(); ++timestep) {
        Cpu::pattype_vector::value_type patType;
        if (timestep == 0)
            patType = PATTYPE_FIRST;
        else if (timestep == words.size() - 1)
            patType = PATTYPE_LAST;
        else
            patType = PATTYPE_NORMAL;

        // frac->m_patTypes[timestep * m_parallelSequences + i] = patType;
        frac->setPatTypes(timestep * m_parallelSequences + i, patType);
    }*/
}
//
// inline float logsumexp(float x, float y)
// {
//     return std::max(x, y) + log( 1 + exp(-fabs(x - y)) );
// }

inline double logsumexp(double x, double y)
{
    return std::max(x, y) + log( 1 + exp(-fabs(x - y)) );
}
template <typename M>
void logSoftmax(M& m, const int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; ++i)
        sum = logsumexp(sum, m(i));
    // m = (m.array() - sum).exp(); // m = log( softmax(m) ) : m - log(sum(exp(m)))
    m = (m.array() - sum); // m = log( softmax(m) ) : m - log(sum(exp(m)))
    // for (int i = 0; i < size; ++i)
    //     m(i) = exp(m(i));
}

template <typename M>
bool nanCheck(M& m, const int size)
{
    for (int i = 0; i < size; ++i)
        if ( std::isnan(m(i)) ) return true;
    return false;
}

template <typename M>
void checkinf(M& m, const int size)
{
    int infarg = -1;
    bool ifplus;
    for (int i = 0; i < size; ++i) {
        if (std::isinf( m(i) )) {
            infarg = i;
            ifplus = m(i) > 0;
        }
    }
    if (infarg == -1)
        return;
    for (int i = 0; i < size; ++i) {
        if (i == infarg) {
            if (ifplus) m(i) = 1;
            else        m(i) = 0;
        }
        else
            if (ifplus) m(i) = 0;
    }
}
std::tuple<std::string, POS_type_t> getLexemeSynset(std::string word, std::string lexeme)
{
    std::string synsets;
    std::stringstream lexss(lexeme);
    std::getline(lexss, synsets, '-');
    std::getline(lexss, synsets, '-');

    std::stringstream ss(synsets);
    std::vector<std::string> v;
    std::string item, w, key;
    // std::cout << "getLexemeSynset: " << lexeme << "  :  " << word << " : " << synsets << std::endl;
    POS_type_t p = POS_OTHER;
    while (std::getline(ss, item, ',')) {
        v.push_back(item);
    }
    for (std::string s : v) {
        // s = "word"(w)%"sensekey"(key) #six%1:23:00::
        if (s == "") continue;
        std::stringstream temp(s);
        std::getline(temp, w, '%'); //get word
        if (w != word) continue;
        std::getline(temp, key, '%'); //get sensekey
        std::stringstream temp2(key);
        std::getline(temp2, item, ':'); // item = pos-info
        // std::cout << key << "'s pos_information:" << item << std::endl;

        switch (std::stoi(item)) {
            case 1:
                p = NOUN;
                break;
            case 2:
                p = VERB;
                break;
            case 3:
                p = ADJ;
                break;
            case 4:
                p = ADV;
                break;
            case 5:
                p = ADJ;
                break;
        }
        return std::make_tuple(s, p);
    }
    return std::make_tuple("", p);

}

std::string wsd(const std::string& word,
                const POS_type_t& pos,
                const std::vector<std::string>& synsets,
                Cpu::real_vector& output,
                const std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>>& lexeme_emb)
{

    // Cpu::real_vector W;
    int d = lexeme_emb.begin()->second->size();

    int num_syn = synsets.size();
    // Eigen::Matrix<real_t, num_syn, d, ColMajor> W_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_(num_syn, d);
    // Eigen::MatrixXd W_(num_syn, d);
    // W.resize(num_syn * d);
    // int i = 0;
    for (auto lex : synsets) {
        // thrust::copy(lexeme_emb[lex]->begin(), d, W.begin() + i * d);
        auto embit = lexeme_emb.find(lex);

        if (embit == lexeme_emb.end()) {

            for ( int j = 0; j < d; ++j)
                // W_ << 0.0f;
                W_ << 0.0;
            continue;
        }

        for ( int j = 0; j < d; ++j) {
            W_ << (double)(*(embit->second))[j];

        }
        // ++i;
    }
    // std::cout << std::endl;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h(d, 1);
    // Eigen::MatrixXd h(d, 1);
    // Eigen::Matrix<double, d, 1, ColMajor> h;
    for (int j = 0; j < d; ++j)
        h << (double)output[j];

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> result(num_syn, 1);
    // Eigen::MatrixX result(num_syn, 1);
    // Eigen::Matrix<float, num_syn, 1, ColMajor> result;


    /*
    printf("calculate score \n");
    printf("size: numsyn:%d  d:%d\n", num_syn, d);
    printf("W_:\n");
    std::cout << W_ << std::endl;
    printf("h:\n");
    std::cout << h << std::endl;
    */

    //check nan
    assert( !nanCheck(W_, d) );

    if (nanCheck(h, d)) {
        printf("hidden layer has nan, skipping this word\n");
        return word;
    }


    result = W_ * h;

    if (nanCheck(result, num_syn)) {
        printf("scores has nan, skipping this word\n");
        return word;
    }

    printf("result:\n");
    std::cout << result << std::endl;

    checkinf(result, num_syn);

    logSoftmax(result, num_syn);

    printf("result:\n");
    std::cout << result << std::endl;

    double max = -DBL_MAX;
    int maxarg = -1;
    std::string max_syn;

    std::string syn;
    POS_type_t pos_;
    int counting = 0;
    for (int i = 0; i < num_syn; ++i) {
        std::tie(syn, pos_) = getLexemeSynset(word, synsets[i]);
        if (pos != pos_) continue;
        ++counting;
        if (max < result(i)) {
            max = result(i);
            max_syn = syn;
            maxarg = i;
        }
    }
    assert(maxarg != -1 || counting == 0);
    // return synsets[maxarg];
    if (counting == 0)
        return word;
    return max_syn;

}
