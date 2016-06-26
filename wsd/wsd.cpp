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


#include <cereal/types/unordered_map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/binary.hpp>

#include <fstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <algorithm>
#include <algorithm>
#include <stdarg.h>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <cfloat>
#include <iomanip>

namespace beam { // namespace for class 'Beam_state'

    class Beam_state
    {
    private:
        std::shared_ptr<std::vector<std::string>> m_created_words;
        int m_max_size;
        int m_position;
        double m_score;
        double m_wsdscores;
    public:
        Beam_state(int max_size);
        Beam_state(const std::vector<std::string>& v, double score, int max_size, double wsdscores=0.0);
        Beam_state(const Beam_state& bs);
        // getter
        double score() const;
        double wsdscore() const;
        std::shared_ptr<std::vector<std::string>> words();
        const std::shared_ptr<std::vector<std::string>> cwords() const;
        int state_length() const;
        int position() const;

        void transition(std::string& word, double score);
        //setter
        void set_score(const double& score);
        void set_posi(const int& position);
        void set_words(const std::vector<std::string>& v);
        bool operator>(const Beam_state& bs) const;
        bool operator<(const Beam_state& bs) const;
    };

    Beam_state::Beam_state(int max_size)
    {
        m_created_words = std::make_shared<std::vector<std::string>>();
        m_created_words->reserve(max_size);
        m_score = 0.0;
        m_wsdscores = 0.0;
        m_max_size = max_size;
        m_position = 0;
    }

    Beam_state::Beam_state(const std::vector<std::string>& v, double score, int max_size, double wsdscores)
    {
        m_created_words = std::make_shared<std::vector<std::string>>(v.size());
        m_created_words->reserve(max_size);
        std::copy( v.begin(), v.end(), m_created_words->begin() );
        m_score = score;
        m_wsdscores = wsdscores;
        m_max_size = max_size;
        m_position = 0;
    }

    // copy constructor
    Beam_state::Beam_state(const Beam_state& bs)
    {
        m_score = bs.m_score;
        m_max_size = bs.m_max_size;
        m_position = bs.m_position;
        m_wsdscores = bs.m_wsdscores;
        m_created_words = std::make_shared<std::vector<std::string>>(bs.state_length());
        m_created_words->reserve(m_max_size);
        std::copy( bs.cwords()->begin(), bs.cwords()->end(), m_created_words->begin() );
    }

    //getter
    double Beam_state::score() const { return m_score; }
    double Beam_state::wsdscore() const { return m_wsdscores; }
    std::shared_ptr<std::vector<std::string>> Beam_state::words() { return m_created_words; }
    const std::shared_ptr<std::vector<std::string>> Beam_state::cwords() const { return m_created_words; }
    int Beam_state::state_length() const { return (int)m_created_words->size(); }
    int Beam_state::position() const { return m_position; }
    //setter
    void Beam_state::set_score(const double& score) { m_score = score; }
    void Beam_state::set_posi(const int& position) { m_position = position; };
    void Beam_state::set_words(const std::vector<std::string>& v)
    {
        if (m_created_words->size() < v.size())
            m_created_words->resize(v.size());
        m_created_words->clear();
        std::copy( v.begin(), v.end(), m_created_words->begin() );
    }

    void Beam_state::transition(std::string& word, double score)
    {
        m_created_words->push_back(word);
        m_score = score;
    }

    bool Beam_state::operator>(const Beam_state& bs) const { return m_score > bs.score(); }
    bool Beam_state::operator<(const Beam_state& bs) const { return m_score < bs.score(); }

} // end of namespace beam

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
void importDictBinary(std::unordered_map<std::string, int> &m, std::string &fname);
void loadLexemes(const std::string& filename,
                 std::unordered_map< std::string, std::vector<std::string> >& word_synsets,
                 std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>>& lexeme_emb);
void readLine(std::string line, std::vector<std::string>& words, std::vector<POS_type_t>& pos);
void readLine_(std::string line, std::vector<std::string>& words);
void makeFrac(const std::vector<std::string>& words,
              const std::unordered_map<std::string, int>& dic,
              boost::shared_ptr<data_sets::CorpusFraction> frac,
              int size);

std::tuple<std::string, POS_type_t> getLexemeSynset(std::string word, std::string lexeme);
std::string getsynset(const std::string& lexeme);

std::string wsd(const std::string& word,
                const POS_type_t& pos,
                const std::vector<std::string>& synsets,
                Cpu::real_vector& output,
                const std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>>& lexeme_emb,
                double thr = 0.0,
                int *wsd_count = NULL,
                double *logprob = NULL,
                int bestK = 1,
                std::vector<std::shared_ptr<std::pair<std::string,double>>> *v = NULL);

template <typename TDevice>
int simple_wsd(NeuralNetwork<Cpu>& neuralNetwork,
               const std::string& wsd_input,
               const std::string& wsd_output,
               std::unordered_map<std::string, std::vector<std::string>> &word_synsets,
               const std::unordered_map<std::string, std::unique_ptr<Cpu::real_vector>> &lexeme_emb,
               double thr,
               boost::shared_ptr<data_sets::Corpus> &testSet,
               const std::unordered_map<std::string, int>& _wordDict2);

template <typename TDevice>
int beam_wsd(NeuralNetwork<Cpu>& neuralNetwork,
             const std::string& wsd_input,
             const std::string& wsd_output,
             std::unordered_map<std::string, std::vector<std::string>> &word_synsets,
             const std::unordered_map<std::string, std::unique_ptr<Cpu::real_vector>> &lexeme_emb,
             double thr,
             int beam_size,
             boost::shared_ptr<data_sets::Corpus> &testSet,
             const std::unordered_map<std::string, int>& _wordDict2);



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
        std::string importdir = config.importDir();
        readJsonFile(&netDoc, networkFile);
        printf("done.\n");
        printf("\n");
        std::unordered_map<std::string, int> _wordDict;
        std::unordered_map<std::string, int> _wordDict2;
        // loadDict(&netDoc, &_wordDict);

        if (importdir != "") {
            std::string fname = importdir + "/wdict.cereal";
            importDictBinary(_wordDict, fname);
        }

        _wordDict2 = _wordDict;

        // load data sets
        boost::shared_ptr<data_sets::Corpus> trainingSet    = boost::make_shared<data_sets::Corpus>();
        boost::shared_ptr<data_sets::Corpus> validationSet  = boost::make_shared<data_sets::Corpus>();
        boost::shared_ptr<data_sets::Corpus> testSet        = boost::make_shared<data_sets::Corpus>();
        boost::shared_ptr<data_sets::Corpus> feedForwardSet = boost::make_shared<data_sets::Corpus>();


        // calculate
        if (!config.validationFiles().empty())
            testSet = loadDataSet(DATA_SET_VALIDATION, config.max_vocab_size(), &_wordDict2, 0);
        else// wsd
            testSet = loadDataSet(DATA_SET_TEST, config.max_vocab_size(), &_wordDict2, 1);
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
        outputSize = _wordDict.size();
        // int vocab_size = (int)_wordDict.size();
        int vocab_size = (int)_wordDict2.size();
        printf("vocab_size:%d\n", vocab_size);
        // if (config.max_vocab_size() != -1 && config.max_vocab_size() < outputSize)
        //     outputSize = config.max_vocab_size();

        NeuralNetwork<TDevice> neuralNetwork(netDoc, parallelSequences, maxSeqLength, inputSize, outputSize, vocab_size, config.devices());
        // neuralNetwork.setWordDict(&_wordDict);
        neuralNetwork.setWordDict(&_wordDict2);


        // if (config.fixedLookup())
        neuralNetwork.fixLookup();

        printf("done.\n");
        printf("loading layers weights... ");
        fflush(stdout);
        if (importdir != "")
            neuralNetwork.importWeightsBinary(importdir, &_wordDict);
        printf("done.\n");

        // loading lexeme-embeddings to LookupLayer
        std::unordered_map<std::string, int> wdic_lex;
        wdic_lex = _wordDict2;
        int c = (int)wdic_lex.size();
        std::string syn_;
        for (auto it = lexeme_emb.begin(); it != lexeme_emb.end(); ++it) {
            syn_ = getsynset(it->first);
            if (wdic_lex.find(syn_) == wdic_lex.end())
                wdic_lex[syn_] = c++;
        }
        neuralNetwork.setWordDict(&wdic_lex);
        if (config.pretrainedEmbeddings() != "")
            neuralNetwork.loadEmbeddings(config.pretrainedEmbeddings());
        // neuralNetwork.loadEmbeddings(config.lexeme_file());
        for (auto it = lexeme_emb.begin(); it != lexeme_emb.end(); ++it) {
            syn_ = getsynset(it->first);
            neuralNetwork.lookupLayer().replaceEmbeddings(syn_, *(it->second));
        }

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

        if (!config.validationFiles().empty()) {
            int fracIdx = 0;
            boost::shared_ptr<data_sets::CorpusFraction> frac;
            real_t error, cross_entropy;
            real_t average_error = 0.0;
            int c = 0;
            std::ifstream ifs(Configuration::instance().validationFiles()[0]);
            std::ofstream ofs(Configuration::instance().wsdResult());
                // /*
            std::string line, word, wsd_result;
            std::vector<std::string> words = std::vector<std::string>();
            std::vector<int> words_len = std::vector<int>();
            while( std::getline(ifs, line) ){
                // boost::shared_ptr<data_sets::CorpusFraction> frac = testSet->getNewFrac();
                readLine_(line, words);
                words_len.push_back(words.size());
            }
                // makeFrac(words, _wordDict2, frac, outputSize);
                // */

            while (((frac = testSet->getNextFraction()))) {

                neuralNetwork.loadSequences(*frac);
                neuralNetwork.computeForwardPass();
                cross_entropy = neuralNetwork.calculateError() / words_len[c]; // / words.size();
                error = std::pow(2, cross_entropy);
                ofs << error << std::endl;
                average_error += error;
                ++c;
            }
            average_error /= c;
            ofs << "average: " << average_error << std::endl;
            return 0;
        }

        // process all data set fractions
        printf("open test-file and output-result-file\n");
        std::string wsd_input = Configuration::instance().testFiles()[0];
        std::string wsd_output = config.wsdResult();
        double thr = config.wsd_threshold();
        // boost::shared_ptr<data_sets::CorpusFraction> frac = testSet->getNewFrac();
        // cand_average = simple_wsd(neuralNetwork, wsd_input, wsd_output, word_synsets, lexeme_emb, thr, frac);
        int beam_size = 5;
        beam_wsd<Cpu>(neuralNetwork, wsd_input, wsd_output, word_synsets, lexeme_emb, thr, beam_size, testSet, wdic_lex);


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
        // return trainerMain<Gpu>(config);
        return 1;
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

void importDictBinary(std::unordered_map<std::string, int> &m, std::string &fname)
{
    std::ifstream ifs(fname, std::ios::binary);
    cereal::BinaryInputArchive archive(ifs);
    archive(m);
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
    pos.clear();
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

void readLine_(std::string line, std::vector<std::string>& words)
{
    words.clear();
    std::stringstream ss(line);
    std::string word, item;
    while (std::getline(ss, word, ' ')) {
        // std::stringstream lempos(word);
        // std::getline(lempos, item, '@');
        words.push_back(word);
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

std::string getWord(std::string w)
{
    std::string word;
    std::stringstream ss(w);
    std::getline(ss, word, '%');
    return word;
}
void makeTarget(const std::vector<std::string>& words,
                const std::unordered_map<std::string, int>& dic,
                Cpu::int_vector *target)
{
    target->reserve(words.size());
    std::string unk = "<UNK>";
    int unk_int = dic.find(unk)->second;
    for (auto w : words) {
        if (w.find('%') != std::string::npos) {
            w = getWord(w);
        }
        auto it = dic.find(w);
        if (it == dic.end())
            target->push_back(unk_int);
        else
            target->push_back(it->second);
    }
}
void show_(Cpu::int_vector &v) {
    for (int i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
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

    // printf("setting fraction\n");
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
    // printf("make input\n");
    Cpu::int_vector input = Cpu::int_vector();
    makeInput(words, dic, &input);
    show_(input);

    // printf("place vector to memory\n");
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
    // /*
    // Cpu::int_vector targetClasses = _loadTargetClassesFromCache(seq);
    Cpu::int_vector targetClasses = Cpu::int_vector();
    makeTarget(words, dic, &targetClasses);

    for (int timestep = 0; timestep < words.size(); ++timestep) {
        int tgt = 0; // default class (make configurable?)
        if (timestep >= output_lag)
            tgt = targetClasses[timestep - output_lag];
        // frac->m_targetClasses[timestep * m_parallelSequences + i] = tgt;
        frac->setTargetClasses(timestep, tgt);
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
        frac->setPatTypes(timestep , patType);
    }
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
    std::string synsets, tmp;
    std::stringstream lexss(lexeme);
    std::getline(lexss, tmp, '-');  // word
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
        // printf("synset, pos:%s %d\n", s.c_str(), p);
        return std::make_tuple(s, p);
    }
    return std::make_tuple("", p);

}

std::string getsynset(const std::string& lexeme)
{
    std::stringstream ss(lexeme);
    std::string word, syn;
    POS_type_t p;
    std::getline(ss, word, '-');
    std::tie(syn, p) = getLexemeSynset(word, lexeme);
    return syn;
}

std::string wsd(const std::string& word,
                const POS_type_t& pos,
                const std::vector<std::string>& synsets,
                Cpu::real_vector& output,
                const std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>>& lexeme_emb,
                double thr,
                int *count_wsd,
                double *logprob,
                int bestK,
                std::vector<std::shared_ptr<std::pair<double, std::string>>> *v)
{

    // Cpu::real_vector W;
    int d = lexeme_emb.begin()->second->size();

    int num_syn = synsets.size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_ = Eigen::MatrixXd::Zero(num_syn, d);
    int i = 0;
    for (auto lex : synsets) {
        auto embit = lexeme_emb.find(lex);

        if (embit == lexeme_emb.end()) {
            for ( int j = 0; j < d; ++j)
                W_(i * num_syn + j) = 0.0;
            continue;
        }

        for ( int j = 0; j < d; ++j) {
            W_(i * num_syn + j) = (double)(*(embit->second))[j];

        }
        ++i;
    }

#ifdef SHOW
    printf("W_:\n");

#endif//#ifdef SHOW
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h = Eigen::MatrixXd::Zero(d, 1);
    for (int j = 0; j < d; ++j)
        h(j) = (double)output[j];

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> result = Eigen::MatrixXd::Zero(num_syn, 1);


    //check nan
#ifdef SHOW
    assert( !nanCheck(W_, d) );
    if (nanCheck(h, d)) {
        printf("hidden layer has nan, skipping this word\n");
        return word;
    }
#endif

    result = W_ * h;

#ifdef SHOW
    if (nanCheck(result, num_syn)) {
        printf("scores has nan, skipping this word\n");
        return word;
    }
    printf("result:\n");
    std::cout << result << std::endl;

    checkinf(result, num_syn);
#endif//#ifdef SHOW

    logSoftmax(result, num_syn);

#ifdef SHOW
    printf("result:\n");
    std::cout << result << std::endl;
#endif//#ifdef SHOW

    double _max = -DBL_MAX;
    double max2 = -DBL_MAX;
    int maxarg = -1;
    std::string max_syn;

    std::string syn;
    POS_type_t pos_;
    int counting = 0;
    std::vector<std::pair<double, std::string>> scores = std::vector<std::pair<double, std::string>>();
    scores.reserve(num_syn);
    for (int i = 0; i < num_syn; ++i) {
        std::tie(syn, pos_) = getLexemeSynset(word, synsets[i]);
        if (pos != pos_) continue;
        ++counting;
        scores.push_back(std::make_pair(result(i), syn));
    }
    if (counting == 0)
        return word;
    std::sort(scores.begin(), scores.end(), [](const std::pair<double, std::string>& a, const std::pair<double, std::string>& b){return a.first > b.first;});
    *count_wsd = 0;

    max_syn = scores.at(0).second;
    _max = scores.at(0).first;
    if (counting > 1)
        max2 = scores.at(1).first;

#ifdef SHOW
    printf("max, max2 , counting: %lf %1.4lf %d\n", _max, max2, counting);
#endif//#ifdef SHOW

    if (bestK > 1) {
        v->clear();
        int K = (bestK < scores.size()) ? bestK : scores.size();
        std::cout << "cant-size : " << K << std::endl;
        for (int i = 0; i < K; ++i) {
            std::cout << "wsd, cand : " << scores.at(i).second << std::endl;
            auto cand = std::make_shared<std::pair<double, std::string>>(scores.at(i));
            v->push_back(cand);
        }
        return max_syn;
    }
    if (thr < (_max - max2)) {
        *count_wsd = counting;
        *logprob = _max;
        return max_syn;
    }
    else
        return word;

}


template <typename TDevice>
int simple_wsd(NeuralNetwork<Cpu>& neuralNetwork,
               const std::string& wsd_input,
               const std::string& wsd_output,
               std::unordered_map<std::string, std::vector<std::string>> &word_synsets,
               const std::unordered_map<std::string, std::unique_ptr<Cpu::real_vector>> &lexeme_emb,
               double thr,
               boost::shared_ptr<data_sets::Corpus> &testSet,
               const std::unordered_map<std::string, int>& _wordDict2)
{
    std::ifstream ifs(wsd_input);
    std::ofstream ofs(wsd_output);
    std::string line, word, wsd_result;
    POS_type_t _p;
    std::vector<std::string> words;
    std::vector<POS_type_t> pos;
    int outputSize = neuralNetwork.outputLayer().size();
    // std::vector<int> POS;
    int d = (int)(lexeme_emb.begin()->second)->size();
    Cpu::real_vector output_;
    output_.reserve(d);
    typename TDevice::real_vector outputs;
    printf("start wsd\n");
    int cand = 0;
    int cand_sum = 0;
    double cand_average = 0;
    int did_wsd = 0;
    while (std::getline(ifs, line)) {
        boost::shared_ptr<data_sets::CorpusFraction> frac = testSet->getNewFrac();
        readLine(line, words, pos);
        makeFrac(words, _wordDict2, frac, outputSize);
        neuralNetwork.loadSequences(*frac);
        neuralNetwork.computeForwardPass();
        outputs = neuralNetwork.last_layer();
        for (int i = 0; i < words.size(); ++i) {
            thrust::copy(outputs.begin() + i * d, outputs.begin() + (i+1) * d, output_.begin());

            word = words.at(i);
            _p = pos.at(i);
            if (word_synsets.find(word) == word_synsets.end())
                wsd_result = word;
            else if (_p == POS_OTHER)
                wsd_result = word;
            else {
                wsd_result = wsd(word, _p, word_synsets[word], output_, lexeme_emb, thr, &cand);
                if (cand != 0) {
                    cand_sum += cand;
                    did_wsd += 1;
                    cand = 0;
                }
            }
            std::cout << "word: " << word << " || result: " << wsd_result << std::endl;
            ofs << wsd_result << " ";
        }
        ofs << std::endl;

        // printf(" done.\n");
    }
    cand_average = (double)cand_sum / (double)did_wsd;
    std::cout << "the average of the number of candidate included in disambiguated word : " << cand_average << " (";
    std::cout <<  cand_sum << ", " << did_wsd << " )" << std::endl;
}

void outputResult(std::ofstream &ofs, std::vector<std::string> *words)
{
    for (auto word : *words) {
        ofs << word << " ";
    }
    ofs << std::endl;
}


// template <typename TDevice>
void beam_wsd_act(NeuralNetwork<Cpu>& neuralNetwork,
                  std::vector<std::string> *words,
                  std::vector<POS_type_t>& pos,
                  std::unordered_map<std::string, std::vector<std::string>> &word_synsets,
                  const std::unordered_map<std::string, std::unique_ptr<Cpu::real_vector>> &lexeme_emb,
                  double thr,
                  int start_pos,
                  double wsdscore,
                  int beam_size,
                  std::vector<std::shared_ptr<beam::Beam_state>>& beamv,
                  boost::shared_ptr<data_sets::Corpus> &testSet,
                  const std::unordered_map<std::string, int>& _wordDict2)
{
    std::string line, word, wsd_result;
    POS_type_t _p;
    double logp;
    double score;
    int start_pos_;
    int cand;

    int outputSize = neuralNetwork.outputLayer().size();
    boost::shared_ptr<data_sets::CorpusFraction> frac = testSet->getNewFrac();
    makeFrac(*words, _wordDict2, frac, outputSize);

    int d = (int)(lexeme_emb.begin()->second)->size();
    // typename TDevice::real_vector outputs;
    Cpu::real_vector outputs;
    Cpu::real_vector output_;
    output_.reserve(d);

    neuralNetwork.loadSequences(*frac);
    neuralNetwork.computeForwardPass();
    outputs = neuralNetwork.last_layer();
    std::vector<std::shared_ptr<std::pair<double, std::string>>> results;
    results.reserve(beam_size);
    bool ifbreak = false;
    for (int i = start_pos; i < words->size(); ++i) {
        thrust::copy(outputs.begin() + i * d, outputs.begin() + (i+1) * d, output_.begin());

        word = words->at(i);
        _p = pos.at(i);
        if (word_synsets.find(word) == word_synsets.end())
            wsd_result = word;
        else if (_p == POS_OTHER)
            wsd_result = word;
        else {
            wsd(word, _p, word_synsets[word], output_, lexeme_emb, thr, &cand, &logp, beam_size, &results);
            if (results.size() == 0) {
                printf("result is 0\n");
                continue;
            }
            start_pos_ = i + 1;
            ifbreak = true;
            break;
        }
    }
    double logsumprob;
    if (ifbreak){
        logsumprob = -(double)neuralNetwork.calculateError(start_pos_ - 1);
        std::cout << start_pos_ << ": " << std::endl;
        for (auto cand : results) {
            std::cout << "  " <<  cand->second << ": " << cand->first  << std::endl;
            score = logsumprob + cand->first + wsdscore;
            beamv.push_back(std::make_shared<beam::Beam_state>(*words, score, words->size(), wsdscore + cand->first));
            beamv.back()->words()->at(start_pos_ - 1) = cand->second;
            beamv.back()->set_posi(start_pos_);
        }
    }
    else {
        logsumprob = -(double)neuralNetwork.calculateError();
        start_pos_ = words->size();
        score = logsumprob + wsdscore;
        //no changes
        beamv.push_back(std::make_shared<beam::Beam_state>(*words, score, words->size(), wsdscore));
        beamv.back()->set_posi(start_pos_);
    }

}

template <typename T>
void show_v(std::vector<T> v)
{
    for (const T& a : v) {
        std::cout << a << " ";
    }
    std::cout << std::endl;
}

template <typename TDevice>
int beam_wsd(NeuralNetwork<Cpu>& neuralNetwork,
             const std::string& wsd_input,
             const std::string& wsd_output,
             std::unordered_map<std::string, std::vector<std::string>> &word_synsets,
             const std::unordered_map<std::string, std::unique_ptr<Cpu::real_vector>> &lexeme_emb,
             double thr,
             int beam_size,
             boost::shared_ptr<data_sets::Corpus> &testSet,
             const std::unordered_map<std::string, int>& _wordDict2)
{
    std::ifstream ifs(wsd_input);
    std::ofstream ofs(wsd_output);
    std::string line, word, wsd_result;
    POS_type_t _p;
    std::vector<std::string> words;
    std::vector<POS_type_t> pos;
    // std::vector<int> POS;
    printf("start wsd\n");
    int cand = 0;
    int cand_sum = 0;
    double cand_average = 0;
    int did_wsd = 0;

    std::vector<std::shared_ptr<beam::Beam_state>> beamv;
    std::vector<std::shared_ptr<beam::Beam_state>> next_cand;
    beamv.reserve(beam_size);
    next_cand.reserve(beam_size * beam_size);
    while (std::getline(ifs, line)) {
        beamv.clear();
        readLine(line, words, pos);
        // for (int i = 0; i < beam_size; ++i)
        double last_score=1;
        beamv.push_back(std::make_shared<beam::Beam_state>(words, 0.0, words.size(), 0.0));
        int start_pos = 0;
        while(start_pos < words.size()) {
            for (auto cand : beamv) {
                // beam_wsd_act<TDevice>(
                show_v(*(cand.get()->words()));
                if (last_score == cand->score())
                    continue;
                last_score = cand->score();
                beam_wsd_act(
                    neuralNetwork,
                    cand.get()->words().get(),
                    pos,
                    word_synsets,
                    lexeme_emb,
                    thr,
                    start_pos,
                    cand->wsdscore(),
                    beam_size,
                    next_cand,
                    testSet,
                    _wordDict2
                );
            }
            beamv.clear();
            // sort and get next beam
            std::sort(
                next_cand.begin(),
                next_cand.end(),
                [](std::shared_ptr<beam::Beam_state> &l, std::shared_ptr<beam::Beam_state> &r)
                { return *l.get() > *r.get(); }
            );
            int next_size = (next_cand.size() < beam_size) ? next_cand.size() : beam_size;
            std::cout << "next beam : " << next_cand.size() << std::endl;
            beamv.resize(next_size);
            std::copy (next_cand.begin(), next_cand.begin() + next_size, beamv.begin());
            next_cand.clear();
            start_pos = beamv.begin()->get()->position();
            std::cout << "now, at " << start_pos << " size" << beamv.size() << std::endl;
            if (beamv.size() == 0) throw std::runtime_error("there is no state in beam");
        }
        // output wsd result
        if (beamv.size() > 1){
            if (beamv[0]->score() - beamv[1]->score() > thr)
                outputResult(ofs, beamv[0]->words().get());
            else
                outputResult(ofs, &words);
        }
        else
            outputResult(ofs, beamv[0]->words().get());
    }
    // cand_average = (double)cand_sum / (double)did_wsd;
}


/*
    score stores the sum of log-probability of selected sense
*/
