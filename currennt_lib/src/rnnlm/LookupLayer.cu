
#include "LookupLayer.hpp"
#include "intInputLayer.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../Configuration.hpp"

#include <stdexcept>
#include <memory>
#include <climits>
#include <set>
#include <map>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/generate.h>
#include <thrust/copy.h>

namespace internal {
namespace {

    template <typename TActFn>
    struct ComputeOutputFn
    {
        int    layerSize;
        real_t bias;

        const real_t *biasWeights;

        __host__ __device__ real_t operator() (real_t a, const int &outputIdx) const
        {
            // calculate indices
            int blockIdx = outputIdx % layerSize;

            // add the bias
            a += bias * biasWeights[blockIdx];

            // apply the activation function
            real_t b = TActFn::fn(a);

            // store the activation
            return b;
        }
    };

    template <typename TActFn>
    struct ComputeDeltaFn
    {
        // since calculating the derivatives is very cheap for our activation functions,
        // we simple calculate the deltas of all timesteps, including dummies

        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&> &t) const
        {
            real_t delta = TActFn::deriv(t.get<1>()) * t.get<0>();
            t.get<0>() = delta;
        }
    };

    struct ComputeBiasWeightUpdateFn
    {
        int    layerSize;
        int    patternsCount;
        real_t bias;

        const real_t *deltas;

        __host__ __device__ real_t operator() (const int &biasWeightIdx) const
        {
            const real_t *offDeltas = deltas + biasWeightIdx;

            real_t wu = 0;
            for (int i = 0; i < patternsCount; ++i) {
                wu += bias * *offDeltas;
                offDeltas += layerSize;
            }

            return wu;
        }
    };

} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice>
    void LookupLayer<TDevice>::_AddEmbedding(Cpu::real_vector &tmp, const int i, const int maximum_gpusize)
    {
        if ( i  > maximum_gpusize){
            std::unique_ptr<helpers::Embedding<TDevice> > newemb(new helpers::Embedding<TDevice>(&tmp, std::string(typeid(Cpu).name()) ));
            m_embeddings.push_back( std::move(newemb) );
        }
        else{
            std::unique_ptr<helpers::Embedding<TDevice> > newemb(new helpers::Embedding<TDevice>(&tmp, std::string(typeid(TDevice).name()) ));
            m_embeddings.push_back( std::move(newemb) );
        }
    }

    template <typename TDevice>
    LookupLayer<TDevice>::LookupLayer(
        const helpers::JsonValue &layerChild,
        const helpers::JsonValue &weightsSection,
        Layer<TDevice> &precedingLayer)
        : Layer<TDevice>           (layerChild, precedingLayer.parallelSequences(), precedingLayer.maxSeqLength())
        , m_precedingLayer         (precedingLayer)
        , m_wsize                  (layerChild->HasMember("w_size") ? static_cast<int>((*layerChild)["w_size"].GetInt()) : 0)
        // , m_inputWeightsPerBlock   (inputWeightsPerBlock)
        // , m_internalWeightsPerBlock(internalWeightsPerBlock)
        , m_bias                   (layerChild->HasMember("bias") ? static_cast<real_t>((*layerChild)["bias"].GetDouble()) : 0)
        , m_learningRate           (layerChild->HasMember("learningRate") ? static_cast<real_t>((*layerChild)["learningRate"].GetDouble()) : -1)
    {

        Cpu::real_vector weights;
        // m_wdict = std::map<std::string, int>();

        int maximum_gpusize = (layerChild->HasMember("max_gpusize"))? static_cast<int>((*layerChild)["max_gpusize"].GetInt()) : INT_MAX;

        // for random initialization
        const Configuration &config = Configuration::instance();
        static boost::mt19937 *gen = NULL;
        if (!gen) {
            gen = new boost::mt19937;
            gen->seed(config.randomSeed());
        }

        m_embeddings.reserve( m_wsize );
        boost::random::normal_distribution<real_t> dist(config.weightsDistributionNormalMean(), config.weightsDistributionNormalSigma());

        if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
            if (!weightsSection->HasMember(this->name().c_str()))
                throw std::runtime_error(std::string("Missing weights section for layer '") + this->name() + "'");
            const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            // if (!weightsChild.IsObject())
            //     throw std::runtime_error(std::string("Weights section for layer '") + this->name() + "' is not an object");

            int arraysize = 0;
            for (rapidjson::Value::ConstValueIterator eit = weightsChild.Begin(); eit != weightsChild.End(); ++eit){
                ++arraysize;
            }
            m_wsize = std::max(arraysize, m_wsize);
            weights.reserve(this->size());

            int c = 0;
            for (rapidjson::Value::ConstValueIterator eit = weightsChild.Begin(); eit != weightsChild.End(); ++eit){
                // const rapidjson::Value embedingsSection = *(eit);
                if (!eit->HasMember("name"))
                    throw std::runtime_error(std::string("Missing embedings section for layer '") + this->name() + "'");
                if (! eit->HasMember("id"))
                    throw std::runtime_error("Missing embeddings id for lookup layer.");
                std::string word = (*eit)["name"].GetString();
                int id = (*eit)["id"].GetInt();
                m_wdict[word] = id;

                const rapidjson::Value &array = (*eit)["array"];
                int arraysize = 0;
                for (rapidjson::Value::ConstValueIterator it = array.Begin(); it != array.End(); ++it){
                    weights.push_back(static_cast<real_t>(it->GetDouble()));
                    arraysize++;
                }
                if(weights.size() != this->size())
                    throw std::runtime_error("the dimension of loaded embedding does not match this layer's embeddings-size.");
                _AddEmbedding(weights, c, maximum_gpusize);
                weights.clear();
                ++c;
            }
            if (c < m_wsize){
                while( c != m_wsize ){
                    for (size_t i = 0; i < this->size(); ++i)
                        weights.push_back( dist(*gen) );
                    _AddEmbedding(weights, c, maximum_gpusize);
                    weights.clear();
                    ++c;
                }
            }
        }
        // create random weights if no weights are given in the network file
        else {

            weights.resize(this->size());

            if (config.weightsDistributionType() == Configuration::DISTRIBUTION_UNIFORM) {
                real_t range = config.weightsDistributionUniformMax() - config.weightsDistributionUniformMin();
                boost::random::uniform_real_distribution<real_t> dist(0, range);
                for (size_t i = 0; i < weights.size(); ++i)
                    weights[i] = dist(*gen) + config.weightsDistributionUniformMin();
            }
            else {
                for (int c = 0; c < m_wsize; ++c){
                    for (size_t i = 0; i < this->size(); ++i)
                        weights[i] = dist(*gen);
                    _AddEmbedding(weights, c, maximum_gpusize);
                }

            }
        }
        m_weightUpdates = real_vector(this->parallelSequences() * this->maxSeqLength() * this->size());
        // making embeddings
        /*
        Cpu::real_vector tmp;
        tmp.resize(this->size());
        m_embeddings.reserve( m_wsize );
        long int i = 0;
        while ( i * this->size() <  weights.size() ){
            thrust::copy(
                weights.begin() + i * this->size(),
                weights.begin() + (i+1) * this->size(),  //? need -1 ?
                tmp.begin());
            // m_wdict should assign lower id to more frequent word
            if ( i  > maximum_gpusize){
                std::unique_ptr<helpers::Embedding<TDevice> > newemb(new helpers::Embedding<TDevice>(&tmp, std::string(typeid(Cpu).name()) ));
                m_embeddings.push_back( std::move(newemb) );
            }
            else{
                std::unique_ptr<helpers::Embedding<TDevice> > newemb(new helpers::Embedding<TDevice>(&tmp, std::string(typeid(TDevice).name()) ));
                m_embeddings.push_back( std::move(newemb) );
            }
            ++i;
        }
        */

        m_device_vectors.reserve( this->parallelSequences() * this->maxSeqLength() );
        for (int i = 0; i < this->parallelSequences() * this->maxSeqLength(); ++i){
            std::unique_ptr<real_vector> p_vec = std::unique_ptr<real_vector>(new real_vector(this->size()));
            m_device_vectors.push_back(std::move(p_vec));
        }

    }

    // if embedings are loaded from json file, we should align
    template <typename TDevice>
    void LookupLayer<TDevice>::setWordDict(std::map<std::string, int> *wdic)
    {
       /*
        // for random initialization
        const Configuration &config = Configuration::instance();
        static boost::mt19937 *gen = NULL;
        if (!gen) {
            gen = new boost::mt19937;
            gen->seed(config.randomSeed());
        }
        boost::random::normal_distribution<real_t> dist(config.weightsDistributionNormalMean(), config.weightsDistributionNormalSigma());
        auto _rand =  [&]{ return dist(*gen); };

        std::vector< std::unique_ptr<helpers::Embedding<TDevice>> > new_embeddings;
        new_embeddings.resize(m_embeddings.size());

        std::set<int> unplaced;
        for (int i = 0; i < m_embeddings.size(); ++i)
            unplaced.emplace(i);

        for (auto it = wdic->begin(); it != wdic->end(); ++it){
            std::string word = it->first;
            int wid = it->second;
            auto wordIt = m_wdict.find(word);
            if (wordIt == wdic->end()){
                // we should use this space for another embedding
                // thus re-initialization randomly
                Cpu::real_vector h_tmp;
                thrust::generate(h_tmp.begin(), h_tmp.end(), _rand);
                real_vector tmp = h_tmp;
                m_embeddings[wid]->replace(&tmp);
                continue;
            }

            int newid = wordIt->second;
            new_embeddings[newid] = std::move(m_embeddings[wid]);
            unplaced.erase(newid);
        }
        //operation for remaining word
        for (int i = 0; i < m_embeddings.size(); ++i){ // unique_ptr
            if( m_embeddings.at(i) == nullptr ) // already moved
                continue;
            int nextid = *(unplaced.begin());
            new_embeddings[nextid] = std::move(m_embeddings.at(i));
            unplaced.erase(nextid);
        }
        // restore embeddings in m_embeddings
        for (int i = 0; i < new_embeddings.size(); ++i){
            m_embeddings[i] = std::move(new_embeddings[i]);
        }
        */  // no need
        m_wdict = *(wdic);
    }
    template <typename TDevice>
    LookupLayer<TDevice>::~LookupLayer()
    {
    }

    // for loadEmbeddings
    void Loadvector(Cpu::real_vector* vec, std::stringstream& ss){
        std::string item;
        int i = 0;
        while(std::getline(ss, item, ' ')) {
            // loading vector
            (*vec)[i++] = (real_t)std::stof(item);
        }
    }
    /**
         * this method loads word and its embeddings from word2vec-style txtfile
         * if a word which is in 'm_wdict' exists in this file,
         * this method replace its embedding with the loaded embeddings.

         input : filename(std::stirng) -- filename of txt-file
    */
    template <typename TDevice>
    void LookupLayer<TDevice>::loadEmbeddings(const std::string& filename)
    {
        std::ifstream ifs(filename);
        std::string line, word;
        int size;
        long long int wordnum;

        // start should be "TheNumberOfWord(wordnum) Dimension(size)"
        // load first line
        std::getline(ifs, line);
        sscanf(line.data(), "%lld %d", &wordnum, &size);

        Cpu::real_vector vec;
        real_vector Dvec;
        vec.resize(size);
        Dvec.resize(size);

        while(std::getline(ifs, line)) {
            // loading word
            std::stringstream ss(line);
            std::getline(ss, word, ' ');
            auto it = m_wdict.find(word);
            if (it == m_wdict.end()) continue;

            // loading word
            Loadvector(&vec, ss);

            //copying to device (this is needed if TDevice==GPU)
            thrust::copy(vec.begin(), vec.end(), Dvec.begin());
            // replace embedding
            m_embeddings[it->second]->replace(&Dvec);
        }
    }


    template <typename TDevice>
    const std::string& LookupLayer<TDevice>::type() const
    {
        static std::string s;

        if (s.empty())
            s = "lookup";

        return s;
    }

    // ********************
    //this is same as which is defined inLookupLayer
    template <typename TDevice>
    Layer<TDevice>& LookupLayer<TDevice>::precedingLayer()
    {
        return m_precedingLayer;
    }

    template <typename TDevice>
    const Layer<TDevice>& LookupLayer<TDevice>::precedingLayer() const
    {
        return m_precedingLayer;
    }

    template <typename TDevice>
    real_t LookupLayer<TDevice>::learningRate() const
    {
        return m_learningRate;
    }

    template <typename TDevice>
    const typename LookupLayer<TDevice>::real_vector& LookupLayer<TDevice>::weightUpdates() const
    {
        return m_weightUpdates;
    }
    // ********************

    template <typename TDevice>
    void LookupLayer<TDevice>::computeForwardPass()
    {
        // collect outputs from preceding layer
        {{
            // i wanna write like ...
            int i = 0;
            intInputLayer<TDevice>* layer =  dynamic_cast<intInputLayer<TDevice>*>(&(this->precedingLayer()));
            if (!layer)  // TODO throw runtime error
              throw std::runtime_error("the input of LookupLayer should be int. (use intInputLayer)");
            for (int w: layer->intoutputs()){
                // need condition ?
                real_vector* emb = this->embeddings(w, i);  // maybe &this->embeddings(w) returns cpu::real_vector while emb is gpu::real_vector
                // printf("next start %d, allsize %d", i * this->size(), this->_outputs().size())
                if(emb->size() != this->size())
                    throw std::runtime_error("the dimension of loaded embedding does not match this layer's embeddings-size.");
                assert(i * this->size() + emb->size() <= this->_outputs.size());
                thrust::copy(emb->begin(), emb->end(), this->_outputs().begin() + i * this->size());
                ++i;
            }
        }}

    }

    template <typename TDevice>
    void LookupLayer<TDevice>::computeBackwardPass()
    {
        thrust::copy(
            this->outputErrors().begin(),
            this->outputErrors().end(),
            this->_weightUpdates().begin()
        );

    }

    template <typename TDevice>
    void LookupLayer<TDevice>::exportWeights(const helpers::JsonValue &weightsObject, const helpers::JsonAllocator &allocator)
    {
        if (!weightsObject->IsObject())
            throw std::runtime_error("The JSON value is not an object");

        // do nothing if we don't have any weights

        rapidjson::Value weightsSection(rapidjson::kArrayType);
        // add each embedding for weightSection
        for (auto it = m_wdict.begin(); it != m_wdict.end(); ++it){
            rapidjson::Value embeddingsSection(rapidjson::kObjectType);
            std::string word = it->first;
            int w = it->second;
            real_vector* vec = this->embeddings(w, 0);
            // create and fill the weight arrays
            rapidjson::Value embeddingWeightsArray(rapidjson::kArrayType);
            int emb_size = this->size();
            embeddingWeightsArray.Reserve(emb_size, allocator);
            for (int i = 0; i < emb_size; ++i)
                embeddingWeightsArray.PushBack((*vec)[i], allocator);
            // fill the weights subsection
            embeddingsSection.AddMember("name", word.c_str(), allocator);
            embeddingsSection.AddMember("id", w, allocator);
            embeddingsSection.AddMember("array", embeddingWeightsArray, allocator);
            weightsSection.PushBack(embeddingsSection, allocator);
        }

        // add the weights section tot he weights object
        weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
    }

    template <typename TDevice>
    void LookupLayer<TDevice>::exportDict(const helpers::JsonDocument &Object, const helpers::JsonAllocator &allocator) const
    {

        rapidjson::Value dictSection(rapidjson::kArrayType);
        // we should storing order
        std::vector<std::pair<std::string, int>> v(m_wdict.size());
        std::copy(m_wdict.begin(), m_wdict.end(), v.begin());
        std::sort(v.begin(), v.end(),
                  [](const std::pair<std::string, int> &l, const std::pair<std::string, int> &r){
                      return l.second < r.second;
                  });
        for (auto it = v.begin(); it != v.end(); ++it){
            std::string word = it->first;
            int wordid = it->second;
            rapidjson::Value wordObject(rapidjson::kObjectType);
            wordObject.AddMember("name", word.c_str(), allocator);
            wordObject.AddMember("id", wordid, allocator);
            dictSection.PushBack(wordObject, allocator);
        }
        Object->AddMember("word_dict", dictSection, allocator);
    }

    template <typename TDevice>
    LookupLayer<TDevice>::real_vector* LookupLayer<TDevice>::embeddings(const int w, const int i) {
        if ( w > (int)(m_embeddings.size()) )
            throw std::runtime_error("Unknown word appeared not as UNK");
        // printf("word: %d\n", m_embeddings.size(), w);
        helpers::Embedding<TDevice>* emb = m_embeddings.at(w).get();
        if ( emb->type() == typeid(TDevice).name() )
            return emb->get_data();
        // else
            // m_device_vectors.push_back(real_vector());
            return emb->get_data( m_device_vectors.at(i).get() );
    };


    template <typename TDevice>
    helpers::Embedding<TDevice>* LookupLayer<TDevice>::get_emb(const int w)
    {
        return m_embeddings.at(w).get();
    }


    template <typename TDevice>
    typename LookupLayer<TDevice>::real_vector& LookupLayer<TDevice>::_weightUpdates()
    {
        return m_weightUpdates;
    }


    // explicit template instantiations
    template class LookupLayer<Cpu>;
    template class LookupLayer<Gpu>;

} // namespace layers
