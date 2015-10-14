
#include "LookupLayer.hpp"


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
    LookupLayer<TDevice>::LookupLayer(
        const helpers::JsonValue &layerChild,
        const helpers::JsonValue &weightsSection,
        Layer<TDevice> &precedingLayer)
        : Layer<TDevice>           (layerChild, precedingLayer.parallelSequences(), precedingLayer.maxSeqLength())
        , m_precedingLayer         (precedingLayer)
        , m_inputWeightsPerBlock   (inputWeightsPerBlock)
        , m_internalWeightsPerBlock(internalWeightsPerBlock)
        , m_bias                   (layerChild->HasMember("bias") ? static_cast<real_t>((*layerChild)["bias"].GetDouble()) : 0)
        , m_learningRate           (layerChild->HasMember("learningRate") ? static_cast<real_t>((*layerChild)["learningRate"].GetDouble()) : -1)
    {
        // : TrainableLayer<TDevice>(layerChild, weightsSection, 1, 0, precedingLayer)
        Cpu::real_vector weights;

        if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
            if (!weightsSection->HasMember(this->name().c_str()))
                throw std::runtime_error(std::string("Missing weights section for layer '") + this->name() + "'");
            const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + this->name() + "' is not an object");

            if (!weightsChild.HasMember("input") || !weightsChild["input"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + this->name() + "/input'");

            const rapidjson::Value &inputWeightsChild    = weightsChild["input"];

            if (inputWeightsChild.Size() != this->size() * inputWeightsPerBlock * m_precedingLayer.size())
                throw std::runtime_error(std::string("Invalid number of input weights for layer '") + this->name() + "'");

            weights.reserve(inputWeightsChild.Size());

            for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); it != inputWeightsChild.End(); ++it)
                weights.push_back(static_cast<real_t>(it->GetDouble()));
        }
        // create random weights if no weights are given in the network file
        else {
            // ???? size?
            weights.resize(this->size() *  this->curMaxSeqLength() * this->parallelSequences());

            const Configuration &config = Configuration::instance();

            static boost::mt19937 *gen = NULL;
            if (!gen) {
                gen = new boost::mt19937;
                gen->seed(config.randomSeed());
            }

            if (config.weightsDistributionType() == Configuration::DISTRIBUTION_UNIFORM) {
                real_t range = config.weightsDistributionUniformMax() - config.weightsDistributionUniformMin();
                boost::random::uniform_real_distribution<real_t> dist(0, range);
                for (size_t i = 0; i < weights.size(); ++i)
                    weights[i] = dist(*gen) + config.weightsDistributionUniformMin();
            }
            else {
                boost::random::normal_distribution<real_t> dist(config.weightsDistributionNormalMean(), config.weightsDistributionNormalSigma());
                for (size_t i = 0; i < weights.size(); ++i)
                    weights[i] = dist(*gen);
            }
        }
        // making embeddings
        Cpu::real_vector tmp;
        tmp.resize(this->size());
        int i = 0;
        while ( i * this->size() >  weights.size() ){
            thrust::copy(
                weights.begin() + i * this->size(),
                weights.begin() + (i+1) * this->size(),  //? need -1 ?
                tmp.begin());
            m_embeddings.push_back( std::make_unique<Embedding>(tmp, typeid(TDevice).name) );
        }
    }

    template <typename TDevice>
    LookupLayer<TDevice>::~LookupLayer()
    {
    }

    template <typename TDevice>
    const std::string& LookupLayer<TDevice>::type() const
    {
        static std::string s;

        if (s.empty())
            s = "lookup";

        return s;
    }

    template <typename TDevice>
    void LookupLayer<TDevice>::computeForwardPass()
    {
        // collect outputs from preceding layer
        {{
            // i wanna write like ...
            real_vector& emb;
            int i = 0;
            for (int w: this->precedingLayer().outputs()){
                // need condition ?
                emb = this->embeddings(w, i);  // maybe &this->embeddings(w) returns cpu::real_vector while emb is gpu::real_vector
                thrust::copy(emb.begin(), emb.end(), this->_outputs().begin() + i * this->size());
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
    real_vector& LookupLayer<TDevice>::embeddings(const int w, const int i) {
        if ( w > (int)(m_embeddings.size()) )
            throw std::runtime_error("Unknown word appeared not as UNK");
        Embedding& emb = m_embeddings.at(w);
        if ( emb.type() == typeid(TDevice).name() )
            return m_embeddings.at(w)->get_data<TDevice>();
        // else
            // m_device_vectors.push_back(real_vector());
            return m_embeddings.at(w)->get_data<TDevice>( &(m_device_vectors.at(i)) );
    };


    template <typename TDevice>
    Embedding& LookupLayer<TDevice>::get_emb(const int w)
    {
        return m_embeddings.at(w);
    }

    // explicit template instantiations
    template class LookupLayer<Cpu>;
    template class LookupLayer<Gpu>;

} // namespace layers
