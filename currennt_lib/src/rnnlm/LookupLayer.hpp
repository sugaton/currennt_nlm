
#ifndef LOOKUP_LAYER_HPP
#define LOOKUP_LAYER_HPP

#include <memory>
#include <map>

#include "../layers/Layer.hpp"
#include "Embedding.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a feed forward layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class LookupLayer : public Layer<TDevice>
    {
    typedef typename TDevice::real_vector real_vector;
    private:
        Layer<TDevice> &m_precedingLayer;
        int m_wsize;
        int m_maximum_gpusize;
        int m_UNKid;
        bool m_fixed;
        bool m_allowCpuEmb;

        // need?
        // const int    m_inputWeightsPerBlock;
        // const int    m_internalWeightsPerBlock;
        const real_t m_bias;
        const real_t m_learningRate;
        std::map<std::string, int> m_wdict;
        // real_vector m_outputErrors;
        // real_vector m_weights;
        real_vector m_weightUpdates;

        void _AddEmbedding(Cpu::real_vector &tmp, const int i, const int maximum_gpusize);
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         */
        LookupLayer(
            const helpers::JsonValue &layerChild,
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer
            );

        /**
         * Destructs the Layer
         */
        virtual ~LookupLayer();


        void setWordDict(std::map<std::string, int> *wdic);
        // from TrainableLayer
        Layer<TDevice>& precedingLayer();
        const Layer<TDevice>& precedingLayer() const;
        real_t learningRate() const;
        const real_vector& weightUpdates() const;
        real_vector& _weightUpdates();
        size_t lookupSize() const;

        /**
         * this method loads word and its embeddings from word2vec-style txtfile
         * if a word which is in 'm_wdict' exists in this file,
         * this method replace its embedding with the loaded embeddings.

         input : filename(std::stirng) -- filename of txt-file
        */
        void loadEmbeddings(const std::string& filename);

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass();

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass();

        /**
         * export each word's embedding
         */
        void exportWeights(const helpers::JsonValue &weightsObject, const helpers::JsonAllocator &allocator);

        /**
         * export word list which registered in this table
         */
        void exportDict(const helpers::JsonDocument &Object, const helpers::JsonAllocator &allocator) const;

        real_vector* embeddings(const int w, const int i);

        helpers::Embedding<TDevice>* get_emb(const int w);

        void fixEmb();
        bool fixed() const;

        // void clear_tmpvecs();
    private:
        std::vector< std::unique_ptr<helpers::Embedding<TDevice>> > m_embeddings;
        std::vector<std::unique_ptr<real_vector>> m_device_vectors;
    };

} // namespace layers

#endif
