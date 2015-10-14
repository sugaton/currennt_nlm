
#ifndef LOOKUP_LAYER_HPP
#define LOOKUP_LAYER_HPP

#include <memory>

#include "../layers/TrainableLayer.hpp"
#include "Embedding.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a feed forward layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     * @param TActFn  The activation function to use
     *********************************************************************************************/
    template <typename TDevice>
    class LookupLayer : public TrainableLayer<TDevice>
    {
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

        real_vector embeddings(const int w);

        // void clear_tmpvecs();
    private:
        std::vector<std::unique_ptr<Embedding>> m_embeddings;
        std::vector<real_vector> m_device_vectors;
    };

} // namespace layers

#endif
