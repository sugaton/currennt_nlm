#ifndef INT_INPUT_LAYER_HPP
#define INT_INPUT_LAYER_HPP
#endif

#include "../layers/Layer.hpp"
#include "intDataSetFraction.hpp"

namespace layers {


    /******************************************************************************************//**
     * Represents the input layer of the neural language models
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class intInputLayer : public Layer<TDevice>
    {
    public:
        /**
         * Construct the layer
         *
         * @param layerChild        The layer section of the JSON configuration
         * @param parallelSequences The maximum number of sequences that shall be computed in parallel
         * @param maxSeqLength      The maximum length of a sequence
         */
        template <typename TDevice>
        intInputLayer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength);

        virtual ~intInputLayer();

        virtual const std::string& type() const;

        virtual void computeForwardPass();

        virtual void computeBackwardPass();

        virtual void loadSequences(const data_sets::intDataSetFraction &fraction)

    private:
        int_vector m_outputsInt;
    }

}
