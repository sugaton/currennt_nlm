#ifndef INT_INPUT_LAYER_HPP
#define INT_INPUT_LAYER_HPP

#include "../layers/Layer.hpp"
#include "../corpus/CorpusFraction.hpp"

namespace layers {


    /******************************************************************************************//**
     * Represents the input layer of the neural language models
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class intInputLayer : public Layer<TDevice>
    {
        typedef typename TDevice::int_vector int_vector;
        typedef typename TDevice::pattype_vector pattype_vector;
    public:
        /**
         * Construct the layer
         *
         * @param layerChild        The layer section of the JSON configuration
         * @param parallelSequences The maximum number of sequences that shall be computed in parallel
         * @param maxSeqLength      The maximum length of a sequence
         */
        // template <typename TDevice>
        intInputLayer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength);

        // virtual int_vector& outputs();
        virtual int_vector& intoutputs();

        virtual ~intInputLayer();

        virtual const std::string& type() const;

        virtual void computeForwardPass();

        virtual void computeBackwardPass();

        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

    protected:
        int_vector& _intoutputs();
    private:
        int_vector m_outputsInt;

        int               m_curMaxSeqLength;
        int               m_curMinSeqLength;
        int               m_curNumSeqs;
        // real_vector       m_outputs;
        // real_vector       m_outputErrors;
        pattype_vector    m_patTypes;
    };

}

#endif
