
#include "intInputLayer.hpp"

namespace layers {

    template <typename TDevice>
    typename intInputLayer<TDevice>::int_vector& Layer<TDevice>::_outputs()
    {
        return m_outputsInt;
    }

    template <typename TDevice>
    intInputLayer<TDevice>::intInputLayer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength)
        : Layer<TDevice>(layerChild, parallelSequences, maxSeqLength)
    {
        m_outputsInt = Cpu::int_vector(m_parallelSequences * m_maxSeqLength * m_size);
    }

    template <typename TDevice>
    intInputLayer<TDevice>::~intInputLayer()
    {
    }

    template <typename TDevice>
    const std::string& intInputLayer<TDevice>::type() const
    {
        static const std::string s("int_input");
        return s;
    }

    template <typename TDevice>
    typename intInputLayer<TDevice>::int_vector& Layer<TDevice>::outputs()
    {
        return m_outputsInt;
    }

    // TODO write intDataSetFraction class
    template <typename TDevice>
    void intInputLayer<TDevice>::loadSequences(const data_sets::intDataSetFraction &fraction)
    {
        if (fraction.inputPatternSize() != this->size()) {
            throw std::runtime_error(std::string("Input layer size of ") + boost::lexical_cast<std::string>(this->size())
            + " != data input pattern size of " + boost::lexical_cast<std::string>(fraction.inputPatternSize()));
        }

        // Layer<TDevice>::loadSequences(fraction);
        m_curMaxSeqLength = fraction.maxSeqLength();
        m_curMinSeqLength = fraction.minSeqLength();
        m_curNumSeqs      = fraction.numSequences();
        m_patTypes        = fraction.patTypes();

        thrust::copy(fraction.inputs().begin(), fraction.inputs().end(), this->_outputs().begin());
    }

    template <typename TDevice>
    void intInputLayer<TDevice>::computeForwardPass(){}

    template <typename TDevice>
    void intInputLayer<TDevice>::computeBackwardPass(){}

}
