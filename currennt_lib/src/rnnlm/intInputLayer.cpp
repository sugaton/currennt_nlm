
#include "intInputLayer.hpp"
#include <iostream>
#include <iterator>

namespace layers {

    template <typename TDevice>
    typename intInputLayer<TDevice>::int_vector& intInputLayer<TDevice>::_intoutputs()
    {
        return m_outputsInt;
    }

    template <typename TDevice>
    intInputLayer<TDevice>::intInputLayer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength)
        : Layer<TDevice>(layerChild, parallelSequences, maxSeqLength)
    {
        // int m_size = (layerChild->HasMember("size") ? (*layerChild)["size"].GetInt()     : 0);
        m_outputsInt = Cpu::int_vector(parallelSequences * maxSeqLength);
    }

    template <typename TDevice>
    intInputLayer<TDevice>::~intInputLayer()
    {
    }

    template <typename TDevice>
    const std::string& intInputLayer<TDevice>::type() const
    {
        static const std::string s("intinput");
        return s;
    }

    template <typename TDevice>
    typename intInputLayer<TDevice>::int_vector& intInputLayer<TDevice>::intoutputs()
    {
        return m_outputsInt;
    }

    template <typename TDevice>
    void intInputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        // if (fraction.inputPatternSize() != this->size()) {
        //     throw std::runtime_error(std::string("Input layer size of ") + boost::lexical_cast<std::string>(this->size())
        //     + " != data input pattern size of " + boost::lexical_cast<std::string>(fraction.inputPatternSize()));
        // }

        // Layer<TDevice>::loadSequences(fraction);
        m_curMaxSeqLength = fraction.maxSeqLength();
        m_curMinSeqLength = fraction.minSeqLength();
        m_curNumSeqs      = fraction.numSequences();
        m_patTypes        = fraction.patTypes();


        // printf("fraction_input:\n");
        // thrust::copy(boost::get<Cpu::int_vector>(fraction.inputs()).begin(),
        //              boost::get<Cpu::int_vector>(fraction.inputs()).end(),
        //              std::ostream_iterator<int>(std::cout, " "));

        // thrust::copy(fraction.intinput_const().begin(),
        //              fraction.intinput_const().end(),
        //              std::ostream_iterator<int>(std::cout, " "));
        // printf("fraction_input end:\n");

        // thrust::copy(boost::get<Cpu::int_vector>(fraction.inputs()).begin(),
        //              boost::get<Cpu::int_vector>(fraction.inputs()).end(),
        thrust::copy(fraction.intinput_const().begin(),
                     fraction.intinput_const().end(),
                     this->_intoutputs().begin());
        // printf("copyied outputs\n");
        // thrust::copy(this->intoutputs().begin(), this->intoutputs().end(), std::ostream_iterator<int>(std::cout, " ") );
    }

    template <typename TDevice>
    void intInputLayer<TDevice>::computeForwardPass(){}

    template <typename TDevice>
    void intInputLayer<TDevice>::computeBackwardPass(){}


    template class intInputLayer<Cpu>;
    template class intInputLayer<Gpu>;
}
