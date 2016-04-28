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

#ifndef DATA_SETS_DATASETFRACTION_HPP
#define DATA_SETS_DATASETFRACTION_HPP

#include "../Types.hpp"
// #include "../corpus/CorpusFraction.hpp"

#include <vector>
#include <string>


namespace data_sets {

    /******************************************************************************************//**
     * Contains a fraction of the data sequences in a DataSet that is small enough to be
     * transferred completely to the GPU
     *********************************************************************************************/
    class DataSetFraction
    {
        friend class DataSet;
        // friend class CorpusFraction;

    public:
        struct seq_info_t {
            int         originalSeqIdx;
            int         length;
            std::string seqTag;
        };

    private:
        int m_inputPatternSize;
        int m_outputPatternSize;
        int m_maxSeqLength;
        int m_minSeqLength;

        std::vector<seq_info_t> m_seqInfo;

        Cpu::real_vector    m_inputs;
        Cpu::int_vector    m_intinputs;
        Cpu::real_vector    m_outputs;
        Cpu::pattype_vector m_patTypes;
        Cpu::int_vector     m_targetClasses;
        int m_inputType;  //0: real, 1: int. default: 0

    // private:
    protected:
        /**
         * Creates the instance
         */
        DataSetFraction();

    public:
        /**
         * Destructor
         */
        ~DataSetFraction();

        /**
         * Returns the size of each input pattern
         *
         * @return The size of each input pattern
         */
        int inputPatternSize() const;

        /**
         * Returns the size of each output pattern
         *
         * @return The size of each output pattern
         */
        int outputPatternSize() const;

        /**
         * Returns the length of the longest sequence
         *
         * @return The length of the longest sequence
         */
        int maxSeqLength() const;

        /**
         * Returns the length of the shortest sequence
         *
         * @return The length of the shortest sequence
         */
        int minSeqLength() const;

        /**
         * Returns the number of sequences in the fraction
         *
         * @return The number of sequences in the fraction
         */
        int numSequences() const;

        /**
         * Returns information about a sequence
         *
         * @param seqIdx The index of the sequence
         */
        const seq_info_t& seqInfo(int seqIdx) const;

        /**
         * Returns the pattern types vector
         *
         * @return The pattern types vector
         */
        const Cpu::pattype_vector& patTypes() const;

        /**
         * Returns the input patterns vector
         *
         * @return The input patterns vector
         */
        const input_vector_type& inputs() const;

        /**
         * Returns the output patterns vector
         *
         * @return The output patterns vector
         */
        const Cpu::real_vector& outputs() const;

        /**
         * Returns the target classes vector
         *
         * @return The target classes vector
         */
        const Cpu::int_vector& targetClasses() const;

        /**
         * Set member values
         */
        Cpu::int_vector* intinput();

        const Cpu::int_vector& intinput_const() const;

        void use_intInput();

        void set_inputPatternSize(int inputPatternSize);

        void set_outputPatternSize(int outputPatternSize);

        void set_maxSeqLength(int maxSeqLength);

        int set_minSeqLength(int minSeqLength);

        void set_seqInfo(DataSetFraction::seq_info_t& seqInfo);

        void setTargetClasses(int posi, int value);

        void setPatTypes(int posi, char value);

        void vectorResize(int parallelSequences, char pat, int num);

    };

} // namespace data_sets


#endif // DATA_SETS_DATASETFRACTION_HPP
