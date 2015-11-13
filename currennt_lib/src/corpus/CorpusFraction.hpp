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

#ifndef DATA_SETS_CORPUSFRACTION_HPP
#define DATA_SETS_CORPUSFRACTION_HPP

#include "../Types.hpp"
#include "../data_sets/DataSetFraction.hpp"

#include <vector>
#include <string>


namespace data_sets {

    /******************************************************************************************//**
     * Contains a fraction of the data sequences in a DataSet that is small enough to be
     * transferred completely to the GPU
     *********************************************************************************************/
    class CorpusFraction : public DataSetFraction
    {
        friend class Corpus;

    // private:
    //     Cpu::int_vector m_inputs;
    private:
        /**
         * Creates the instance
         */
        CorpusFraction();

    public:
        /**
         * Destructor
         */
        ~CorpusFraction();


        // const Cpu::int_vector& inputs() const;

        void set_inputPatternSize(int inputPatternSize);
        void set_outputPatternSize(int outputPatternSize);
        void set_maxSeqLength(int maxSeqLength);
        int set_minSeqLength(int minSeqLength);
        void set_seqInfo(DataSetFraction::seq_info_t& seqInfo);

    };

} // namespace data_sets


#endif // DATA_SETS_DATASETFRACTION_HPP
