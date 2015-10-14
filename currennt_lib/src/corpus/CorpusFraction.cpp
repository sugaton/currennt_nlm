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

#include "CorpusFraction.hpp"


namespace data_sets {

    CorpusFraction::CorpusFraction()
    {
    }

    CorpusFraction::~CorpusFraction()
    {
    }
    /*
    int CorpusFraction::inputPatternSize() const
    {
        return m_inputPatternSize;
    }

    int CorpusFraction::outputPatternSize() const
    {
        return m_outputPatternSize;
    }

    int CorpusFraction::maxSeqLength() const
    {
        return m_maxSeqLength;
    }

    int CorpusFraction::minSeqLength() const
    {
        return m_minSeqLength;
    }

    int CorpusFraction::numSequences() const
    {
        return (int)m_seqInfo.size();
    }

    const CorpusFraction::seq_info_t& CorpusFraction::seqInfo(int seqIdx) const
    {
        return m_seqInfo[seqIdx];
    }

    const Cpu::pattype_vector& CorpusFraction::patTypes() const
    {
        return m_patTypes;
    }
    */
    const Cpu::int_vector& CorpusFraction::inputs() const
    {
        return m_inputs;
    }

    /*
    const Cpu::real_vector& CorpusFraction::outputs() const
    {
        return m_outputs;
    }

    const Cpu::int_vector& CorpusFraction::targetClasses() const
    {
        return m_targetClasses;
    }
    */
} // namespace data_sets
