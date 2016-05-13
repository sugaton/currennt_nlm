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

#ifndef CORPUS_HPP
#define CORPUS_HPP

#include "CorpusFraction.hpp"

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

#include <string>
#include <unordered_map>
#include <vector>
#include <climits>
#include <fstream>
#include <mpi.h>


namespace data_sets {

    // the ******* nvcc hates boost headers :(
    struct thread_data_t;

    /******************************************************************************************//**
     * Contains input and/or output data of the neural network. This class is used to read input
     * data, training data, validation data or test data sets from a file and to write output data
     * to a file.
     *********************************************************************************************/
    class Corpus : boost::noncopyable
    {
    public:
        struct sequence_t {
            int         originalSeqIdx;
            int         length;
            std::string seqTag;

            std::streampos inputsBegin;
            std::streampos targetsBegin;
        };

    private:
        void _nextFracThreadFn();
        void _shuffleSequences();
        void _shuffleFractions();
        void _addNoise(Cpu::real_vector *v);
        Cpu::int_vector _loadInputsFromCache(const sequence_t &seq);
        Cpu::real_vector _loadOutputsFromCache(const sequence_t &seq);
        Cpu::int_vector _loadTargetClassesFromCache(const sequence_t &seq);
        boost::shared_ptr<CorpusFraction> _makeFractionTask(int firstSeqIdx);
        boost::shared_ptr<CorpusFraction> _makeFirstFractionTask();
        void _makeWordDict(const std::vector<std::string> &txtfiles);
        // Cpu::int_vector&& _makeInputFromLine(const std::string& line, int *loadLength);
        // Cpu::int_vector&& _makeTargetFromLine(const std::string& line);
        Cpu::int_vector _makeInputFromLine(const std::string& line, int *loadLength);
        Cpu::int_vector _makeTargetFromLine(const std::string& line);
        int _getWordId(const std::string& word);

        // for mpi
        int _wordToIndex(std::string& line, std::vector<int> *v);
        void _writeTemp(std::string txtfile, std::string outputfile, int size);
        Cpu::int_vector _makeInputFromBuffer(int* buf, int startpos, int size);
        Cpu::int_vector _makeTargetFromBuffer(int* buf, int startpos, int size);

    private:
        bool             m_fractionShuffling;
        bool             m_sequenceShuffling;
        bool             m_isClassificationData;
        real_t           m_noiseDeviation;
        int              m_parallelSequences;
        long long int    m_totalSequences;
        long long int    m_totalTimesteps;
        int              m_minSeqLength;
        int              m_maxSeqLength;
        int              m_inputPatternSize;
        int              m_outputPatternSize;
        int              m_appearing_threshold;
        int              m_max_vocab_size;

        Cpu::real_vector m_outputMeans;
        Cpu::real_vector m_outputStdevs;

        std::fstream m_cacheFile;
        std::string m_cacheFileName;

        std::vector<sequence_t> m_sequences;

        boost::scoped_ptr<thread_data_t> m_threadData; // just because nvcc hates boost headers
        int m_curFirstSeqIdx;
        std::unordered_map<std::string, int> m_wordids;
        bool m_fixed_wordDict;
        long long int m_nextid;

    public:
        /**
         * Creates an empty data set
         */
        Corpus();

        /**
         * Loads the data set from a raw txt file (filename.txt)
         *
         * @param parSeq   Number of parallel sequences
         * @param fraction Fraction of all sequences to load
         * @param fracShuf Apply fraction shuffling
         * @param seqShuf  Apply sequence shuffling
         * @param noiseDev Static noise deviation
         * @param wordids  Pointer of fixed dictionary
         */
        Corpus(const std::vector<std::string> &ncfiles, int parSeq, real_t fraction=1,
            int truncSeqLength=0,
            bool fracShuf=false, bool seqShuf=false, real_t noiseDev=0,
            std::string cachePath = "",
            std::unordered_map<std::string, int>* wordids=NULL, int constructDict = 0,
            int max_vocab_size = -1, int appearing_threshold = 5);

        // mpi ver constructor
        Corpus(const std::string inputfn, const std::string outputfn, const int rank, const int procs,
            int parSeq, real_t fraction=1, int truncSeqLength=0,
            bool fracShuf=false, bool seqShuf=false, real_t noiseDev=0,
            std::string cachePath = "",
            std::unordered_map<std::string, int>* wordids=NULL, int constructDict = 0,
            int max_vocab_size = -1, int appearing_threshold = 5);
        /**
         * Destructor
         */
        virtual ~Corpus();

        /**
         * Check if the data set contains classification data
         *
         * @return True if the data set contains classification data
         */
        bool isClassificationData() const;

        /**
         * Check if the data set is empty
         *
         * @return True if the data set is empty
         */
        bool empty() const;

        /**
         * Returns the next fraction or an empty pointer once after all fractions
         * have been processed.
         *
         * If the data set is split in 3 fractions, the consecutive calls to this
         * function will lead to the following returned values:
         *   1st call: pointer to 1st fraction
         *   2nd call: pointer to 2nd fraction
         *   3rd call: pointer to 3rd fraction
         *   4th call: empty pointer
         *   5th call: pointer to 1st fraction
         *   ...
         *
         * @return Next fraction or an empty pointer
         */
        boost::shared_ptr<CorpusFraction> getNextFraction();

        /**
         * Returns the local file name used to cache the data
         *
         * @return the local file name used to cache the data
         */
        std::string cacheFileName() const;

        /**
         * Returns the total number of sequences
         *
         * @return The total number of sequences
         */
        int totalSequences() const;

        /**
         * Returns the total number of timesteps
         *
         * @return The total number of timesteps
         */
        int totalTimesteps() const;

        /**
         * Returns the length of the shortest sequence
         *
         * @return The length of the shortest sequence
         */
        int minSeqLength() const;

        /**
         * Returns the length of the longest sequence
         *
         * @return The length of the longest sequence
         */
        int maxSeqLength() const;

        /**
         * Returns the size of the input patterns
         *
         * @return The size of the input patterns
         */
        int inputPatternSize() const;

        /**
         * Returns the size of the output patterns
         *
         * @return The size of the output patterns
         */
        int outputPatternSize() const;

        /**
         * Returns the output means (per feature) indicated in the NC file
         *
         * @return vector of output means
         */
        Cpu::real_vector outputMeans() const;

        /**
         * Returns the output standard deviations (per feature) indicated in the NC file
         *
         * @return vector of output standard deviations
         */
        Cpu::real_vector outputStdevs() const;

        /**
         * Returns the pointer of m_wordids (word-id dictionary)
         * this function will be used for make validateSet and testSet.
         * @return pointer of unordered_map (m_wordids)
         */
        std::unordered_map<std::string, int>* dict();

    };

} // namespace data_sets


#endif // DATA_SETS_DATASET_HPP
