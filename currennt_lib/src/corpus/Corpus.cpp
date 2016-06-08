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

 /***
arrange of DataSet.cpp, for NLP corpus reading
 ***/

#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <boost/function.hpp>

#include "Corpus.hpp"
#include "../Configuration.hpp"

// #include "../netcdf/netcdf.h"

#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cassert>

#ifdef _MYMPI
//#include "../../../mpi/mpiumap.hpp"
#endif

namespace {
namespace internal {
    /*
    int readNcDimension(int ncid, const char *dimName)
    {
        int ret;
        int dimid;
        size_t x;

        if ((ret = nc_inq_dimid(ncid, dimName, &dimid)) || (ret = nc_inq_dimlen(ncid, dimid, &x)))
            throw std::runtime_error(std::string("Cannot get dimension '") + dimName + "': " + nc_strerror(ret));

        return (int)x;
    }

    bool hasNcDimension(int ncid, const char *dimName)
    {
        try {
            readNcDimension(ncid, dimName);
            return true;
        }
        catch (...) {
            return false;
        }
    }

    std::string readNcStringArray(int ncid, const char *arrName, int arrIdx, int maxStringLength)
    {
        int ret;
        int varid;
        char *buffer = new char[maxStringLength+1];
        size_t start[] = {arrIdx, 0};
        size_t count[] = {1, maxStringLength};

        if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_text(ncid, varid, start, count, buffer)))
            throw std::runtime_error(std::string("Cannot read variable '") + arrName + "': " + nc_strerror(ret));

        buffer[maxStringLength] = '\0';
        return std::string(buffer);
    }

    int readNcIntArray(int ncid, const char *arrName, int arrIdx)
    {
        int ret;
        int varid;
        size_t start[] = {arrIdx};
        size_t count[] = {1};

        int x;
        if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_int(ncid, varid, start, count, &x)))
            throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

        return x;
    }

    template <typename T>
    int _readNcArrayHelper(int ncid, int varid, const size_t start[], const size_t count[], T *v);

    template <>
    int _readNcArrayHelper<float>(int ncid, int varid, const size_t start[], const size_t count[], float *v)
    {
        return nc_get_vara_float(ncid, varid, start, count, v);
    }

    template <>
    int _readNcArrayHelper<double>(int ncid, int varid, const size_t start[], const size_t count[], double *v)
    {
        return nc_get_vara_double(ncid, varid, start, count, v);
    }

    template <>
    int _readNcArrayHelper<int>(int ncid, int varid, const size_t start[], const size_t count[], int *v)
    {
        return nc_get_vara_int(ncid, varid, start, count, v);
    }

    template <typename T>
    thrust::host_vector<T> readNcArray(int ncid, const char *arrName, int begin, int n)
    {
        int ret;
        int varid;
        size_t start[] = {begin};
        size_t count[] = {n};

        thrust::host_vector<T> v(n);
        if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = _readNcArrayHelper<T>(ncid, varid, start, count, v.data())))
            throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

        return v;
    }

    Cpu::real_vector readNcPatternArray(int ncid, const char *arrName, int begin, int n, int patternSize)
    {
        int ret;
        int varid;
        size_t start[] = {begin, 0};
        size_t count[] = {n, patternSize};

        Cpu::real_vector v(n * patternSize);
        if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = _readNcArrayHelper<real_t>(ncid, varid, start, count, v.data())))
            throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

        return v;
    }
    */
    Cpu::real_vector targetClassesToOutputs(const Cpu::int_vector &targetClasses, int numLabels)
    {
        if (numLabels == 2) {
            Cpu::real_vector v(targetClasses.size());
            for (size_t i = 0; i < v.size(); ++i)
                v[i] = (real_t)targetClasses[i];

            return v;
        }
        else {
            Cpu::real_vector v(targetClasses.size() * numLabels, 0);

            for (size_t i = 0; i < targetClasses.size(); ++i)
                v[i * numLabels + targetClasses[i]] = 1;

            return v;
        }
    }

    bool comp_seqs(const data_sets::Corpus::sequence_t &a, const data_sets::Corpus::sequence_t &b)
    {
        return (a.length < b.length);
    }

    struct rand_gen {
        unsigned operator()(unsigned i)
        {
            static boost::mt19937 *gen = NULL;
            if (!gen) {
                gen = new boost::mt19937;
                gen->seed(Configuration::instance().randomSeed());
            }

            boost::uniform_int<> dist(0, i-1);
            return dist(*gen);
        }
    };

} // namespace internal
} // anonymous namespace


namespace data_sets {

    struct thread_data_t
    {
        boost::thread             thread;
        boost::mutex              mutex;
        boost::condition_variable cv;
        bool                      terminate;

        boost::function<boost::shared_ptr<CorpusFraction> ()> taskFn;
        boost::shared_ptr<CorpusFraction> frac;
        bool finished;
    };

    void Corpus::_nextFracThreadFn()
    {
        for (;;) {
            // wait for a new task
            boost::unique_lock<boost::mutex> lock(m_threadData->mutex);
            while (m_threadData->taskFn.empty() && !m_threadData->terminate)
                m_threadData->cv.wait(lock);

            // terminate the thread?
            if (m_threadData->terminate)
                break;

            // execute the task
            m_threadData->frac.reset();
            m_threadData->frac = m_threadData->taskFn();
            m_threadData->finished = true;
            m_threadData->taskFn.clear();

            // tell the others that we are ready
            m_threadData->cv.notify_one();
        }
    }

    void Corpus::_shuffleSequences()
    {
        internal::rand_gen rg;
        std::random_shuffle(m_sequences.begin(), m_sequences.end(), rg);
    }

    void Corpus::_shuffleFractions()
    {
        std::vector<std::vector<sequence_t> > fractions;
        for (size_t i = 0; i < m_sequences.size(); ++i) {
            if (i % m_parallelSequences == 0)
                fractions.resize(fractions.size() + 1);
            fractions.back().push_back(m_sequences[i]);
        }

        internal::rand_gen rg;
        std::random_shuffle(fractions.begin(), fractions.end(), rg);

        m_sequences.clear();
        for (size_t i = 0; i < fractions.size(); ++i) {
            for (size_t j = 0; j < fractions[i].size(); ++j)
                m_sequences.push_back(fractions[i][j]);
        }
    }

    void Corpus::_addNoise(Cpu::real_vector *v)
    {
        if (!m_noiseDeviation)
            return;

        static boost::mt19937 *gen = NULL;
        if (!gen) {
            gen = new boost::mt19937;
            gen->seed(Configuration::instance().randomSeed());
        }

        boost::normal_distribution<real_t> dist((real_t)0, m_noiseDeviation);

        for (size_t i = 0; i < v->size(); ++i)
            (*v)[i] += dist(*gen);
    }

    Cpu::int_vector Corpus::_loadInputsFromCache(const sequence_t &seq)
    {
        Cpu::int_vector v(seq.length * m_inputPatternSize);

        m_cacheFile.seekg(seq.inputsBegin);
        m_cacheFile.read((char*)v.data(), sizeof(int) * v.size());
        assert (m_cacheFile.tellg() - seq.inputsBegin == v.size() * sizeof(int));

        return v;
    }

    Cpu::real_vector Corpus::_loadOutputsFromCache(const sequence_t &seq)
    {
        Cpu::real_vector v(seq.length * m_outputPatternSize);

        m_cacheFile.seekg(seq.targetsBegin);
        m_cacheFile.read((char*)v.data(), sizeof(real_t) * v.size());
        assert (m_cacheFile.tellg() - seq.targetsBegin == v.size() * sizeof(real_t));

        return v;
    }

    Cpu::int_vector Corpus::_loadTargetClassesFromCache(const sequence_t &seq)
    {
        Cpu::int_vector v(seq.length);

        m_cacheFile.seekg(seq.targetsBegin);
        m_cacheFile.read((char*)v.data(), sizeof(int) * v.size());
        assert (m_cacheFile.tellg() - seq.targetsBegin == v.size() * sizeof(int));

        return v;
    }

    boost::shared_ptr<CorpusFraction> Corpus::_makeFractionTask(int firstSeqIdx)
    {
        int context_left = Configuration::instance().inputLeftContext();
        int context_right = Configuration::instance().inputRightContext();
        int context_length = context_left + context_right + 1;
        int output_lag = Configuration::instance().outputTimeLag();

        //printf("(%d) Making task firstSeqIdx=%d...\n", (int)m_sequences.size(), firstSeqIdx);
        boost::shared_ptr<CorpusFraction> frac(new CorpusFraction);
        // frac->m_inputPatternSize  = m_inputPatternSize * context_length;
        frac->use_intInput();
        frac->set_inputPatternSize(m_inputPatternSize * context_length);
        // frac->m_outputPatternSize = m_outputPatternSize;
        frac->set_outputPatternSize(m_outputPatternSize);
        // frac->m_maxSeqLength      = std::numeric_limits<int>::min();
        frac->set_maxSeqLength(std::numeric_limits<int>::min());
        // frac->m_minSeqLength      = std::numeric_limits<int>::max();
        frac->set_minSeqLength(std::numeric_limits<int>::max());

        // fill fraction sequence info
        for (int seqIdx = firstSeqIdx; seqIdx < firstSeqIdx + m_parallelSequences; ++seqIdx) {
            if (seqIdx < (int)m_sequences.size()) {
                // frac->m_maxSeqLength = std::max(frac->m_maxSeqLength, m_sequences[seqIdx].length);
                frac->set_maxSeqLength(std::max(frac->maxSeqLength(), m_sequences[seqIdx].length));
                // frac->m_minSeqLength = std::min(frac->m_minSeqLength, m_sequences[seqIdx].length);
                frac->set_minSeqLength(std::min(frac->maxSeqLength(), m_sequences[seqIdx].length));

                CorpusFraction::seq_info_t seqInfo;
                seqInfo.originalSeqIdx = m_sequences[seqIdx].originalSeqIdx;
                seqInfo.length         = m_sequences[seqIdx].length;
                seqInfo.seqTag         = m_sequences[seqIdx].seqTag;

                // frac->m_seqInfo.push_back(seqInfo);
                frac->set_seqInfo(seqInfo);
            }
        }

        // allocate memory for the fraction
        // only m_inputs can be accessed directly
        frac->intinput()->resize(frac->maxSeqLength() * m_parallelSequences * frac->inputPatternSize(), 0);
        // resize of targetclass and patTypes
        frac->vectorResize(m_parallelSequences, PATTYPE_NONE, -1);

        // load sequences from the cache file and create the fraction vectors
        for (int i = 0; i < m_parallelSequences; ++i) {
            if (firstSeqIdx + i >= (int)m_sequences.size())
                continue;

            const sequence_t &seq = m_sequences[firstSeqIdx + i];

            // inputs
            Cpu::int_vector inputs = _loadInputsFromCache(seq);
            // _addNoise(&inputs);
            for (int timestep = 0; timestep < seq.length; ++timestep) {
                int srcStart = m_inputPatternSize * timestep;
                int offset_out = 0;
                for (int offset_in = -context_left; offset_in <= context_right; ++offset_in) {
                    int srcStart = m_inputPatternSize * (timestep + offset_in);
                    // duplicate first time step if needed
                    if (srcStart < 0)
                        srcStart = 0;
                    // duplicate last time step if needed
                    else if (srcStart > m_inputPatternSize * (seq.length - 1))
                        srcStart = m_inputPatternSize * (seq.length - 1);
                    int tgtStart = frac->inputPatternSize() * (timestep * m_parallelSequences + i) + offset_out * m_inputPatternSize;
                    //std::cout << "copy from " << srcStart << " to " << tgtStart << " size " << m_inputPatternSize << std::endl;
                    thrust::copy_n(inputs.begin() + srcStart, m_inputPatternSize, frac->intinput()->begin() + tgtStart);
                    ++offset_out;
                }
            }
            /*std::cout << "original inputs: ";
            thrust::copy(inputs.begin(), inputs.end(), std::ostream_iterator<real_t>(std::cout, ";"));
            std::cout << std::endl;*/

            //  target classes
            //  only classification
            Cpu::int_vector targetClasses = _loadTargetClassesFromCache(seq);
            for (int timestep = 0; timestep < seq.length; ++timestep) {
                int tgt = 0; // default class (make configurable?)
                if (timestep >= output_lag)
                    tgt = targetClasses[timestep - output_lag];
                // frac->m_targetClasses[timestep * m_parallelSequences + i] = tgt;
                frac->setTargetClasses(timestep * m_parallelSequences + i, tgt);
            }
            // }
            // // outputs
            // else {
            //     Cpu::real_vector outputs = _loadOutputsFromCache(seq);
            //     for (int timestep = 0; timestep < seq.length; ++timestep) {
            //         int tgtStart = m_outputPatternSize * (timestep * m_parallelSequences + i);
            //         if (timestep >= output_lag) {
            //             int srcStart = m_outputPatternSize * (timestep - output_lag);
            //             thrust::copy_n(outputs.begin() + srcStart, m_outputPatternSize, frac->m_outputs.begin() + tgtStart);
            //         }
            //         else {
            //             for (int oi = 0; oi < m_outputPatternSize; ++oi) {
            //                 frac->m_outputs[tgtStart + oi] = 1.0f; // default value (make configurable?)
            //             }
            //         }
            //     }
            // }

            // pattern types
            for (int timestep = 0; timestep < seq.length; ++timestep) {
                Cpu::pattype_vector::value_type patType;
                if (timestep == 0)
                    patType = PATTYPE_FIRST;
                else if (timestep == seq.length - 1)
                    patType = PATTYPE_LAST;
                else
                    patType = PATTYPE_NORMAL;

                // frac->m_patTypes[timestep * m_parallelSequences + i] = patType;
                frac->setPatTypes(timestep * m_parallelSequences + i, patType);
            }
        }
        return frac;
    }

    boost::shared_ptr<CorpusFraction> Corpus::_makeFirstFractionTask()
    {
        //printf("(%d) Making first task...\n", (int)m_sequences.size());

        if (m_sequenceShuffling)
            _shuffleSequences();
        if (m_fractionShuffling)
            _shuffleFractions();

        return _makeFractionTask(0);
    }

    int Corpus::_getWordId(const std::string& word)
    {
        auto it = m_wordids.find(word);
        if ( it == m_wordids.end() ){
            if (m_fixed_wordDict)
                return m_wordids["<UNK>"];
            // else, add word to dict
            m_wordids[word] = m_nextid;
            return m_nextid++;
            // m_nextid++;
        }
        else {
            // ids.push_back(it->second);
            return it->second;
        }
    }

    Cpu::int_vector Corpus::_makeInputFromLine(const std::string& line, int *loadLength)
    {
        std::stringstream ss(line);
        std::string word;
        char delim = ' ';
        std::deque<int> ids;
        ids.push_back(m_wordids["<s>"]);
        *loadLength = 1; // already counting <s>
        while ( std::getline(ss, word, delim) ){
            ids.push_back(_getWordId(word));
            *loadLength += 1;
        }
        Cpu::int_vector vec(*loadLength);
        for ( int i = 0; i < *loadLength; ++i ){
            vec[i] = (ids.at(i));
        }
        // return std::move(vec);
        return vec;
    }

    Cpu::int_vector Corpus::_makeTargetFromLine(const std::string& line)
    {
        std::stringstream ss(line);
        std::string word;
        char delim = ' ';
        // *loadLength = 0;
        int loadlength = 0;
        std::deque<int> ids;
        while ( std::getline(ss, word, delim) ){
            ids.push_back(_getWordId(word));
            ++loadlength;
        }
        Cpu::int_vector vec(loadlength + 1);
        for ( int i = 0; i < loadlength; ++i ){
            vec[i] = ( (ids.at(i) >= m_max_vocab_size )? m_wordids["<UNK>"] : ids.at(i) );
        }
        vec[loadlength] = (m_wordids["</s>"]);
        return vec;
    }

    void Corpus::_makeWordDict(const std::vector<std::string> &txtfiles)
    {
        std::unordered_map<std::string, int> counter;
        for (std::vector<std::string>::const_iterator f_itr = txtfiles.begin();
            f_itr != txtfiles.end(); ++f_itr)
        {
            std::ifstream fin(*f_itr);
            std::string line, word;
            char delim = ' ';
            while (std::getline(fin, line)){
                std::stringstream ss(line);
                while ( std::getline(ss, word, delim) ){
                    if(m_wordids.find(word) == m_wordids.end()){
                        auto it = counter.find(word);
                        if (it == counter.end())
                            counter[word] = 0;
                        else
                            counter[word] += 1;
                    }
                }
            }
        }
        std::vector<std::pair<std::string, int>> v(counter.size());
        std::copy(counter.begin(), counter.end(), v.begin());
        std::sort(v.begin(), v.end(),
                  [](const std::pair<std::string, int>& l, const std::pair<std::string, int> r){
                      return l.second > r.second;
                  });
        for (std::pair<std::string, int> p : v){
            if (p.second < m_appearing_threshold)
                break;
            m_wordids[p.first] = m_nextid++;
        }
    }

    Corpus::Corpus()
        : m_fractionShuffling(false)
        , m_sequenceShuffling(false)
        , m_noiseDeviation   (0)
        , m_parallelSequences(0)
        , m_totalSequences   (0)
        , m_totalTimesteps   (0)
        , m_minSeqLength     (0)
        , m_maxSeqLength     (0)
        , m_inputPatternSize (0)
        , m_outputPatternSize(0)
        , m_curFirstSeqIdx   (-1)
        , m_nextid(0)
    {
    }

    Corpus::Corpus(const std::vector<std::string> &txtfiles, int parSeq, real_t fraction, int truncSeqLength, bool fracShuf, bool seqShuf, real_t noiseDev,
                   std::string cachePath, std::unordered_map<std::string, int>* wordids, int constructDict, int max_vocab_size, int appearing_threshold)
        : m_fractionShuffling(fracShuf)
        , m_sequenceShuffling(seqShuf)
        , m_noiseDeviation   (noiseDev)
        , m_parallelSequences(parSeq)
        , m_totalTimesteps   (0)
        , m_minSeqLength     (std::numeric_limits<int>::max())
        , m_maxSeqLength     (std::numeric_limits<int>::min())
        , m_curFirstSeqIdx   (-1)
        , m_nextid(0)
        , m_inputPatternSize (1)
        , m_max_vocab_size( (max_vocab_size == -1)? INT_MAX : max_vocab_size )
        , m_appearing_threshold(appearing_threshold)
    {
        int ret;
        int ncid;

        m_fixed_wordDict = false;
        if (wordids != NULL){
            m_wordids = *wordids;
           if (!constructDict)
                m_fixed_wordDict = true;
            m_nextid = (long long int)m_wordids.size();
        }

        if (fraction <= 0 || fraction > 1)
            throw std::runtime_error("Invalid fraction");

        // open the cache file
        std::string tmpFileName = "";
        if (cachePath == "") {
            tmpFileName = (boost::filesystem::temp_directory_path() / boost::filesystem::unique_path()).string();
        }
        else {
            tmpFileName = cachePath + "/" + (boost::filesystem::unique_path()).string();
        }
        std::cerr << std::endl << "using cache file: " << tmpFileName << std::endl << "... ";
        m_cacheFileName = tmpFileName;
        m_cacheFile.open(tmpFileName.c_str(), std::fstream::in | std::fstream::out | std::fstream::binary | std::fstream::trunc);
        if (!m_cacheFile.good())
            throw std::runtime_error(std::string("Cannot open temporary file '") + tmpFileName + "'");

        bool first_file = true;

        if (!m_fixed_wordDict){
            if (m_wordids.find("<s>") == m_wordids.end())
                m_wordids["<s>"] = m_nextid++;
            if (m_wordids.find("</s>") == m_wordids.end())
                m_wordids["</s>"] = m_nextid++;
            if (m_wordids.find("<UNK>") == m_wordids.end())
                m_wordids["<UNK>"] = m_nextid++;
            _makeWordDict(txtfiles);
            m_fixed_wordDict = true;
        }

        for (std::vector<std::string>::const_iterator f_itr = txtfiles.begin();
            f_itr != txtfiles.end(); ++f_itr)
        {
            std::vector<sequence_t> sequences;

            std::ifstream fin(*f_itr);

            int seqidxCount = 0;
            int k = 0;
            int loadLength;
            std::string line;
            while ( std::getline(fin, line) ) {
                // reading first
                Cpu::int_vector inputs = _makeInputFromLine(line, &loadLength);
                Cpu::int_vector targets = _makeTargetFromLine(line);
                m_totalTimesteps += loadLength;

                // making sequence
                int loaded = 0;
                while (loadLength > 0){
                    sequence_t seq;
                    seq.originalSeqIdx = k;

                    // read input patterns and store them in the cache file
                    seq.inputsBegin = m_cacheFile.tellp();

                    if (truncSeqLength > 0 && loadLength > 1.5 * truncSeqLength)
                        seq.length = std::min(truncSeqLength, loadLength);
                    else
                        seq.length = loadLength;


                    m_cacheFile.write((const char*)(inputs.data() + loaded), sizeof(int) * seq.length);
                    assert (m_cacheFile.tellp() - seq.inputsBegin == seq.length * sizeof(int));


                    m_minSeqLength = std::min(m_minSeqLength, seq.length);
                    m_maxSeqLength = std::max(m_maxSeqLength, seq.length);

                    // read targets and store them in the cache file
                    seq.targetsBegin = m_cacheFile.tellp();

                    // Cpu::int_vector targets = _makeTargetFromLine(line);
                    m_cacheFile.write((const char*)(targets.data() + loaded), sizeof(int) * seq.length);
                    assert (m_cacheFile.tellp() - seq.targetsBegin == seq.length * sizeof(int));
                    sequences.push_back(seq);
                    ++k;
                    loadLength -= seq.length;
                    loaded += seq.length;
                }
            }

            if (first_file) {
                m_outputMeans  = Cpu::real_vector(m_outputPatternSize, 0.0f);
                m_outputStdevs = Cpu::real_vector(m_outputPatternSize, 1.0f);
                // }
            }

            // create next fraction data and start the thread
            m_threadData.reset(new thread_data_t);
            m_threadData->finished  = false;
            m_threadData->terminate = false;
            m_threadData->thread    = boost::thread(&Corpus::_nextFracThreadFn, this);

            m_sequences.insert(m_sequences.end(), sequences.begin(), sequences.end());

            first_file = false;
        } // txt file loop

        m_totalSequences = m_sequences.size();
        m_outputPatternSize = std::min(m_max_vocab_size, (int)m_wordids.size());
        printf("outputPatternSize: %d(m_max_vocab_size %d, m_wordids.size %d)\n", m_outputPatternSize, m_max_vocab_size, m_wordids.size());
        printf("max_vocab_size %d, m_wordids.size %d)\n", max_vocab_size, INT_MAX);
        // sort sequences by length
        if (Configuration::instance().trainingMode())
            std::sort(m_sequences.begin(), m_sequences.end(), internal::comp_seqs);
    }

    /// 'for mpi ver constructor'
    int Corpus::_wordToIndex(std::string& line, std::vector<int> *v) {
        int c = 0;
        char delim = ' ';
        std::string w;
        std::stringstream ss(line);
        while (std::getline(ss, w, delim)) {
            v->push_back(_getWordId(w));
            ++c;
        }
        return c;
    }
    // make binary file that will be read by all processes
    void Corpus::_writeTemp(std::string txtfile, std::string outputfile, int size) {
        std::ifstream ifs(txtfile);
        std::ofstream ofs(outputfile, std::ios::out|std::ios::binary);

        std::string line;
        std::vector<int> v;
        v.reserve(size*3);
        int count, padding;

        while (std::getline(ifs, line)) {
            count = _wordToIndex(line, &v);
            padding = size - count % size;
            for (int i = 0; i < padding; ++i) {
                v.push_back(-1);
            }
            // displacements.push_back(offset);
            ofs.write((char*)v.data(), v.size() * sizeof(int));
            //showVector(v.data(), v.size());
            v.clear();
        }
        ofs.close();
    }
    Cpu::int_vector Corpus::_makeInputFromBuffer(int* buf, int startpos, int size)
    {
        Cpu::int_vector vec(size+1);
        vec[0] = m_wordids["<s>"];
        for ( int i = 0; i < size; ++i ){
            vec[i+1] = buf[startpos + i];
        }
        return vec;
    }

    Cpu::int_vector Corpus::_makeTargetFromBuffer(int * buf, int startpos, int size)
    {
        Cpu::int_vector vec(size+1);
        int item;
        for ( int i = 0; i < size; ++i ){
            item = buf[startpos + i];
            vec[i] = ( item >= m_max_vocab_size )? m_wordids["<UNK>"] : item ;
        }
        vec[size] = (m_wordids["</s>"]);
        return vec;
    }
    /// end of 'for mpi ver constructor'

#ifdef _MYMPI

    void str_bcast(std::string& s) 
    {
        int size = (int)s.size();
        MPI::COMM_WORLD.Bcast(&size, 1, MPI::INT, 0);
        char* arr = new char[size];
        strcpy(arr, s.c_str());
        MPI::COMM_WORLD.Bcast((void*)arr, (int)size, MPI::CHAR, 0);
        s.assign(arr, size); 
        delete arr;
    }

    void Corpus::_dictBcast()
    {
        std::vector<std::string> keys;
        std::vector<int> values;
        int size = (int)m_wordids.size();
        if (MPI::COMM_WORLD.Get_rank() == 0) {
            keys.reserve(size);
            values.reserve(size);
            for (auto it = m_wordids.begin(); it != m_wordids.end(); it++) {
                keys.push_back(it->first);
                values.push_back(it->second);
            }
        }
        MPI::COMM_WORLD.Bcast(&size, 1, MPI::INT, 0);
        if (MPI::COMM_WORLD.Get_rank() != 0) {
            keys.resize(size);
            values.resize(size);
        }
        for (int i = 0; i < size; ++i) {
            str_bcast(keys.at(i));
        }
        MPI::COMM_WORLD.Bcast(values.data(), size, MPI::INT, 0);
        for (int i = 0; i < size; ++i) {
            m_wordids[keys.at(i)] = values.at(i);
        }
    }
    // this constructor is used with mpi
    Corpus::Corpus(const std::string inputfn, const std::string outputfn, const int rank, const int procs,
                   int parSeq, real_t fraction, int truncSeqLength, bool fracShuf, bool seqShuf, real_t noiseDev,
                   std::string cachePath, std::unordered_map<std::string, int>* wordids, int constructDict, int max_vocab_size, int appearing_threshold)
        : m_fractionShuffling(fracShuf)
        , m_sequenceShuffling(seqShuf)
        , m_noiseDeviation   (noiseDev)
        , m_parallelSequences(parSeq)
        , m_totalTimesteps   (0)
        , m_minSeqLength     (std::numeric_limits<int>::max())
        , m_maxSeqLength     (std::numeric_limits<int>::min())
        , m_curFirstSeqIdx   (-1)
        , m_nextid(0)
        , m_inputPatternSize (1)
        , m_max_vocab_size( (max_vocab_size == -1)? INT_MAX : max_vocab_size )
        , m_appearing_threshold(appearing_threshold)
    {
        int ret;
        int ncid;

        m_fixed_wordDict = false;
        if (wordids != NULL){
            m_wordids = *wordids;
           if (!constructDict)
                m_fixed_wordDict = true;
            m_nextid = (long long int)m_wordids.size();
        }

        if (fraction <= 0 || fraction > 1)
            throw std::runtime_error("Invalid fraction");

        // open the cache file
        std::string tmpFileName = "";
        if (cachePath == "") {
            tmpFileName = (boost::filesystem::temp_directory_path() / boost::filesystem::unique_path()).string();
        }
        else {
            tmpFileName = cachePath + "/" + (boost::filesystem::unique_path()).string();
        }
        std::cerr << std::endl << "using cache file: " << tmpFileName << std::endl << "... ";
        m_cacheFileName = tmpFileName;
        m_cacheFile.open(tmpFileName.c_str(), std::fstream::in | std::fstream::out | std::fstream::binary | std::fstream::trunc);
        if (!m_cacheFile.good())
            throw std::runtime_error(std::string("Cannot open temporary file '") + tmpFileName + "'");
        std::vector<std::string> files = std::vector<std::string>();
        files.push_back(inputfn);
        // TODO rewrite  no need?
        // /*
        if (!m_fixed_wordDict && rank == 0){
            if (m_wordids.find("<s>") == m_wordids.end())
                m_wordids["<s>"] = m_nextid++;
            if (m_wordids.find("</s>") == m_wordids.end())
                m_wordids["</s>"] = m_nextid++;
            if (m_wordids.find("<UNK>") == m_wordids.end())
                m_wordids["<UNK>"] = m_nextid++;
            _makeWordDict(files);
            m_fixed_wordDict = true;
        }
        // */

        // pre loading and make binary data of mini-batch
        // TODO: broadcast m_wordids
        if (rank == 0) {
            _writeTemp(inputfn, outputfn, truncSeqLength);
        }
        // m_wordids is already made

        //broadcast m_wordids
        _dictBcast();
        /*
        boost::mpi::communicator world;
        mpiumap<std::string, int> mm(world, m_wordids);
        mm.mmbroadcast(); // rank-0's map is stored in mm
        if (rank != 0)
            mm.getMap(&m_wordids);
        */

        int *buf;
        int dataAmount; 
        int bufsize = truncSeqLength - 1;
        int mallocsize = 268435456; // 1gb 
        //{{  // parallel loading from binary file
        int readsize;
        int l = 0;
        MPI::Status status;
        MPI::File f = MPI::File::Open(MPI::COMM_WORLD, outputfn.c_str(),
                                          MPI::MODE_RDONLY, MPI::INFO_NULL);
        MPI::Offset fsize = f.Get_size() / sizeof(int);
        dataAmount = (fsize / bufsize) / procs; // number-of-mini-batch / procs
        if (mallocsize > (fsize / procs) ) 
            readsize = dataAmount * bufsize;
        else 
            readsize = (mallocsize / bufsize) * bufsize;
        buf = (int*) malloc(readsize * sizeof(int));
        int max_iteration = ((dataAmount * bufsize) / readsize);
        while (l < max_iteration) {
            
            f.Set_view( (procs * l * readsize) * sizeof(int) + rank * readsize * sizeof(int), MPI_INT, MPI_INT, "native", MPI::INFO_NULL);
            f.Read_all((void*)buf, readsize, MPI_INT, status);
            ++l;
            
            std::vector<sequence_t> sequences;

            int seqidxCount = 0;
            int loadLength, startpos;
            std::string line;
            for (int i = 0; i < (readsize / bufsize); ++i) {
                // reading first
                startpos = i * bufsize;
                Cpu::int_vector inputs = _makeInputFromBuffer(buf, startpos, bufsize);
                Cpu::int_vector targets = _makeTargetFromBuffer(buf, startpos, bufsize);
                m_totalTimesteps += loadLength;

                // making sequence
                int loaded = 0;
                {{ // make sequence
                    sequence_t seq;
                    seq.originalSeqIdx = i;
                    seq.length = inputs.size();

                    // read input patterns and store them in the cache file
                    seq.inputsBegin = m_cacheFile.tellp();


                    m_cacheFile.write((const char*)(inputs.data()), sizeof(int) * seq.length);
                    assert (m_cacheFile.tellp() - seq.inputsBegin == seq.length * sizeof(int));

                    m_minSeqLength = std::min(m_minSeqLength, seq.length);
                    m_maxSeqLength = std::max(m_maxSeqLength, seq.length);

                    // read targets and store them in the cache file
                    seq.targetsBegin = m_cacheFile.tellp();

                    // Cpu::int_vector targets = _makeTargetFromLine(line);
                    m_cacheFile.write((const char*)(targets.data()), sizeof(int) * seq.length);
                    assert (m_cacheFile.tellp() - seq.targetsBegin == seq.length * sizeof(int));
                    sequences.push_back(seq);
                }}
            }

            // create next fraction data and start the thread
            m_threadData.reset(new thread_data_t);
            m_threadData->finished  = false;
            m_threadData->terminate = false;
            m_threadData->thread    = boost::thread(&Corpus::_nextFracThreadFn, this);

            m_sequences.insert(m_sequences.end(), sequences.begin(), sequences.end());
        } // loading end

        free(buf);
        f.Close();
        m_totalSequences = m_sequences.size();
        m_outputPatternSize = std::min(m_max_vocab_size, (int)m_wordids.size());
        printf("outputPatternSize: %d(m_max_vocab_size %d, m_wordids.size %d)\n", m_outputPatternSize, m_max_vocab_size, m_wordids.size());
        printf("max_vocab_size %d, m_wordids.size %d)\n", max_vocab_size, INT_MAX);
        // sort sequences by length
        if (Configuration::instance().trainingMode())
            std::sort(m_sequences.begin(), m_sequences.end(), internal::comp_seqs);
    }
#endif //_MYMPI

    Corpus::~Corpus()
    {
        // terminate the next fraction thread
        if (m_threadData) {
            {{
                boost::lock_guard<boost::mutex> lock(m_threadData->mutex);
                m_threadData->terminate = true;
                m_threadData->cv.notify_one();
            }}

            m_threadData->thread.join();
        }
    }

    bool Corpus::isClassificationData() const
    {
        return m_isClassificationData;
    }

    bool Corpus::empty() const
    {
        return (m_totalTimesteps == 0);
    }

    boost::shared_ptr<CorpusFraction> Corpus::getNextFraction()
    {
        // initial work
        if (m_curFirstSeqIdx == -1) {
            boost::unique_lock<boost::mutex> lock(m_threadData->mutex);
            m_threadData->taskFn = boost::bind(&Corpus::_makeFirstFractionTask, this);
            m_threadData->finished = false;
            m_threadData->cv.notify_one();
            m_curFirstSeqIdx = 0;
        }

        // wait for the thread to finish
        boost::unique_lock<boost::mutex> lock(m_threadData->mutex);
        while (!m_threadData->finished)
            m_threadData->cv.wait(lock);

        // get the fraction
        boost::shared_ptr<CorpusFraction> frac;
        if (m_curFirstSeqIdx < (int)m_sequences.size()) {
            frac = m_threadData->frac;
            m_curFirstSeqIdx += m_parallelSequences;

            // start new task
            if (m_curFirstSeqIdx < (int)m_sequences.size())
                m_threadData->taskFn = boost::bind(&Corpus::_makeFractionTask, this, m_curFirstSeqIdx);
            else
                m_threadData->taskFn = boost::bind(&Corpus::_makeFirstFractionTask, this);

            m_threadData->finished = false;
            m_threadData->cv.notify_one();
        }
        else  {
            m_curFirstSeqIdx = 0;
        }

        return frac;
    }

    int Corpus::totalSequences() const
    {
        return m_totalSequences;
    }

    int Corpus::totalTimesteps() const
    {
        return m_totalTimesteps;
    }

    int Corpus::minSeqLength() const
    {
        return m_minSeqLength;
    }

    int Corpus::maxSeqLength() const
    {
        return m_maxSeqLength;
    }

    int Corpus::inputPatternSize() const
    {
        return m_inputPatternSize;
    }

    int Corpus::outputPatternSize() const
    {
        return m_outputPatternSize;
    }

    Cpu::real_vector Corpus::outputMeans() const
    {
        return m_outputMeans;
    }

    Cpu::real_vector Corpus::outputStdevs() const
    {
        return m_outputStdevs;
    }

    std::string Corpus::cacheFileName() const
    {
        return m_cacheFileName;
    }

    std::unordered_map<std::string, int>* Corpus::dict()
    {
        return &m_wordids;
    }

    boost::shared_ptr<CorpusFraction> Corpus::getNewFrac()
    {
        boost::shared_ptr<CorpusFraction> p(new CorpusFraction());
        return p;
    }

} // namespace data_sets
