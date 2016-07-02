
#include "consistent_wsd.hpp"

namespace wsd {

// State
State::State()
: m_logprob (0.0)
, m_wsdprob (0.0)
{};

State::State(const State& s)
: m_logprob (s.m_logprob)
, m_wsdprob (s.m_wsdprob)
, m_disambiguated (s.m_disambiguated)
{};


void State::transition(const std::string& wp, const std::string& s, const real_t& logp, const real_t& wsdp)
{
    m_disambiguated[wp] = s;
    m_logprob = logp;
    m_wsdprob += wsdp;
}

real_t State::score() const { return m_logprob + m_wsdprob; }
real_t State::wsdprob() const { return m_wsdprob; }

void State::replaceWord(const std::vector<std::vector<std::string>>& orig,
                 std::vector<std::vector<std::string>>& newone,
                 positions_map &wp_position)
{
    std::string wp, sense;
    int linep, wordp;
    newone.resize(orig.size());
    for (size_t i = 0; i < orig.size(); ++i) {
        newone.at(i).resize(orig.at(i).size());
        std::copy(orig[i].begin(), orig[i].end(), newone.at(i).begin());
    }
    for (auto it = m_disambiguated.begin(); it != m_disambiguated.end(); ++it) {
        wp = it->first;
        sense = it->second;
        for (const auto& tuple : wp_position[wp]) {
            std::tie(linep, wordp) = tuple;
            newone.at(linep).at(wordp) = sense;
        }
    }
}

//consistent_wsd

void consistent_wsd::_expandWDict()
{
    int c = (int)m_wdict.size();
    for (auto it = m_doc.words->begin(); it != m_doc.words->end(); ++it) {
        if (m_wdict.find(*it) == m_wdict.end())
            m_wdict[*it] = c++;
    }
    std::string syn, word;
    POS_type_t p;
    for (auto it = m_lexeme_emb.begin(); it != lexeme_emb.end(); ++it) {
        std::tie(word, syn, p) = util::getLexemeSynset(it->first);
        if (m_wdict.find(syn) == m_wdict.end())
            m_wdict[syn] = c++;
    }
}

void consistent_wsd::_expandLookup(std::string& embfile)
{
    std::string syn, word;
    POS_type_t p;

    m_net->loadEmbeddings(embfile);
    for (auto it = m_lexeme_emb.begin(); it != lexeme_emb.end(); ++it) {
        std::tie(word, syn, p) = util::getLexemeSynset(it->first);
        m_net->LookupLayer().replaceEmbeddings(syn, *(it->second));

    }
}

// create AmbiguousWord and correspond Matrix for wsd
void consistent_wsd::_createAmbWords(
    std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<Cpu::real_vector>>> &_lexeme_param, int d)
{
    std::vector<std::string> v;
    for (auto wpit = _lexeme_param.begin(); wpit != _lexeme_param.end(); ++wpit) {
        Cpu::real_vector arr;
        v.clear();
        wp = wpit->first;
        v.reserve(wpit->second.size());
        arr.resize(wpit->second.size() * d);
        int l = 0;
        for (auto lxit = wpit->second.begin(); lxit != wpit->second.end(); ++lxit) {
            v.push_back(lxit->first);
            thrust::copy(lxit->second->begin(), lxit->second->end(), arr.begin() + l * d);
            ++l;
        }
        std::unique_ptr<Matrix<Cpu>> Mptr(new Matrix<Cpu>(&arr, d, N));
        m_wsd_params[wp] = std::move(Mptr);
        ambiguous_words.emplace_back(m_doc.m_wp_position[wp], wp, v);
    }
}
void consistent_wsd::_loadLexemes(const std::string& embfile, const std::string& paramfile)
{
    std::ifstream ifs(embfile);
    std::vector<std::string> Nd;
    std::string line, item, lexeme, word, syn, wp;
    int d;
    char delim = ' ';
    POS_type_t p;
    std::getline(ifs, line);
    util::strSplit(line, &Nd, delim);
    d = std::stoi(Nd[1]);
    std::vector<std::string> word_syn;
    printf("loading lexeme embeddings... \n");
    while(std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::getline(ss, lexeme, delim);
        if (!util::ifLexeme(lexeme)) {
            continue;
        }
        std::tie(word, syn, p) = util::getLexemeSynset(lexeme);

        if (m_word_synsets.find(word) == m_word_synsets.end()) {
            m_word_synsets[word] = std::vector<std::string>();
        }
        m_word_synsets[word].push_back(lexeme);

        std::unique_ptr<Cpu::real_vector> array(new Cpu::real_vector());
        array->reserve(d);
        while(std::getline(ss, item, delim))
            array->push_back(std::stof(item));
            // lexeme_emb[lexeme]->push_back(std::stof(item));
        m_lexeme_emb[lexeme] = std::move(array);

    }
    //
    std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<Cpu::real_vector>>> _lexeme_param;
    std::ifstream ifs2(paramfile);
    // read head
    std::getline(ifs, line);
    util::strSplit(line, &Nd, delim);
    while(std::getline(ifs2, line)) {
        std::stringstream ss(line);
        std::getline(ss, lexeme, delim);
        //check if this word is lexeme
        if (!util::ifLexeme(lexeme)) {
            continue;
        }

        std::tie(word, syn, p) = util::getLexemeSynset(lexeme);
        // if word is contained in doc
        if (m_doc.m_words.find(word) == m_doc.m_words.end())
            continue;
        // get word-pos string
        wp = util::wordPosStr(word, p);
        if (m_doc.m_wp_position.find(wp) == m_doc.m_wp_position.end())
            continue;
        // read array
        std::unique_ptr<Cpu::real_vector> array(new Cpu::real_vector());
        array->reserve(d);
        while(std::getline(ss, item, delim))
            array->push_back(std::stof(item));
        if (_lexeme_param.find(wp) == _lexeme_param.end())
            _lexeme_param[wp] = std::unordered_map<std::unique_ptr<Cpu::real_vector>>();
        _lexeme_param[wp][lexeme] = std::move(array);
    }
    // create AmbiguousWords
    _createAmbWords(_lexeme_param, d);

}


consistent_wsd::consistent_wsd(
    const std::string &target_file,
    const std::string &wsd_output_file,
    const std::string &network_file,
    const std::string &model_importDir,
    const std::string &lexeme_param_file,
    const std::string &lexeme_emb_file,
    const std::string &input_emb_file)
: m_out_file (wsd_output_file)
{
    rapidjson::Document netDoc;

    //import wdict
    if (model_importDir != "") {
        std::string fname = model_importDir + "/wdict.cereal";
        util::importDictBinary(m_wdict, fname);
    }
    else {
        throw std::runtime_error("importDir/wdict.cereal does not exist!");
        return;
    }


    //load network_file
    util::readJsonFile(&netDoc, networkFile);
    // read document
    m_doc = wsdDocument(target_file);

    // construct network
    m_net = std::make_shared<NeuralNetwork>(
                netDoc,          //
                m_doc.len(),     // parallelSequences
                m_doc.max_len(), // maxSeqLength
                1,               //
                m_wdict.size(),  //outputsize
                m_wdict.size(),  //vocab_size
                1);
    m_net->setWordDict(&m_wdict);
    m_net->importWeightsBinary(model_importDir, &m_wdict);

    _expandWDict();
    _expandLookup(input_emb_file);
    _loadLexemes(config.lexeme_file());
    std::sort(ambiguous_words.begin(),
              ambiguous_words.end(),
              std::less<AmbiguousWord>());
};


// for _wsd

void consistent_wsd::_makeFrac(
    std::vector<std::vector<std::string> lines,
    data_sets::CorpusFraction& frac,
    const std::unordered_map<std::string, int>& dic,
    int outputSize)
{
    int maxSeq = m_doc.max_len();
    int minSeq = m_doc.min_len();
    int parallelSequences = m_doc.len();

    int context_left = Configuration::instance().inputLeftContext();
    int context_right = Configuration::instance().inputRightContext();
    int context_length = context_left + context_right + 1;
    int output_lag = Configuration::instance().outputTimeLag();

    frac->use_intInput();
    frac->set_inputPatternSize(context_length);
    frac->set_outputPatternSize(size);
    frac->set_maxSeqLength(maxSeq);
    frac->set_minSeqLength(minSeq);


    for (int seqIdx = 0; seqIdx < parallelSequences; ++seqIdx) {
        if (seqIdx < (int)m_sequences.size()) {

            CorpusFraction::seq_info_t seqInfo;
            seqInfo.originalSeqIdx = seqIdx;
            seqInfo.length         = lines.at(idx).size();
            seqInfo.seqTag         = "";
            frac->set_seqInfo(seqInfo);
        }
    }

    // allocate memory for the fraction
    frac->intinput()->resize(frac->maxSeqLength() * frac->inputPatternSize() * parallelSequences, 0);
    // resize of targetclass and patTypes
    frac->vectorResize(parallelSequences, PATTYPE_NONE, -1);

    Cpu::int_vector input = Cpu::int_vector();
    Cpu::int_vector targetClasses = Cpu::int_vector();

    for (int i = 0; i < parallelSequences; ++i) {
        std::vector<std::string>& words = lines.at(i);

        util::makeInput(words, dic, &input);
        for (int timestep = 0; timestep < words.size(); ++timestep) {
            int srcStart = timestep;
            int tgtStart = frac->inputPatternSize() * (timestep * parallelSequences + i);
            thrust::copy_n(input.begin() + srcStart, 1, frac->intinput()->begin() + tgtStart);
            ++offset_out;
            // }
        }

        util::makeTarget(words, dic, &targetClasses);
        for (int timestep = 0; timestep < words.size(); ++timestep) {
            int tgt = 0; // default class (make configurable?)
            if (timestep >= output_lag)
                tgt = targetClasses[timestep - output_lag];
            frac->setTargetClasses(timestep * parallelSequences + i, tgt);
        }
        // pattern types
        for (int timestep = 0; timestep < words.size(); ++timestep) {
            Cpu::pattype_vector::value_type patType;
            if (timestep == 0)
                patType = PATTYPE_FIRST;
            else if (timestep == words.size() - 1)
                patType = PATTYPE_LAST;
            else
                patType = PATTYPE_NORMAL;
            frac->setPatTypes(timestep * parallelSequences + i , patType);
        }
    }
}

namespace internal {
    struct residue_eq
    {
        int size;
        residue_eq(const int &_size) : size(_size){};
        bool operator() (const int &i, const int &j) const {
            return (i / size == j / size);
        }
    };

    struct residueRow_eq
    {
        int size;
        residueRow_eq(const int &_size) : size(_size){};
        bool operator() (const int &i, const int &j) const {
            return (i % size == j % size);
        }
    };

    struct logsumexp
    {
        logsmexp();
        bool operator() (const real_t& l, const real_t& r)
        {
            return (l > r)? l + log( 1 + exp(l - r)): r + log( 1 + exp(r - l));
        }
    };
    struct minusCol
    {
        real_t* data;
        int colSize;
        minusCol(real_t* data_, int size) : data(data_), colSize(size){};
        real_t operator() (const int& idx, const real_t& val)
        {
            return val - data[idx / colSize];
        }
    }
}

void _copy_correspond(
    Cpu::real_vector &org,
    Cpu::real_vector &target,
    std::vector<std::tuple<int, int>> tpls,
    int size,
    int seqlen)
{
    int line, word, start;
    int i = 0;
    for (size_t i = 0; i < tpls.size(); ++i) {
        std::tie(line, word) = tpls[i];
        start = (line * seqlen + words) * size;
        thrust::copy(
            org.begin() + start,
            org.begin() + start + size,
            target.begin() + i * size
        );
    }
}

real_t consistent_wsd::_calcWsdScore(data_sets::CorpusFraction& frac, const AmbiguousWord& aw, Cpu::real_vector& ret_)
{
    std::string wp = aw.word;
    int size = m_lexeme_emb.begin()->second->size(); // output-layer's size
    // int stateLen =  m_doc.max_len() * m_doc.len();
    int stateLen =  *(m_doc.positions)[wp].size();
    int rsize =  stateLen * aw.senses.size();
    //forward
    m_net->loadSequences(*frac);
    m_net->computeForwardPass();

    Cpu::real_vector outputs = m_net->last_layer();
    Cpu::real_vector outputs_(size * stateLen);
    Cpu::real_vector result(rsize);
    Cpu::real_vector sum_(stateLen);
    // outputs -> outputs_
    _copy_correspond(outputs, outputs_, *(m_doc.positions)[wp], size, doc.max_len());

    Matrix<Cpu> Mo(&outputs_, size, stateLen);
    Matrix<Cpu> Mres(&result, aw.senses.size(), stateLen);
    Mres.assignProduct(*(m_wsd_params[wp]), true, Mo, false);

    // logsumexp for each
    thrust::reduce_by_key(
        thrust::counting_iterator<int>(0),         //keys
        thrust::counting_iterator<int>(0) + rsize,
        result.begin(),                            // input
        thrust::make_discard_iterator(),           // output-keys
        sum_.begin(),                              // output-result
        internal::residue_eq(aw.senses.size()),    // binary-pred
        internal::logsumexp()
    );

    // calculate logprob :( result(i) -= sum_(i / colsize) )
    thrust::transform(
        thrust::counting_iterator<int>(0),         // input1
        thrust::counting_iterator<int>(0) + rsize,
        result.begin(),                            // input2
        result.begin(),                            // result
        internal::minusCol(helpers::getRawPointer(sum_), aw.senses.size())
    );

    // reduce by row
    thrust::reduce_by_key(
        thrust::counting_iterator<int>(0),         //keys
        thrust::counting_iterator<int>(0) + rsize,
        result.begin(),                            // input
        thrust::make_discard_iterator(),           // output-keys
        ret_.begin(),                              // output-result
        internal::residueRow_eq(aw.senses.size())  // binary-pred
    )

}

void consistent_wsd::_pushCandidate(
    Cpu::real_vector& ret,
    std::vector<std::shared_ptr<State>> cand,
    State& curstate,
    AmbiguousWord& aw,
    real_t logp,
    int beam_size)
{
    //sort senses
    std::vector<std::pair<int, std::string>> v;
    v.reserve(aw.senses.size());
    for (size_t i = 0; i < aw.senses.size(); i++) {
        v.emplace_back(ret.at(i), aw.senses.at(i));
    }
    std::sort(v.begin(), v.end(),
              [](const std::pair<int, std::string> &l, const std::pair<int, std::string> &r)
              { return l.first > r.first; });
    // stock top-K candidate in cand
    int csize = (v.size() > beam_size)? beam_size : v.size();
    for (size_t i = 0; i < csize; i++) {
        cand.emplace_back(new State(curstate));
        cand.back()->transition(aw.word, v.at(i).second, logp, v.at(i).first);
    }
}

void consistent_wsd::_writeResult(const State& s)
{
    std::vector<std::vector<std::string>> result;
    s.replaceWord(*(m_doc.lines()), result, *(m_doc.positions()));

    std::ofstream ofs(m_out_file);
    for (const auto& line : result) {
        for (const std::string& word : line)
            ofs << word << " ";
        ofs << std::endl;
    }
}

bool operator> (const std::shared_ptr<State>& l, const std::shared_ptr<State>& r) { return *l > *r; }

void consistent_wsd::run (int beam_size, BeamSearch_type_t search_type)
{
    real_t logp;
    std::vector<std::shared_ptr<State>> beam;
    std::vector<std::shared_ptr<State>> cand;
    beam.reserve(beam_size);
    beam.reserve(beam_size * beam_size);

    std::vector<std::vector<std::string> lines;
    boost::shared_ptr<data_sets::CorpusFraction> frac = m_corpus.getNewFrac();
    for (const AmbiguousWord& aw : ambiguous_words) {
        cand.clear();
        for (std::shared_ptr<State> statep : beam) {
            statep.replaceWord(*m_doc.lines(), lines, *(m_doc.positions()));
            _makeFrac(lines, *frac, m_wdict, m_net->postOutputLayer().size());
            logp = _calcWsdScore(frac, aw, ret_);
            _pushCandidate(ret_, cand, *statep, aw, logp, beam_size);
        }
        beam.clear();
        std::sort(cand.begin(), cand.end(), std::greater<std::shared_ptr<State>>());
        int Bsize = (cand.size() > beam_size)? beam_size : cand.size();
        beam.resize(Bsize);
        for (size_t i = 0; i < Bsize; i++) {
            beam.at(i) = std::move(cand.at(i));
        }
    }
    _writeResult(beam.at(0));
}

}
