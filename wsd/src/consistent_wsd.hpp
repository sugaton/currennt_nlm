#ifndef HPP_CONSISTENT_WSD
#define HPP_CONSISTENT_WSD

#include <fstream>
#include "wsdDocument.hpp"
#include "../currennt_lib/src/corpus/CorpusFraction.hpp"
#include "../currennt_lib/src/NeuralNetwork.hpp"


namespace wsd{

enum BeamSearch_type_t {
    BST_L2R,
    BST_S2C
};

class State
{
private:
    real_t m_logprob;
    real_t m_wsdprob;
    std::unordered_map<std::string, std::string> m_disambiguated;
public:
    State(const State& s);
    void transition(const std::string& w, const std::string& s, const real_t& logp, const real_t& wsdp);
    real_t score() const;
    real_t wsdprob() const;
    void replaceWord(const std::vector<std::vector<std::string>>& orig,
                 std::vector<std::vector<std::string>>& newone,
                 positions_map &wp_position)
}

class consistent_wsd
{
private:
    wsdDocument m_doc;
    std::shared_ptr<NeuralNetwork<Cpu>> m_net;
    std::unique_ptr<data_sets::Corpus> m_corpus;
    std::unordered_map< std::string, std::vector<std::string> > m_word_synsets;
    std::unordered_map< std::string, std::unique_ptr<Matrix<Cpu>>> m_wsd_params;
    std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>> m_lexeme_emb;
    std::vector<AmbiguousWord> ambiguous_words;
    std::unordered_map< std::string, int> m_wdict;
    std::string m_out_file
    
// for public function
private:
    // for constructor
    void _expandWDict();
    void _expandLookup(std::string& embfile);
    void _createAmbWords(std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<Cpu::real_vector>>> &_lexeme_param, int d);
    void _loadLexemes(const std::string& embfile, const std::string& paramfile)

    // for _wsd
    void _makeFrac(
        std::vector<std::vector<std::string> lines,
        data_sets::CorpusFraction& frac,
        const std::unordered_map<std::string, int>& dic,
        int outputSize);

    real_t _calcWsdScore(data_sets::CorpusFraction& frac, const AmbiguousWord& aw, Cpu::real_vector& ret_);

    void _pushCandidate(
        Cpu::real_vector& ret,
        std::vector<std::shared_ptr<State>> cand,
        State& curstate,
        AmbiguousWord& aw,
        real_t logp,
        int beam_size);

    void _writeResult(const State& s);

public:
    consistent_wsd(
        const std::string &target_file,
        const std::string &wsd_output_file,
        const std::string &network_file,
        const std::string &model_importDir,
        const std::string &lexeme_param_file,
        const std::string &lexeme_emb_file,
        const std::string &input_emb_file
    );

    void run(int beam_size,BeamSearch_type_t search_type);
};


} //namespace wsd
