#include <fstream>
#include "wsdDocument.hpp"
#include "../currennt_lib/src/corpus/CorpusFraction.hpp"
#include "../currennt_lib/src/NeuralNetwork.hpp"


namespace wsd{

enum BeamSearch_type_t {
    BST_L2R,
    BST_S2C
};


class consistent_wsd
{
private:
    wsdDocument m_doc;
    NeuralNetwork m_net;
    std::unique_ptr<data_sets::Corpus> m_corpus;
    std::unordered_map< std::string, std::vector<std::string> > m_word_synsets;
    std::unordered_map< std::string, std::unique_ptr<Matrix<Cpu>>> m_wsd_params;
    std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>> m_lexeme_emb;
    std::vector<AmbiguousWord> ambiguous_words;

    void loadLexemes(const std::string& filename);
        // std::unordered_map< std::string, std::vector<std::string> >& word_synsets,
        // std::unordered_map< std::string, std::unique_ptr<Cpu::real_vector>>& lexeme_emb);
public:
    consistent_wsd();
    consistent_wsd(
        const std::string &target_file,
        const std::string &wsd_output_file,
        const std::string &network_file,
        const std::string &model_importDir,
        const std::string &lexeme_param_file,
        const std::string &lexeme_emb_file
    );

    void run(int &beam_size,BeamSearch_type_t search_type);
};


} //namespace wsd
