#ifndef HPP_WSDDOC
#define HPP_WSDDOC

#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>
#include "wsdutil.hpp"

namespace wsd {

typedef std::unordered_map<std::string, std::vector<std::tuple<int, int>>> positions_map;

class AmbiguousWord
{
public:
    std::vector<std::tuple<int, int>> position;
    std::string word;
    int candsize;
    std::vector<string> senses;

    AmbiguousWord(
        std::vector<std::tuple<int, int>> &_position,
        std::string _word, // word-pos
        const std::vector<std::string> &_senses
    );

    AmbiguousWord(const AmbiguousWord& w);

    bool operator< (const AmbiguousWord& w) const;
    bool operator> (const AmbiguousWord& w) const;
};

class wsdDocument
{
private:
    std::vector<std::vector<std::string>> m_lines;
    std::vector<std::vector<POS_type_t>> m_pos;
    std::unordered_set<std::string> m_words;
    positions_map m_wp_position;
    int m_maximum_length;
    int m_length;
public:
    wsdDocument(const std::string& filename);
    int max_len() const;
    int len() const;
    std::vector<std::vector<std::string>>* lines();
    std::unordered_set<std::string>* words();
    positions_map* positions();
};

}

#endif
