#ifndef HPP_WSDDOC

#define HPP_WSDDOC
#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>

namespace wsd {

class AmbiguousWord
{
public:
    std::tuple<int, int> position;
    std::string word;
    int candsize;
    std::vector<string> senses;

    AmbiguousWord(
        int line
        int wordposition
        std::string _word,
        std::vector<std::string> _senses
    );

    AmbiguousWord(const AmbiguousWord& w);

    bool operator< (const AmbiguousWord& w) const;
    bool operator> (const AmbiguousWord& w) const;
};

class wsdDocument
{
private:
    std::vector<std::vector<std::string>> m_lines;
    std::unordered_set<std::string> m_words;
    int m_maximum_length;
    int m_length;
public:
    wsdDocument(const std::string& filename);
    int max_len() const;
    int len() const;
    std::vector<std::vector<std::string>>* lines();
    std::unordered_set<std::string>* words();
};

}

#endif
