#include "wsdDocument.hpp"

namespace wsd {

AmbiguousWord::AmbiguousWord(
    int line
    int wordposition
    std::string _word,
    std::vector<std::string> _senses
)
: position (std::make_tuple(line, wordposition)),
  word     (_word),
  candsize (_senses.size()),
  senses   (_senses)
{};

AmbiguousWord::AmbiguousWord(const AmbiguousWord& w)
: position (w.position),
  word     (w.word),
  candsize (w.senses.size()),
  senses   (w.senses)
{};

bool AmbiguousWord::operator< (const AmbiguousWord& w) const
{
    return this->candsize < w.candsize;
};

bool AmbiguousWord::operator> (const AmbiguousWord& w) const
{
    return this->candsize < w.candsize;
};



wsdDocument::wsdDocument(const std::string& filename)
{

};
int max_len() const;
    int len() const;
    std::vector<std::vector<std::string>>* lines();
    std::unordered_set<std::string>* words();
};

}
