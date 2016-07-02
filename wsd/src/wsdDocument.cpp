#include "wsdDocument.hpp"
#include "wsdutil.hpp"

namespace wsd {

AmbiguousWord::AmbiguousWord(
    std::vector<std::tuple<int, int>> &_position,
    std::string _word,
    const std::vector<std::string>& _senses
)
: position (_position),
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



inline void updateSet(std::unordered_set<std::string>* s, std::vector<std::string>* v)
{
    for (const std::string& item : *v)
        s->insert(item);
}

wsdDocument::wsdDocument(const std::string& filename)
{
    std::ifstream ifs(filename);
    std::string line, wp;
    int l = 0;
    while (std::getline(ifs, line)) {
        std::vector<std::string> words;
        std::vector<POS_type_t> pos;
        util::readLine(line, words, pos);
        m_maximum_length = (words.size() > m_maximum_length) ? words.size() : m_maximum_length;
        updateSet(&m_words, &words);

        // set position
        for (int i = 0; i < words.size(); ++i) {
            wp = util::wordPosStr(words.at(i), pos.at(i));
            if (m_wp_position.find(wp) == m_wp_position.end())
                m_wp_position[wp] = std::vector<std::tuple<int, int>>();
            m_wp_position[wp].push_back(std::make_tuple(l, i));
        }
        m_lines.push_back(words);
        m_pos.push_back(pos);
        ++l;
    }
    m_length = m_lines.size();
};

int wsdDocument::max_len() const { return m_maximum_length}
int wsdDocument::len() const { return m_length; }

std::vector<std::vector<std::string>>* wsdDocument::lines();
{
    return &m_lines;
};

std::unordered_set<std::string>* wsdDocument::words()
{
    return &m_words;
};

}
