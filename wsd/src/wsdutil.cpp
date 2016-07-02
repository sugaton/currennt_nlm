
#include "wsdutil.hpp"

namespace wsd{

namespace util{


// store spliting result to target
void strSplit(std::string& s, std::vector<std::string> *target, char delim) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim))
        target->push_back(item);
}

void getWordSynsetOfLexeme(std::string word, std::vector<std::string> *ret) {
    ret->resize(2);
    std::vector<std::string> v;
    strSplit(word, &v, '-');
    std::string w, syn;
    if (v.size() == 2){
        w = v.at(0);
        syn = v.at(1);
    }
    else if (v.size() > 2) {
        w = v.at(0) + v.at(1);
        syn = "";
        for (int i = 2; i < v.size(); ++i)
            syn += v.at(i);
    }
    ret->at(0) = w;
    ret->at(1) = syn;
}

POS_type_t getPOStype(std::string postype)
{
    if (postype == "NOUN")
        return NOUN;
    else if (postype == "VERB")
        return VERB;
    else if (postype == "ADJ")
        return ADJ;
    else if (postype == "ADV")
        return ADV;
    else
        return POS_OTHER;
}

inline std::string wordPosStr(const std::string& w, const POS_type_t& p)
{
    return w + std::to_string((int)p);
};

void readLine(const std::string& line, std::vector<std::string>& words, std::vector<POS_type_t> pos)
{
    words.clear();
    pos.clear();
    std::stringstream ss(line);
    std::string word, item;
    POS_type_t postag;
    while (std::getline(ss, word, ' ')) {
        postag = POS_OTHER;
        std::stringstream lempos(word);
        std::getline(lempos, item, '@');
        words.push_back(item);
        std::getline(lempos, item, '@');
        postag = getPOStype(item);
        pos.push_back(postag);
    }
}


void makeInput(const std::vector<std::string>& words,
              const std::unordered_map<std::string, int>& dic,
              Cpu::int_vector *input)
{
    input->reserve(words.size());
    std::string unk = "<UNK>";
    int unk_int = dic.find(unk)->second;
    for (auto w : words) {
        auto it = dic.find(w);
        if (it == dic.end())
            input->push_back(unk_int);
        else
            input->push_back(it->second);
    }
}


std::string _getWord(std::string w)
{
    std::string word;
    std::stringstream ss(w);
    std::getline(ss, word, '%');
    return word;
}

void makeTarget(const std::vector<std::string>& words,
                const std::unordered_map<std::string, int>& dic,
                Cpu::int_vector *target)
{
    target->reserve(words.size());
    std::string unk = "<UNK>";
    int unk_int = dic.find(unk)->second;
    for (auto w : words) {
        if (w.find('%') != std::string::npos) {
            w = _getWord(w);
        }
        auto it = dic.find(w);
        if (it == dic.end())
            target->push_back(unk_int);
        else
            target->push_back(it->second);
    }
}

void importDictBinary(std::unordered_map<std::string, int> &m, std::string &fname)
{
    std::ifstream ifs(fname, std::ios::binary);
    cereal::BinaryInputArchive archive(ifs);
    archive(m);
}


void readJsonFile(rapidjson::Document *doc, const std::string &filename)
{
    // open the file
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Cannot open file");

    // calculate the file size in bytes
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    // read the file into a buffer
    char *buffer = new char[size + 1];
    ifs.read(buffer, size);
    buffer[size] = '\0';

    std::string docStr(buffer);
    delete buffer;

    // extract the JSON tree
    if (doc->Parse<0>(docStr.c_str()).HasParseError())
        throw std::runtime_error(std::string("Parse error: ") + doc->GetParseError());
}

// return if char c is conained by string s
bool ifContain(std::string s, std::string c) {
    return (s.find(c) != std::string::npos);
}

bool ifLexeme(std::string word) {
    if ( !ifContain(word, "%") )
        return false;
    std::vector<std::string> v;
    strSplit(word, &v, '-');
    for(std::string s : v)
        if (ifContain(s, "%"))
            return true;
    return false;
}

std::tuple<std::string, std::string, POS_type_t> getLexemeSynset(std::string lexeme)
{
    std::string synsets, tmp, word;
    std::stringstream lexss(lexeme);
    std::getline(lexss, word, '-');  // word
    std::getline(lexss, synsets, '-');

    std::stringstream ss(synsets);
    std::vector<std::string> v;
    std::string item, w, key;
    // std::cout << "getLexemeSynset: " << lexeme << "  :  " << word << " : " << synsets << std::endl;
    POS_type_t p = POS_OTHER;
    while (std::getline(ss, item, ',')) {
        v.push_back(item);
    }
    for (std::string s : v) {
        // s = "word"(w)%"sensekey"(key) #six%1:23:00::
        if (s == "") continue;
        std::stringstream temp(s);
        std::getline(temp, w, '%'); //get word
        if (w != word) continue;
        std::getline(temp, key, '%'); //get sensekey
        std::stringstream temp2(key);
        std::getline(temp2, item, ':'); // item = pos-info
        // std::cout << key << "'s pos_information:" << item << std::endl;

        switch (std::stoi(item)) {
            case 1:
                p = NOUN;
                break;
            case 2:
                p = VERB;
                break;
            case 3:
                p = ADJ;
                break;
            case 4:
                p = ADV;
                break;
            case 5:
                p = ADJ;
                break;
        }
        // printf("synset, pos:%s %d\n", s.c_str(), p);
        return std::make_tuple(word, s, p);
    }
    return std::make_tuple(word, "", p);

}


}
}
