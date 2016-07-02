#ifndef HPP_WSDUTIL
#define HPP_WSDUTIL

#include <string>
#include <stringstream>
#include <ifstream>
#include <vector>
#include <unordered_map>
#include <cereal>
#include "../../currennt_lib/src/helpers/JsonClasses.hpp"
#include "../../currennt_lib/src/rapidjson/prettywriter.h"
#include "../../currennt_lib/src/rapidjson/filestream.h"


namespace wsd{

enum POS_type_t
{
    NOUN,
    VERB,
    ADJ,
    ADV,
    POS_OTHER
};

namespace util{


// store spliting result to target
void strSplit(std::string& s, std::vector<std::string> *target, char delim);

void getWordSynsetOfLexeme(std::string word, std::vector<std::string> *ret);

POS_type_t getPOStype(std::string postype);

inline std::string wordPosStr(const std::string& w, const POS_type_t& p);

void readLine(const std::string& line, std::vector<std::string>& words, std::vector<POS_type_t> pos);

void makeInput(const std::vector<std::string>& words,
              const std::unordered_map<std::string, int>& dic,
              Cpu::int_vector *input);

std::string _getWord(std::string w);

void makeTarget(const std::vector<std::string>& words,
                const std::unordered_map<std::string, int>& dic,
                Cpu::int_vector *target);

void importDictBinary(std::unordered_map<std::string, int> &m, std::string &fname);

void readJsonFile(rapidjson::Document *doc, const std::string &filename);

// return if char c is conained by string s
bool ifContain(std::string s, std::string c);

bool ifLexeme(std::string word);

std::tuple<std::string, std::string, POS_type_t> getLexemeSynset(std::string lexeme);

}
}

#endif
