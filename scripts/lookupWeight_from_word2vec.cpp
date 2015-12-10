#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <climits>

#include "../currennt_lib/src/helpers/JsonClasses.hpp"
#include "../currennt_lib/src/rapidjson/prettywriter.h"
#include "../currennt_lib/src/rapidjson/filestream.h"
#include "../currennt_lib/src/helpers/JsonClasses.hpp"

void make_dictSection(const helpers::JsonDocument &Object,
                      const helpers::JsonAllocator & allocator,
                      const std::vector<std::string> &words)
{
    rapidjson::Value dictSection(rapidjson::kArrayType);
    for(size_t i = 0; i < words.size(); ++i){
        rapidjson::Value wordObject(rapidjson::kObjectType);
        wordObject.AddMember("name", words[i].c_str(), allocator);
        wordObject.AddMember("id", i, allocator);
        dictSection.PushBack(wordObject, allocator);
    }
    Object->AddMember("word_dict", dictSection, allocator);
}

void make_weightSection(const helpers::JsonDocument &Object,
                        const helpers::JsonAllocator & allocator,
                        const std::vector<std::string> &words,
                        const std::vector<std::vector<std::string>> &vecs)
{
    rapidjson::Value weightsObject(rapidjson::kObjectType);
    rapidjson::Value weightsSection(rapidjson::kArrayType);
    for(size_t i = 0; i < words.size(); ++i){
            rapidjson::Value embeddingsSection(rapidjson::kObjectType);
            // create and fill the weight arrays
            rapidjson::Value embeddingWeightsArray(rapidjson::kArrayType);
            embeddingWeightsArray.Reserve(vecs.at(i).size(), allocator);
            for (int j = 0; j < vecs.at(i).size(); ++j)
                embeddingWeightsArray.PushBack(std::stof(vecs.at(i).at(j).c_str()), allocator);
            // fill the weights subsection
            embeddingsSection.AddMember("name", words.at(i).c_str(), allocator);
            embeddingsSection.AddMember("id", i, allocator);
            embeddingsSection.AddMember("array", embeddingWeightsArray, allocator);
            weightsSection.PushBack(embeddingsSection, allocator);
    }
    weightsObject.AddMember("lookup", weightsSection, allocator);
    Object->AddMember("weights", weightsObject, allocator);
}

void sortByCount(std::vector<std::string> *words,
                 std::unordered_map<std::string, int> *m,
                 const std::string &filename)
{
    std::unordered_map<std::string, int> counter;
    std::ifstream ifs(filename);
    std::string line, item, word;
    while (std::getline(ifs, line)){
        std::stringstream ss(line);
        while(std::getline(ss, word, ' ')){
            if (counter.find(word) == counter.end() )
                counter[word] = 1;
            else
                counter[word] += 1;
        }
    }
   // m[0, 1, 2]  are reserved for <s> </s> <UNK>
   counter["<s>"] = INT_MAX;
   counter["</s>"] = INT_MAX-1;
   counter["<UNK>"] = INT_MAX-2;

   std::sort(words->begin(), words->end(),
             [&] (const std::string& l, const std::string r){
                 return counter[l] > counter[r];
             });
   for (int i = 0; i < words->size(); ++i){
       m->insert(std::make_pair(words->at(i), i));
   }
}

int main(int argc, char* argv[]){
    std::string line, item, word;
    long long int N = 0;
    int d = 0;
    std::string ifname(argv[1]);
    std::string ofname(argv[2]);
    bool ifsort = false;
    std::string origin;
    if (argc > 3){
         ifsort = true;
         origin = argv[3];
   }
    // read input file
    std::ifstream ifs(ifname);
    std::getline(ifs, line);
    // get "N d"
    std::stringstream ss(line);
    ss >> N;
    ss >> d;
    if (N <= 0 || d <= 0){
        printf("invalid header 'N d'\n");
        return 1;
    }
    std::unordered_map<std::string, int> id;

    std::vector<std::vector<std::string>> vecs(N);
    for ( size_t i = 0; i < vecs.size(); ++i ){
        vecs.at(i).reserve(d);
    }
    std::vector<std::string> words;
    words.reserve(N);
    //just reading
    int c = 0;
    auto ifs_start = ifs.tellg();
    while ( std::getline(ifs, line) ){
        std::stringstream linestrm(line);
        std::getline(linestrm, word, ' ');
        words.push_back( word );
    }

    if(ifsort)
        sortByCount(&words, &id, origin);
    ifs.clear();
    //ifs.seekg(0, std::ios_base::beg);
    ifs.seekg(ifs_start);
    while ( std::getline(ifs, line) ){
        std::stringstream linestrm(line);
        std::getline(linestrm, word, ' ');
        while ( std::getline(linestrm, item, ' ')){
            if(ifsort)
                vecs.at(id[word]).push_back(item);
            else
                vecs.at(c).push_back(item);
        }
        ++c;
    }

    assert(words.size() == vecs.size());

    rapidjson::Document jsonDoc;
    jsonDoc.SetObject();
    // make jsonDoc
    make_dictSection(jsonDoc, &(jsonDoc.GetAllocator()), words);
    make_weightSection(jsonDoc, &(jsonDoc.GetAllocator()), words, vecs);

    // write it
    FILE *file = fopen(ofname.c_str(), "w");
    if (!file)
        throw std::runtime_error("Cannot open file");
    rapidjson::FileStream os(file);
    rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
    jsonDoc.Accept(writer);
    fclose(file);
    return 0;
}
