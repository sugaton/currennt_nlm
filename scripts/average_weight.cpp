#include <iostream>
#include <algorithm>
#include <functional>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <stdarg.h>
#include <sstream>
#include <cstdlib>
#include <math.h>
#include <iomanip>

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/binary.hpp>

#include "../currennt_lib/src/helpers/JsonClasses.hpp"
#include "../currennt_lib/src/rapidjson/prettywriter.h"

#include "../currennt_lib/src/NeuralNetwork.hpp"


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

std::vector<std::string> make_layerList(std::string networkFile, std::string dir)
{
    rapidjson::Document netDoc;
    readJsonFile(&netDoc, networkFile);
    rapidjson::Value &layersSection  = (netDoc)["layers"];
    //NeuralNetwork<Cpu> neuralNetwork(netDoc, 1, 1, 256, 10, 10, 1);
    std::vector<std::string> tmp;
    for (rapidjson::Value::ValueIterator layerChild = layersSection.Begin(); layerChild != layersSection.End(); ++layerChild) {
        tmp.push_back((*layerChild)["name"].GetString());
    }

    std::vector<std::string> llist = std::vector<std::string>();
    for (std::string l : tmp) {
        if (boost::filesystem::exists(dir + "/" + l) && l != "lookup")
            llist.push_back(l);
    }
    return llist;
}



void importWeightsBinary(const std::string &filename, std::vector<float> &weight)
{
    std::ifstream ifs(filename, std::ios_base::binary);
    size_t size;
    float item;
    ifs.read((char*) &size, sizeof(size_t));
    weight.resize(size);
    for (size_t i = 0; i < size; ++i) {
        ifs.read((char*) &item, sizeof(float));
        weight[i] = item;
    }
}

void load_weights(const std::vector<std::string> &llist, std::vector<std::vector<float>> &weights, const std::string dirname)
{
    std::string  filename;
    weights.resize(llist.size());
    std::string name;
    assert(llist.size() == weights.size());
    for (int i = 0; i < llist.size(); ++i) {
        name = llist.at(i);
        filename = dirname + "/" + name;
        assert(boost::filesystem::exist(filename));
        importWeightsBinary(filename, weights.at(i));
    }
}


void exportWeightsBinary(const std::string &filename, const std::vector<float> &weight)
{
    std::ofstream ofs(filename, std::ios_base::binary);
    size_t size = weight.size();
    float item;
    ofs.write((const char*) &size, sizeof(size_t));
    for (size_t i = 0; i < weight.size(); ++i) {
        item = weight[i];
        ofs.write((const char*) &item, sizeof(float));
    }
}

void write_weights(std::string layername, std::vector<float> &weights, const std::string dirname)
{
    std::string filename;
    filename = dirname + "/" + layername;
    exportWeightsBinary(filename, weights);
}

void load_dict(const std::vector<std::string> &llist, std::unordered_map<std::string, int> &dict, const std::string& dirname)
{
    std::string filename;
    filename = dirname + "/wdict.cereal";
    assert(boost::filesystem::exist(filename));
    std::ifstream ifs(filename, std::ios::binary);
    cereal::BinaryInputArchive archive(ifs);
printf("%s:%s at line %d\n", __FILE__, __FUNCTION__, __LINE__);
    archive(dict);
printf("%s:%s at line %d\n", __FILE__, __FUNCTION__, __LINE__);
}

std::vector<float> average_weight(const std::vector<std::vector<float>> &weights)
{
    assert(weights.size() > 0);
    std::vector<float> average = std::vector<float>(weights.at(0).size(), 0.0);
    for (auto v : weights) {
        std::transform(v.begin(), v.end(), average.begin(), average.begin(), std::plus<float>());
    }
    for (int i = 0; i < average.size(); ++i) {
        average[i] /= (float)weights.size();
    }
    return average;
}

std::vector<std::vector<float>> average_weight_output(const std::vector<std::vector<float>> &weights, const std::vector<std::unordered_map<std::string, int>> &dict)
{
    std::vector<std::vector<float>> averaged_weights(weights.size());
    int size = weights[0].size();
    for (int i = 0; i < weights.size(); ++i)
        averaged_weights[i].resize(size);
    std::unordered_map<std::string, long long int> global = std::unordered_map<std::string, long long int>();
    std::string w;
    int idx;
    long long int c = 0;
    for (auto dic : dict) {
        for (auto it = dic.begin(); it != dic.end(); ++it) {
            w = it->first;
            // idx = it->second;
            if (global.find(w) == global.end())
                global[w] = c++;
        }
    }

    for (auto it = global.begin(); it != global.end(); ++it) {
        w = it->first;
        idx = it->second;
        std::vector<float> average(size, 0.0);
        int count = 0;
        // average
        for (int i = 0; i < dict.size(); ++i) {
            auto itw = dict[i].find(w);
            if (itw != dict[i].end()) {
                int idxw = itw->second * size;
                std::transform(weights[i].begin() + idxw,
                               weights[i].begin() + idxw + size,
                               average.begin(),
                               average.begin(),
                               std::plus<float>());

                ++count;
            }
        }
        for (int i = 0; i < average.size(); ++i)
            average[i] /= (float)count;

        // copy
        for (int i = 0; i < dict.size(); ++i) {
            auto itw = dict[i].find(w);
            if (itw != dict[i].end()) {
                int idxw = itw->second * size;
                std::copy(average.begin(),
                          average.end(),
                          averaged_weights[i].begin() + idxw);
            }
        }
    }
    return averaged_weights;
}

int main(int argc, char** argv)
{
    // get args
    std::string networkfn(argv[argc - 1]);
    std::string iter(argv[argc - 2]);
    std::string rootdir(argv[argc - 3]);
    std::string outputfn(argv[argc - 4]);
    std::vector<std::string> inputlist = std::vector<std::string>(argc - 5);
    for (int i = 1; i < argc - 4; ++i)
    {
        inputlist.at(i-1) = std::string(argv[i]);
    }
    std::string dir;
    dir = rootdir + "/" + inputlist.at(0)  + "/" + iter;
    std::vector<std::string> layerlist = make_layerList(networkfn, dir);
    std::vector<std::vector<std::vector<float>>> weights(inputlist.size());
    std::vector<std::unordered_map<std::string, int>> dicts(inputlist.size());
    int i = 0;
    for (std::string dirname : inputlist)
    {
        dir = rootdir + "/" + dirname + "/" + iter;
        load_weights(layerlist, weights.at(i), dir);
        load_dict(layerlist, dicts.at(i), dir);
        ++i;
    }

    std::vector<std::vector<float>> inputweights(inputlist.size());
    // int i = 0;
    i = 0;
    for (std::string l : layerlist) {
        for (int j = 0; j < inputlist.size(); ++j) {
            inputweights.at(j) = weights.at(j).at(i);
        }
        if (l != "output") {
            std::vector<float> average = average_weight(inputweights);
            for (std::string dirname : inputlist) {
                std::string dir = rootdir + "/" + dirname + "/" + iter;
                write_weights(l, average, dir);
            }
        }
        else {
            std::vector<std::vector<float>> averages = average_weight_output(inputweights, dicts);
            int j = 0;
            for (std::string dirname : inputlist) {
                std::string dir = rootdir + "/" + dirname + "/" + iter + "__";
                write_weights(l, averages[j], dir);
                // write dict
                std::string dictfn = rootdir + "/" + dirname + "/" + iter + "__/wdict.cereal";
                std::ifstream ifs(dictfn, std::ios::binary);
                cereal::BinaryInputArchive archive(ifs);
                archive(dicts[j]);
                ++j;
            }
        }
        ++i;
    }
}
