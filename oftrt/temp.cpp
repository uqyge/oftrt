#include "csv.h"
#include "iostream"

#include "fstream"

#include "boost/tokenizer.hpp"

void readcsv_method_boost()
{
    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    std::string s = "boost c++ libraries";
    boost::char_separator<char> sep{" "};
    tokenizer tok(s, sep);
    for (const auto &t : tok)
        std::cout << t << '\n';
}

void readcsv_method_1()
{
    io::CSVReader<2> in("iris.csv");
    in.read_header(io::ignore_extra_column, "PetalWidth", "Name");
    std::string Name;
    float PetalWidth;
    std::vector<std::string> Names;
    while (in.read_row(PetalWidth, Name))
    {
        // std::cout << Name << '\n';
        Names.emplace_back(Name);
    }
    std::cout << "There are " << Names.size() << " samples. The 6th is " << Names[5] << '\n';
}

void readcsv_method_2()
{
    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    boost::char_separator<char> sep{","};
    std::vector<std::string> in;
    std::ifstream myFile;

    myFile.open("iris.csv");
    int i = 0;
    std::vector<std::string> vec;
    std::vector<std::vector<std::string>> vec2;
    std::string line;
    std::getline(myFile, line);
    while (myFile.good())
    {
        std::getline(myFile, line);
        tokenizer tok(line, sep);
        vec.assign(tok.begin(), tok.end());
        std::cout << vec[4] << '\n';
        vec2.push_back(vec);
    }
    std::cout << "in size is " << vec2.size() << '\n';
    std::cout << "the fifth on the first row is " << vec2[0][4] << '\n';
}

int main()
{
    // readcsv_method_1();
    readcsv_method_2();
    // readcsv_method_boost();
}