#include "csv.h"
#include "iostream"
int main()
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
