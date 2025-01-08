//
// Created by lenovo on 25-1-7.
//

#ifndef READDATA_H
#define READDATA_H

#include <vector>
#include <string>

namespace readFile {
    class readData {
    public:
        static std::vector<std::vector<double> > readImageData(std::string path);

        static std::vector<int> readTagData(std::string path);
    };
}


#endif //READDATA_H
