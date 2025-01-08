//
// Created by lenovo on 25-1-7.
//

#include "readData.h"
#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;

namespace readFile {
    unsigned int reverseint(unsigned int A) {
        return ((((uint32_t) (A) & 0xff000000) >> 24) |
                (((uint32_t) (A) & 0x00ff0000) >> 8) |
                (((uint32_t) (A) & 0x0000ff00) << 8) |
                (((uint32_t) (A) & 0x000000ff) << 24));
    }

    unsigned char reverse(unsigned char A) {
        return ((((A) & (unsigned char) 0xff00) >> 8) |
                (((A) & (unsigned char) 0x00ff) << 8));
    }

    vector<vector<double> > readData::readImageData(std::string path) {
        ifstream inFile(path, ios::in | ios::binary);
        if (!inFile) {
            cout << "error" << endl;
            return vector<vector<double> >();
        }

        unsigned int n, a;

        int size;
        int height;
        int width;

        inFile.read((char *) &a, sizeof(unsigned int));
        cout << "magic number: " << reverseint(a) << endl;

        inFile.read((char *) &n, sizeof(unsigned int));
        size = reverseint(n);
        cout << "number of images: " << reverseint(n) << endl;

        inFile.read((char *) &a, sizeof(unsigned int));
        height = reverseint(a);
        cout << "number of rows: " << reverseint(a) << endl;
        inFile.read((char *) &a, sizeof(unsigned int));
        width = reverseint(a);
        cout << "number of colums" << reverseint(a) << endl;

        vector<vector<double> > imageList(size, vector<double>(height * width));

        for (int k = 0; k < size; k++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    unsigned char temp;
                    inFile.read((char *) &temp, sizeof(unsigned char));
                    imageList[k][i * width + j] = ((double) temp / 255);
                }
            }
        }

        return imageList;
    }

    static std::vector<int> readTagData(std::string path) {
        ifstream inFileLable(path, ios::in | ios::binary);
        if (!inFileLable) {
            cout << "error" << endl;
            return vector<int>();
        }

        int size;
        unsigned int n, a;
        inFileLable.read((char *) &a, sizeof(unsigned int));
        cout << "magic number: " << reverseint(a) << endl;

        inFileLable.read((char *) &n, sizeof(unsigned int));
        cout << "number of tags: " << reverseint(n) << endl;
        size = reverseint(n);

        vector<int> tagList(size);

        for (int k = 0; k < size; k++) {
            unsigned char temp;
            inFileLable.read((char *) &temp, sizeof(unsigned char));
            tagList[k] = temp;
        }

        return tagList;
    }
}
