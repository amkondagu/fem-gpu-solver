#ifndef READ_MTX_H
#define READ_MTX_H

#include <string>
#include <vector>

struct COO {
    int numRows; 
    int numCols;
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
};

struct CSR {
    int numRows; 
    int numCols;
    std::vector<int> rowIdx;
    std::vector<int> colIdx;
    std::vector<double> vals;
};

COO read_mtx(const std::string& matrix);
CSR convert_COO_CSR(const COO& coo);

#endif