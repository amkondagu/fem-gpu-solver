#include "read_mtx.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

COO read_mtx(const std::string& matrix) {

    std::ifstream f(matrix);

    if (!f.is_open()) {
        throw std::runtime_error("Error opening the file");
    }

    COO coo; 
    std::string s;
    std::getline(f, s);

    std::istringstream header(s);
    std::string matrixMarket;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;

    header >> matrixMarket >> object >> format >> field >> symmetry;

    bool symmetric = (symmetry == "symmetric");

    while (std::getline(f, s)) {
        if (!s.empty() && s[0] == '%') {
            continue;
        }

        int row;
        int col;
        int nnz; 
        std::istringstream dim(s);
        dim >> row >> col >> nnz;
        coo.numRows = row;
        coo.numCols = col; 

        double val; 
        for (int i=0; i < nnz; i++) {
            f >> row >> col >> val;
            coo.rows.push_back(row-1);  //because c++ is 0 indexed
            coo.cols.push_back(col-1);
            coo.vals.push_back(val);

            // one triangle, so add the mirrored entry too.
            // Do not duplicate diagonal entries.
            if (symmetric && row != col) {
                coo.rows.push_back(col-1);
                coo.cols.push_back(row-1);
                coo.vals.push_back(val);
            }
        }

        break; 
    }
        
    f.close();

    return coo;
}

CSR convert_COO_CSR(const COO& coo) {
    CSR csr; 
    csr.numRows = coo.numRows;
    csr.numCols = coo.numCols; 

    std::vector<int> rowCount(csr.numRows, 0);
    for (int i=0; i < coo.rows.size(); i++) {
        rowCount[coo.rows[i]] += 1; 
    }

    csr.rowIdx.push_back(0);
    for (int i=0; i < csr.numRows; i++) {
        csr.rowIdx.push_back(rowCount[i] + csr.rowIdx[i]); 
    }

    std::vector<int> sortIdx = csr.rowIdx;
    csr.colIdx.resize(coo.rows.size());
    csr.vals.resize(coo.rows.size());

    for (int i=0; i < coo.rows.size(); i++) {
        int pos = sortIdx[coo.rows[i]];
        csr.colIdx[pos] = coo.cols[i];
        csr.vals[pos] = coo.vals[i];
        sortIdx[coo.rows[i]] += 1;
    } 
    
    return csr; 
}