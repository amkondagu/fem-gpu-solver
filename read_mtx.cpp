#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
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

COO read_mtx(const std::string& matrix) {

    std::ifstream f(matrix);

    if (!f.is_open()) {
        throw std::runtime_error("Error opening the file");
    }

    COO coo; 
    std::string s;

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

int main(void) {
    try {
        COO coo = read_mtx("matrices/generated_symmetric_matrices/elas2d_n32_lambda10p0.mtx");
        CSR csr = convert_COO_CSR(coo);

        std::cout << "Rows: " << coo.numRows << "\n";
        std::cout << "Cols: " << coo.numCols << "\n";
        std::cout << "Entries read: " << coo.vals.size() << "\n";

        std::cout << "\nFirst few entries:\n";

        for (int i=0; i < 5; i++) {
            std::cout << "(" << coo.rows[i] << ", " << coo.cols[i] << ") = " << coo.vals[i] << "\n";
        }

        for (int i=0; i < 5; i++) {
            std::cout << "Row " << i << ": ";

            for (int j=csr.rowIdx[i]; j < csr.rowIdx[i+1]; j++) {
                std::cout << "(" << csr.colIdx[j] << ", " << csr.vals[j] << ") ";
            }

            std::cout << "\n";
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}