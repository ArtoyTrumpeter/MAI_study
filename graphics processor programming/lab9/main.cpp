#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include "mpi.h"

#define _i(i, j, k) (((k) + 1) * (ny + 2) * (nx + 2) + ((j) + 1) * (nx + 2) + (i) + 1)

#define _ib(i, j, k) ((k) * nby * nbx + (j) * nbx + (i))

double calculation(double* data, double* next, double hx, double hy, double hz, int nx, int ny, int nz) {
    int i = 0, j = 0, k = 0;
    double max_val = 0;
    #pragma omp parallel for shared(data, next, hx, hy, hz, nx, ny, nz) private(i, j, k) reduction(max:max_val)
    for (k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
                double xVal = (data[_i(i+1, j, k)] + data[_i(i-1, j, k)]) / (hx * hx);
                double yVal = (data[_i(i, j+1, k)] + data[_i(i, j-1, k)]) / (hy * hy);
                double zVal = (data[_i(i, j, k+1)] + data[_i(i, j, k-1)]) / (hz * hz);
                next[_i(i,j,k)] = 0.5 * (xVal + yVal + zVal) / (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
                max_val = std::max(max_val, fabs(next[_i(i,j,k)] - data[_i(i,j,k)]));
            }
        }
    }
    return max_val;
}

class Processor {
    int id, numproc, procNameLen;
    char procName[MPI_MAX_PROCESSOR_NAME];
    char file_path[256];

    int ib, jb, kb, i, j, k;

    int nbx, nby, nbz, nx, ny, nz;

    double lx, ly, lz, hx, hy, hz;
    double bc_down, bc_up, bc_left, bc_right, bc_front, bc_back;

    double* data;
    double* temp;
    double* next;
    double* maxVals;

    double eps, u0;

    int it = 0;

    MPI_Status status;

    MPI_Datatype X_BLOCK_2D;
    MPI_Datatype Y_BLOCK_2D;
    MPI_Datatype Z_BLOCK_2D;

    void copyBeginData() {
        MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nbx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nby, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nbz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bc_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bc_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(file_path, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    void readData() {
        if (id == 0) {
            std::cin >> nbx >> nby >> nbz
                >> nx >> ny >> nz
                >> file_path >> eps
                >> lx >> ly >> lz
                >> bc_down >> bc_up >> bc_left >> bc_right >> bc_front >> bc_back
                >> u0;
        }
    }

    void initData() {
        ib = id % nbx;
        jb = (id / nbx) % nby;
        kb = id / (nbx * nby);

        hx = lx / (nx * nbx);
        hy = ly / (ny * nby);
        hz = lz / (nz * nbz);

        int size = (nx + 2) * (ny + 2) * (nz + 2);
        data = new double[size];
        next = new double[size];
        maxVals  = new double[numproc];
        
        MPI_Datatype X_ROW;
        MPI_Datatype Y_ROW;

        MPI_Type_vector(nx, 1, 1, MPI_DOUBLE, &X_ROW);
        MPI_Type_vector(ny, 1, nx + 2, MPI_DOUBLE, &Y_ROW);
        MPI_Type_commit(&X_ROW);
        MPI_Type_commit(&Y_ROW);
        MPI_Type_create_hvector(nz, 1, (nx + 2) * (ny + 2) * sizeof(double), Y_ROW, &X_BLOCK_2D);
        MPI_Type_create_hvector(nz, 1, (nx + 2) * (ny + 2) * sizeof(double), X_ROW, &Y_BLOCK_2D);
        MPI_Type_create_hvector(ny, 1, (nx + 2) * sizeof(double), X_ROW, &Z_BLOCK_2D);
        MPI_Type_commit(&Y_BLOCK_2D);
        MPI_Type_commit(&X_BLOCK_2D);
        MPI_Type_commit(&Z_BLOCK_2D);
    }

    void exchange_X() {
        if (nbx <= 1) {
            return;
        }
        if (ib == 0) {
            MPI_Sendrecv(data + _i(nx-1, 0, 0), 1, X_BLOCK_2D, _ib(1,jb,kb), id,
                         data + _i(nx, 0, 0), 1, X_BLOCK_2D, _ib(1,jb,kb), _ib(1,jb,kb), MPI_COMM_WORLD, &status);
        } else if (ib < nbx-1) {
            MPI_Sendrecv(data + _i(nx-1, 0, 0), 1, X_BLOCK_2D, _ib(ib+1, jb, kb), id,
                         data + _i(-1, 0, 0), 1, X_BLOCK_2D, _ib(ib-1, jb, kb), _ib(ib-1, jb, kb), MPI_COMM_WORLD, &status);
            MPI_Sendrecv(data + _i(0, 0, 0), 1, X_BLOCK_2D, _ib(ib-1, jb, kb), id,
                         data + _i(nx, 0, 0), 1, X_BLOCK_2D, _ib(ib+1, jb, kb), _ib(ib+1, jb, kb), MPI_COMM_WORLD, &status);
        } else {
            MPI_Sendrecv(data + _i(0, 0, 0), 1, X_BLOCK_2D, _ib(ib - 1, jb, kb), id,
                         data + _i(-1, 0, 0), 1, X_BLOCK_2D, _ib(ib - 1, jb, kb), _ib(ib-1, jb, kb), MPI_COMM_WORLD, &status);
        }
    }

    void exchange_Y() {
        if (nby <= 1) {
            return;
        }
        if (jb == 0) {
            MPI_Sendrecv(data + _i(0, ny-1, 0), 1, Y_BLOCK_2D, _ib(ib,1,kb), id,
                         data + _i(0, ny, 0), 1, Y_BLOCK_2D, _ib(ib,1,kb), _ib(ib, 1, kb), MPI_COMM_WORLD, &status);
        } else if (jb < nby-1) {
            MPI_Sendrecv(data + _i(0, ny-1, 0), 1, Y_BLOCK_2D, _ib(ib, jb+1, kb), id,
                         data + _i(0, -1, 0), 1, Y_BLOCK_2D, _ib(ib, jb-1, kb), _ib(ib, jb-1, kb), MPI_COMM_WORLD, &status);
            MPI_Sendrecv(data + _i(0, 0, 0), 1, Y_BLOCK_2D, _ib(ib, jb-1, kb), id,
                         data + _i(0, ny, 0), 1, Y_BLOCK_2D, _ib(ib, jb+1, kb), _ib(ib, jb+1, kb), MPI_COMM_WORLD, &status);
        } else {
            MPI_Sendrecv(data + _i(0, 0, 0), 1, Y_BLOCK_2D, _ib(ib, jb-1, kb), id,
                         data + _i(0, -1, 0), 1, Y_BLOCK_2D, _ib(ib, jb-1, kb), _ib(ib, jb-1, kb), MPI_COMM_WORLD, &status);
        }
    }

    void exchange_Z() {
        if (nbz <= 1) {
            return;
        }
        if (kb == 0) {
            MPI_Sendrecv(data + _i(0, 0, nz-1), 1, Z_BLOCK_2D, _ib(ib,jb,kb+1), id,
                         data + _i(0, 0, nz), 1, Z_BLOCK_2D, _ib(ib,jb,kb+1), _ib(ib,jb,kb+1), MPI_COMM_WORLD, &status);
        } else if (kb < nbz-1) {
            MPI_Sendrecv(data + _i(0, 0, nz-1), 1, Z_BLOCK_2D, _ib(ib, jb, kb+1), id,
                         data + _i(0, 0, -1), 1, Z_BLOCK_2D, _ib(ib, jb, kb-1), _ib(ib, jb, kb-1), MPI_COMM_WORLD, &status);
            MPI_Sendrecv(data + _i(0, 0, 0), 1, Z_BLOCK_2D, _ib(ib, jb, kb-1), id,
                         data + _i(0, 0, nz), 1, Z_BLOCK_2D, _ib(ib, jb, kb+1), _ib(ib, jb, kb+1), MPI_COMM_WORLD, &status);
        } else {
            MPI_Sendrecv(data + _i(0, 0, 0), 1, Z_BLOCK_2D, _ib(ib, jb, kb-1), id,
                         data + _i(0, 0, -1), 1, Z_BLOCK_2D, _ib(ib, jb, kb-1), _ib(ib, jb, kb-1), MPI_COMM_WORLD, &status);
        }
    }

    void exchange_data() {
        exchange_X();
        exchange_Y();
        exchange_Z();
    }

    enum Bc {DOWN, UP, LEFT, RIGHT, FRONT, BACK};

    void bcInit(Bc bcType, int xn, int yn) {
        for (int y = 0; y < yn; y++) {
            for (int x = 0; x < xn; x++) {
                switch (bcType) {
                    case LEFT:
                        data[_i(-1, x, y)] = bc_left;
                        next[_i(-1, x, y)] = bc_left;
                        break;
                    case FRONT:
                        data[_i(x, -1, y)] = bc_front;
                        next[_i(x, -1, y)] = bc_front;
                        break;
                    case DOWN:
                        data[_i(x, y, -1)] = bc_down;
                        next[_i(x, y, -1)] = bc_down;
                        break;
                    case RIGHT:
                        data[_i(nx, x, y)] = bc_right;
                        next[_i(nx, x, y)] = bc_right;
                        break;
                    case BACK:
                        data[_i(x, ny, y)] = bc_back;
                        next[_i(x, ny, y)] = bc_back;
                        break;
                    case UP:
                        data[_i(x, y, nz)] = bc_up;
                        next[_i(x, y, nz)] = bc_up;
                        break;
                }
            }
        }
    }

    void mpi_print_answer() {
        int numSize = 14;
        int bufSize = nx * ny * nz * numSize;
        char* resultBuff = (char*)malloc(sizeof(char) * bufSize);
        memset(resultBuff, ' ', sizeof(char) * bufSize);
        for(int k = 0; k < nz; k++) {
            for(int j = 0; j < ny; j++) {
                for(int i = 0; i < nx; i++) {
                    sprintf(resultBuff + (i + j * nx + k * nx * ny) * numSize, "%.6e ", data[_i(i, j, k)]);
                }
            }
        }
    
        for (int i = 0; i < bufSize; i++) {
            if (resultBuff[i] == '\0') {
                resultBuff[i] = ' ';
            }
        }
    
        int procOffset = (nx * nbx * ny * nby * nz * kb + nx * nbx * ny * jb + nx * ib) * numSize;
    
        MPI_Datatype block2D,  filetype;
        MPI_Type_create_hvector(ny, nx * numSize, nbx * nx * numSize, MPI_CHAR, &block2D);
        MPI_Type_commit(&block2D);
        MPI_Type_create_hvector(nz, 1, nx * nbx * ny * nby * numSize, block2D, &filetype);
        MPI_Type_commit(&filetype);
        
    
        MPI_File_delete(file_path, MPI_INFO_NULL);
        MPI_File fp;
        MPI_File_open(MPI_COMM_WORLD, file_path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
        MPI_File_set_view(fp, sizeof(char) * procOffset, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
        
        MPI_File_write_all(fp, resultBuff, bufSize, MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fp);
        free(resultBuff);
    }

public:
    void process() {
        MPI_Comm_size(MPI_COMM_WORLD, &numproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
        MPI_Get_processor_name(procName, &procNameLen);
        MPI_Barrier(MPI_COMM_WORLD);

        readData();
        copyBeginData();
        initData();

        for (k = 0; k <= nz; ++k)
            for (j = 0; j <= ny; ++j)
                for (i = 0; i <= nx; ++i)
                    data[_i(i,j,k)] = u0;

        double maxVal = 0;

        if (ib == 0) bcInit(LEFT, ny, nz);
        if (jb == 0) bcInit(FRONT, nx, nz);
        if (kb == 0) bcInit(DOWN, nx, ny);
        if (ib+1 == nbx) bcInit(RIGHT, ny, nz);
        if (jb+1 == nby) bcInit(BACK, nx, nz);
        if (kb+1 == nbz) bcInit(UP, nx, ny);

        do {
            MPI_Barrier(MPI_COMM_WORLD);
            exchange_data();
            MPI_Barrier(MPI_COMM_WORLD);

            maxVal = calculation(data, next, hx, hy, hz, nx, ny, nz);

            MPI_Allgather(&maxVal, 1, MPI_DOUBLE, maxVals, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            for (i = 0; i < numproc; i++) {
                maxVal = std::max(maxVal, maxVals[i]);
            }
            temp = next;
            next = data;
            data = temp;
            it++;
        } while (maxVal >= eps);
        mpi_print_answer();
    }

    ~Processor() {
        delete[] data;
        delete[] next;
        delete[] maxVals;
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Processor proc;
    proc.process();
    MPI_Finalize();
    return 0;
}