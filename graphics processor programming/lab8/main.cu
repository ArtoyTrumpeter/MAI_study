#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include "mpi.h"

#define CSC(call)                                       \
do {                                                    \
  cudaError_t res = call;                               \
  if (res != cudaSuccess) {                             \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",    \
        __FILE__, __LINE__, cudaGetErrorString(res));   \
    exit(0);                                            \
  }                                                     \
} while(0);

#define _i(i, j, k) (((k) + 1) * (ny + 2) * (nx + 2) + ((j) + 1) * (nx + 2) + (i) + 1)

#define _ib(i, j, k) ((k) * nby * nbx + (j) * nbx + (i))

const int FILENAME_SIZE = 256;

struct Comparator {  
    __device__ __host__
    bool operator ()(double lhs, double rhs){
        return abs(lhs) < abs(rhs);
    }
};

__global__ void calculation(double* data, double* next, double hx, double hy, double hz, int nx, int ny, int nz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	int size = nx * ny * nz;

	int i, j, k;

	for(int l = idx; l < size; l += offset)	{
		i = l % nx;
		j = (l / nx ) % ny;
		k = l / (nx * ny);
		next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
											(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
											(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) / 
												(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
	}
}

__global__ void calculate_diff(double *dev_data, double *dev_next, int nx, int ny, int nz, int nbx, int nby) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetz = blockDim.z * gridDim.z;

    for (int k = idz - 1; k <= nz; k += offsetz) {
        for (int j = idy - 1; j <= ny; j += offsety) {
            for (int i = idx - 1; i <= nx; i += offsetx) {
                if (i == -1 || j == -1 || k == -1 || i == nx || j == ny || k == nz) {
                    dev_data[_i(i,j,k)] = 0.0;
                } else {
                    dev_data[_i(i,j,k)] = abs(dev_next[_i(i,j,k)] - dev_data[_i(i,j,k)]);
                }
            }
        }
    }
}

__global__ void extractXPart(double* dev_data, double* dev_buf, int nx, int ny, int nz, int nbx, int nby, int bufLen, int i) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int offsety = blockDim.x * gridDim.x;
    int offsetz = blockDim.y * gridDim.y;

    for(int k = idz; k < nz; k += offsetz) {
        for(int j = idy; j < ny; j += offsety) {
            dev_buf[bufLen*k + j] = dev_data[_i(i,j,k)];
        }
    }
}

__global__ void extractYPart(double* dev_data, double* dev_buf, int nx, int ny, int nz, int nbx, int nby, int bufLen, int j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsetz = blockDim.y * gridDim.y;

    for(int k = idz; k < nz; k += offsetz) {
        for(int i = idx; i < nx; i += offsetx) {
            dev_buf[bufLen*k + i] = dev_data[_i(i,j,k)];
        }
    }
}

__global__ void extractZPart(double* dev_data, double* dev_buf, int nx, int ny, int nz, int nbx, int nby, int bufLen, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < ny; j += offsety) {
        for (int i = idx; i < nx; i += offsetx) {
            dev_buf[bufLen*j + i] = dev_data[_i(i,j,k)];
        }
    }
}


__global__ void putXPart(double* dev_data, double* dev_buf, int nx, int ny, int nz, int nbx, int nby, int bufLen, int i) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int offsety = blockDim.x * gridDim.x;
    int offsetz = blockDim.y * gridDim.y;

    for(int k = idz; k < nz; k += offsetz) {
        for(int j = idy; j < ny; j += offsety) {
            dev_data[_i(i,j,k)] = dev_buf[bufLen*k + j];
        }
    }
}

__global__ void putYPart(double* dev_data, double* dev_buf, int nx, int ny, int nz, int nbx, int nby, int bufLen, int j) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idz = blockIdx.y*blockDim.y + threadIdx.y;
    int offsetx = blockDim.x*gridDim.x;
    int offsetz = blockDim.y*gridDim.y;

    for(int k = idz; k < nz; k += offsetz) {
        for(int i = idx; i < nx; i += offsetx) {
            dev_data[_i(i,j,k)] = dev_buf[bufLen*k + i];
        }
    }
}

__global__ void putZPart(double* dev_data, double* dev_buf, int nx, int ny, int nz, int nbx, int nby, int bufLen, int k) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int offsetx = blockDim.x*gridDim.x;
    int offsety = blockDim.y*gridDim.y;

    for(int j = idy; j < ny; j += offsety) {
        for(int i = idx; i < nx; i += offsetx) { 
            dev_data[_i(i,j,k)] = dev_buf[bufLen*j + i];
        } 
    }
}

class Processor {
    int id, numproc, procNameLen;
    char procName[MPI_MAX_PROCESSOR_NAME];
    char file_path[FILENAME_SIZE];

    int ib, jb, kb, i, j, k;

    int nbx, nby, nbz, nx, ny, nz;

    double lx, ly, lz, hx, hy, hz;
    double bc_down, bc_up, bc_left, bc_right, bc_front, bc_back;

    double* data;
    double* temp;
    double* next;
    double* buff_rcv;
    double* buff_send;
    double* maxVals;
    double* dev_data;
    double* dev_next;
    double* dev_buf;

    int devCnt, bufLen, bufSize;

    double eps;
    double u0;

    int it = 0;

    MPI_Status status;

    void fillbuff_sendX(int idx) {
        extractXPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, idx);
        CSC(cudaGetLastError());
        CSC(cudaMemcpy(buff_send, dev_buf, bufSize * sizeof(double), cudaMemcpyDeviceToHost));
    }

    void fromBufRecvX(int idx) {
        CSC(cudaMemcpy(dev_buf, buff_rcv, bufSize * sizeof(double), cudaMemcpyHostToDevice));
        CSC(cudaGetLastError());
        putXPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, idx);
    }

    void fillbuff_sendY(int idx) {
        extractYPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, idx);
        CSC(cudaGetLastError());
        CSC(cudaMemcpy(buff_send, dev_buf, bufSize * sizeof(double), cudaMemcpyDeviceToHost));
    }

    void fromBufRecvY(int idx) {
        CSC(cudaMemcpy(dev_buf, buff_rcv, bufSize * sizeof(double), cudaMemcpyHostToDevice));
        CSC(cudaGetLastError());
        putYPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, idx);
    }

    void fillbuff_sendZ(int idx) {
        extractZPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, idx);
        CSC(cudaGetLastError());
        CSC(cudaMemcpy(buff_send, dev_buf, bufSize * sizeof(double), cudaMemcpyDeviceToHost));
    }

    void fromBufRecvZ(int idx) {
        CSC(cudaMemcpy(dev_buf, buff_rcv, bufSize * sizeof(double), cudaMemcpyHostToDevice));
        CSC(cudaGetLastError());
        putZPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, idx);
    }

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
        MPI_Bcast(file_path, FILENAME_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
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

        bufLen = max(nx, max(ny, nz));
        bufSize = bufLen*bufLen;

        int size = (nx + 2) * (ny + 2) * (nz + 2);

        data = (double*) malloc(sizeof(double) * size);
        next = (double*) malloc(sizeof(double) * size);
        buff_rcv = (double*) malloc(sizeof(double) * bufSize);
        buff_send = (double*) malloc(sizeof(double) * bufSize);
        maxVals  = (double*) malloc(sizeof(double) * numproc);

        for (int k = 0; k <= nz; ++k) {
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    data[_i(i,j,k)] = u0;
                }
            }
        }

        if (ib == 0) bcInit(LEFT, ny, nz);
        if (jb == 0) bcInit(FRONT, nx, nz);
        if (kb == 0) bcInit(DOWN, nx, ny);
        if (ib+1 == nbx) bcInit(RIGHT, ny, nz);
        if (jb+1 == nby) bcInit(BACK, nx, nz);
        if (kb+1 == nbz) bcInit(UP, nx, ny);

        CSC(cudaMalloc(&dev_data, size * sizeof(double) ));
        CSC(cudaMalloc(&dev_next, size * sizeof(double) ));
        CSC(cudaMalloc(&dev_buf, bufSize * sizeof(double)));
        CSC(cudaMemcpy(dev_data, data, size * sizeof(double), cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_next, next, size * sizeof(double), cudaMemcpyHostToDevice));
    }

    void exchangeEdgeData() {
        if (nbx > 1) {
            if (ib == 0) {
                fillbuff_sendX(nx - 1);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(1,jb,kb), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(1,jb,kb), _ib(1,jb,kb), MPI_COMM_WORLD, &status);
                fromBufRecvX(nx);
            } else if (ib < nbx-1) {
                fillbuff_sendX(nx - 1);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib+1, jb, kb), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib-1, jb, kb), _ib(ib-1, jb, kb), MPI_COMM_WORLD, &status);
                fromBufRecvX(-1);
                fillbuff_sendX(0);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib-1, jb, kb), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib+1, jb, kb), _ib(ib+1, jb, kb), MPI_COMM_WORLD, &status);
                fromBufRecvX(nx);
            } else {
                fillbuff_sendX(0);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib - 1, jb, kb), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib-1, jb, kb), MPI_COMM_WORLD, &status);
                fromBufRecvX(-1);
            }
        }
        extractXPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, -1);
        putXPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_next, dev_buf, nx, ny, nz, nbx, nby, bufLen, -1);
        extractXPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, nx);
        putXPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_next, dev_buf, nx, ny, nz, nbx, nby, bufLen, nx);

        if (nby > 1) {
            if (jb == 0) {
                fillbuff_sendY(ny - 1);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib,1,kb), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib,1,kb), _ib(ib, 1, kb), MPI_COMM_WORLD, &status);
                fromBufRecvY(ny);
            } else if (jb < nby-1) {
                fillbuff_sendY(ny - 1);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib, jb+1, kb), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib, jb-1, kb), _ib(ib, jb-1, kb), MPI_COMM_WORLD, &status);
                fromBufRecvY(-1);
                fillbuff_sendY(0);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib, jb-1, kb), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib, jb+1, kb), _ib(ib, jb+1, kb), MPI_COMM_WORLD, &status);
                fromBufRecvY(ny);
            } else {
                fillbuff_sendY(0);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib, jb-1, kb), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib, jb-1, kb), _ib(ib, jb-1, kb), MPI_COMM_WORLD, &status);
                fromBufRecvY(-1);
            }
        }
        extractYPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, -1);
        putYPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_next, dev_buf, nx, ny, nz, nbx, nby, bufLen, -1);
        extractYPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, ny);
        putYPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_next, dev_buf, nx, ny, nz, nbx, nby, bufLen, ny);

        if (nbz > 1) {
            if (kb == 0) {
                fillbuff_sendZ(nz - 1);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib,jb,kb+1), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib,jb,kb+1), _ib(ib,jb,kb+1), MPI_COMM_WORLD, &status);
                fromBufRecvZ(nz);
            } else if (kb < nbz-1) {
                fillbuff_sendZ(nz - 1);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib, jb, kb+1), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib, jb, kb-1), _ib(ib, jb, kb-1), MPI_COMM_WORLD, &status);
                fromBufRecvZ(-1);
                fillbuff_sendZ(0);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib, jb, kb-1), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib, jb, kb+1), _ib(ib, jb, kb+1), MPI_COMM_WORLD, &status);
                fromBufRecvZ(nz);
            } else {
                fillbuff_sendZ(0);
                MPI_Sendrecv(buff_send, bufSize, MPI_DOUBLE, _ib(ib, jb, kb-1), id,
                             buff_rcv, bufSize, MPI_DOUBLE, _ib(ib, jb, kb-1), _ib(ib, jb, kb-1), MPI_COMM_WORLD, &status);
                fromBufRecvZ(-1);
            }
        }
        extractZPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, -1);
        putZPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_next, dev_buf, nx, ny, nz, nbx, nby, bufLen, -1);
        extractZPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_data, dev_buf, nx, ny, nz, nbx, nby, bufLen, nz);
        putZPart<<<dim3(4, 4), dim3(4, 4)>>>(dev_next, dev_buf, nx, ny, nz, nbx, nby, bufLen, nz);
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
    }

    double getMaxVal() {
        calculate_diff<<<dim3(8, 8, 8), dim3(8, 8, 8)>>>(dev_data, dev_next, nx, ny, nz, nbx, nby);
        CSC(cudaGetLastError());
        double maxVal = 0;
        Comparator cmp;
        thrust::device_ptr<double> ptr = thrust::device_pointer_cast(dev_data);
        maxVal = *thrust::max_element(ptr, ptr + (nx+2)*(ny+2)*(nz+2), cmp);
        MPI_Allgather(&maxVal, 1, MPI_DOUBLE, maxVals, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        for (i = 0; i < numproc; i++) {
            maxVal = max(maxVal, maxVals[i]);
        } 
        return maxVal;
    }

public:
    void process() {
        MPI_Comm_size(MPI_COMM_WORLD, &numproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
        MPI_Get_processor_name(procName, &procNameLen);
        CSC(cudaGetDeviceCount(&devCnt));
        CSC(cudaSetDevice(id%devCnt));
        readData();
        copyBeginData();
        initData();

        double maxVal = 0;
        do {
            MPI_Barrier(MPI_COMM_WORLD);
            exchangeEdgeData();
            calculation<<<64, 64>>>(dev_data, dev_next, hx, hy, hz, nx, ny, nz);
            maxVal = getMaxVal();
            temp = dev_next;
            dev_next = dev_data;
            dev_data = temp;
            it++;
        } while (maxVal >= eps);
        MPI_Barrier(MPI_COMM_WORLD);
        CSC(cudaMemcpy(data, dev_data, (nx+2) * (ny+2) * (nz+2) * sizeof(double), cudaMemcpyDeviceToHost));
        mpi_print_answer();
    }

    ~Processor() {
        delete[] data;
        delete[] next;
        delete[] buff_send;
        delete[] buff_rcv;
        delete[] maxVals;
        CSC(cudaFree(dev_data));
        CSC(cudaFree(dev_next));
        CSC(cudaFree(dev_buf));
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Processor proc;
    proc.process();
    MPI_Finalize();
    return 0;
}