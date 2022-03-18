#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <algorithm>
#include <cmath>
#include "mpi.h"

#define _i(i, j, k) (((k) + 1) * (nx + 2) * (ny + 2) + ((j) + 1) * (nx + 2) + (i) + 1)

#define _ib(i, j, k) ((k) * nbx * nby + (j) * nbx + (i))

using namespace std;

class Processor {
    int id, numproc, proc_name_len;
	char proc_name[MPI_MAX_PROCESSOR_NAME];

    int nbx, nby, nbz, nx, ny, nz;
	int ib, jb, kb, i, j, k;
	
	string FileName;
    double eps;
    double u0;
    double diff;

    int size;

	double lx, ly, lz, hx, hy, hz, bcDown, bcUp, bcLeft, bcRight, bcBack, bcFront;
	double *data, *temp, *next, *buff_send, *buff_rcv;

	MPI_Status status;

    void scan_data() {
        if (id == 0) {		
            cin >> nbx >> nby >> nbz
                >> nx >> ny >> nz
                >> FileName >> eps
                >> lx >> ly >> lz
                >> bcFront >> bcBack >> bcLeft >> bcRight >> bcDown >> bcUp
                >> u0;
        }
    }

    void mpi_bcast() {
        MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nbx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nby, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nbz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcDown, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcUp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcLeft, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcRight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcBack, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcFront, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void init_indices() {
        ib = id % nbx;
        jb = (id / nbx) % nby;
        kb = id / (nbx * nby);
    
        hx = lx / (double)(nx * nbx);	
        hy = ly / (double)(ny * nby);
        hz = lz / (double)(nz * nbz);
        
        size = (nx + 2) * (ny + 2) * (nz + 2);
        data = (double *)malloc(sizeof(double) * size);	
        next = (double *)malloc(sizeof(double) * size);
    
        int n_max1 = std::max(nx, ny);
        int n_max2 = std::max(nz, nx + ny - n_max1);
    
        buff_send = (double *)malloc(sizeof(double) * (n_max1 + 2) * (n_max2 + 2));
        buff_rcv = (double *)malloc(sizeof(double) * (n_max1 + 2) * (n_max2 + 2));
    }

    void exchangeX() {
        for (int dir = 0; dir < 2; dir++) {
            if ((ib + dir) & 1) {
                if (ib > 0) {
                    for (j = 0; j < ny; j++) {
                        for (k = 0; k < nz; k++) {
                            buff_send[j * nz + k] = data[_i(0, j, k)];
                        }
                    }
                    MPI_Sendrecv(buff_send, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), id,
                                buff_rcv, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
                    for (j = 0; j < ny; j++) {
                        for (k = 0; k < nz; k++) {
                            data[_i(-1, j, k)] = buff_rcv[j * nz + k];
                        }
                    }
                } else {
                    for (j = 0; j < ny; j++) {
                        for (k = 0; k < nz; k++) {
                            data[_i(-1, j, k)] = bcLeft;
                        }
                    }
                }
            } else {
                if (ib < nbx - 1) {
                    for (j = 0; j < ny; j++) {
                        for (k = 0; k < nz; k++) {
                            buff_send[j * nz + k] = data[_i(nx - 1, j, k)];
                        }
                    }
                    MPI_Sendrecv(buff_send, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), id,
                                buff_rcv, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
                    for (j = 0; j < ny; j++) {
                        for (k = 0; k < nz; k++) {
                            data[_i(nx, j, k)] = buff_rcv[j * nz + k];
                        }
                    }
                } else {
                    for (j = 0; j < ny; j++) {
                        for (k = 0; k < nz; k++) {
                            data[_i(nx, j, k)] = bcRight;
                        }
                    }
                }
            }
        }
    }

    void exchangeY() {
        for (int dir = 0; dir < 2; dir++) {
            if ((jb + dir) & 1) {
                if (jb > 0) {
                    for (i = 0; i < nx; i++) {
                        for (k = 0; k < nz; k++) {
                            buff_send[i * nz + k] = data[_i(i, 0, k)];
                        }
                    }
                    MPI_Sendrecv(buff_send, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), id,
                                buff_rcv, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
                    for (i = 0; i < nx; i++) {
                        for (k = 0; k < nz; k++) {
                            data[_i(i, -1, k)] = buff_rcv[i * nz + k];
                        }
                    }
                } else {
                    for (i = 0; i < nx; i++) {
                        for (k = 0; k < nz; k++) {
                            data[_i(i, -1, k)] = bcDown;
                        }
                    }
                }
            } else {
                if (jb < nby - 1) {
                    for (i = 0; i < nx; i++) {
                        for (k = 0; k < nz; k++) {
                            buff_send[i * nz + k] = data[_i(i, ny - 1, k)];
                        }
                    }
                    MPI_Sendrecv(buff_send, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), id,
                                buff_rcv, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
                    for (i = 0; i < nx; i++) {
                        for (k = 0; k < nz; k++) {
                            data[_i(i, ny, k)] = buff_rcv[i * nz + k];
                        }
                    }
                } else {
                    for (i = 0; i < nx; i++) {
                        for (k = 0; k < nz; k++) {
                            data[_i(i, ny, k)] = bcUp;
                        }
                    }
                }
            }
        }
    }

    void exchangeZ() {
        for (int dir = 0; dir < 2; dir++) {
            if ((kb + dir) & 1) {
                if (kb > 0) {
                    for (i = 0; i < nx; i++) {
                        for (j = 0; j < ny; j++) {
                            buff_send[i * ny + j] = data[_i(i, j, 0)];
                        }
                    }
                    MPI_Sendrecv(buff_send, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), id,
                                buff_rcv, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
                    for (i = 0; i < nx; i++) {
                        for (j = 0; j < ny; j++) {
                            data[_i(i, j, -1)] = buff_rcv[i * ny + j];
                        }
                    }
                } else {
                    for (i = 0; i < nx; i++) {
                        for (j = 0; j < ny; j++) {
                            data[_i(i, j, -1)] = bcFront;
                        }
                    }
                }
            } else {
                if (kb < nbz - 1) {
                    for (i = 0; i < nx; i++) {
                        for (j = 0; j < ny; j++) {
                            buff_send[i * ny + j] = data[_i(i, j, nz - 1)];
                        }
                    }
                    MPI_Sendrecv(buff_send, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), id,
                                buff_rcv, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
                    for (i = 0; i < nx; i++) {
                        for (j = 0; j < ny; j++) {
                            data[_i(i, j, nz)] = buff_rcv[i * ny + j];
                        }
                    }
                } else {
                    for (i = 0; i < nx; i++) {
                        for (j = 0; j < ny; j++) {
                            data[_i(i, j, nz)] = bcBack;
                        }
                    }
                }
            }
        }
    }

    void exchange_data() {
        exchangeX();
        exchangeY();
        exchangeZ();
    }

    void print_answer() {
        if (id != 0) {
            for (k = 0; k < nz; k++) {
                for (j = 0; j < ny; j++) {
                    for (i = 0; i < nx; i++) {
                        buff_send[i] = data[_i(i, j, k)];
                    }
                    MPI_Send(buff_send, nx, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
                }
            }
        } else {
            FILE *output_file = fopen(FileName.c_str(), "w");
            for (kb = 0; kb < nbz; kb++) {
                for (k = 0; k < nz; k++) {
                    for (jb = 0; jb < nby; jb++) {
                        for (j = 0; j < ny; j++) {
                            for (ib = 0; ib < nbx; ib++) {
                                if (_ib(ib, jb, kb) == 0) {
                                    for (i = 0; i < nx; i++) {
                                        buff_rcv[i] = data[_i(i, j, k)];
                                    }
                                } else {
                                    MPI_Recv(buff_rcv, nx, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, &status);
                                }
                                for (i = 0; i < nx; i++) {
                                    fprintf(output_file, "%.6e ", buff_rcv[i]);
                                }
                                if (ib + 1 == nbx) {
                                    fprintf(output_file, "\n");
                                }
                            }
                        }
                    }
                }
            }
            fclose(output_file);
        }
    }

public:
    void process() {
        MPI_Comm_size(MPI_COMM_WORLD, &numproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
        MPI_Get_processor_name(proc_name, &proc_name_len);	
        MPI_Barrier(MPI_COMM_WORLD);

        scan_data();
        mpi_bcast();
        init_indices();

        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                for (k = 0; k < nz; k++) {
                    data[_i(i, j, k)] = u0;
                }
            }
        }

        diff = eps + 1;
        
        while (diff >= eps) {
            MPI_Barrier(MPI_COMM_WORLD);
            exchange_data();
            MPI_Barrier(MPI_COMM_WORLD);
    
            diff = 0;
    
            for(i = 0; i < nx; i++)
                for(j = 0; j < ny; j++)
                    for(k = 0; k < nz; k++) {
                        next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
                                                (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
                                                (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) / 
                                                    (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
                        diff = fmax(diff, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
                    }
        
            temp = next;
            next = data;
            data = temp;

            MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }
        print_answer();
    }
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Processor proc;
    proc.process();
    MPI_Finalize();
    return 0;
}