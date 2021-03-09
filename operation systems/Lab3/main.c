#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_THREAD_NUM 2047

void swap(int *lhs, int *rhs) {
	int tmp = *lhs;
	*lhs = *rhs;
	*rhs = tmp;
}			

int partition(int *arr, int l, int r) {
	int p = arr[(l + r) / 2];
	int i = l;
	int j = r;
	while (i <= j) {
		while (arr[i] < p)
			++i;
		while (arr[j] > p)
			--j;
		if (i >= j)
			break;
		swap(&arr[i++], &arr[j--]);
	}
	return j;
}

void quick_sort(int *arr, int l, int r) {
	if (l >= r)
		return ;
	int j = partition(arr, l, r);
	quick_sort(arr, l, j);
	quick_sort(arr, j + 1, r);
}

int max(int a, int b) {
	return (a > b) ? a : b;
}

int min(int a, int b) {
	return (a < b) ? a : b;
}	

typedef struct {
	int win_h;
	int win_w;
	int mrx_h;
	int mrx_w;
	int **i_mrx; 
	int **o_mrx;
} io_mrx_t;

typedef struct {
	io_mrx_t* io_mrx;
	int i;
	int j;
} thr_data_t;

int** mrx_create(int h, int w) {
	int **arr = (int**)malloc(sizeof(int*) * h);
	for (int i = 0; i < h; ++i) {
		arr[i]  = (int *)malloc(sizeof(int) * w);
	}
	return arr;
}

void mrx_destroy(int **arr, int h) {
	for (int i = 0; i < h; ++i) {
		free(arr[i]);
		arr[i] = NULL;
	}
	free(arr);
}

void mrx_input(int **arr, int h, int w) {
	for (int i = 0; i < h; ++i)
		for (int j = 0; j < w; ++j)
			scanf("%d", &arr[i][j]);
}

int window_is_valid(int h, int w) {
	if (h < 1 || w < 1)
		return 0;
	return 1;
}

io_mrx_t* io_mrx_create(void) {
	io_mrx_t* io_mrx = (io_mrx_t*)malloc(sizeof(io_mrx_t));
	scanf("%d%d%d%d", &io_mrx->win_h,
		&io_mrx->win_w, &io_mrx->mrx_h, &io_mrx->mrx_w);
	if (!window_is_valid(io_mrx->win_h, io_mrx->win_w)) {
		printf("Window is not valid\n");
		exit(0);
	}
	io_mrx->i_mrx = mrx_create(io_mrx->mrx_h, io_mrx->mrx_w);
	io_mrx->o_mrx = mrx_create(io_mrx->mrx_h, io_mrx->mrx_w);
	mrx_input(io_mrx->i_mrx, io_mrx->mrx_h, io_mrx->mrx_w);
	return io_mrx;
}

void io_mrx_destroy(io_mrx_t* io_mrx) {
	mrx_destroy(io_mrx->i_mrx, io_mrx->mrx_h);
	mrx_destroy(io_mrx->o_mrx, io_mrx->mrx_h);
	io_mrx->i_mrx = NULL;
	io_mrx->o_mrx = NULL;
	free(io_mrx);
}

void io_mrx_print(io_mrx_t* io_mrx) {
	printf("window width: %d\nwindow height: %d\nwidth: %d\nheight: %d\n",
	io_mrx->win_w, io_mrx->win_h, io_mrx->mrx_w, io_mrx->mrx_h);
	printf("\nResult matrix:\n");
	for (int i = 0; i < io_mrx->mrx_h; ++i) {
		for (int j = 0; j < io_mrx->mrx_w; ++j) {
			printf("%5d", io_mrx->o_mrx[i][j]);
			printf(j == io_mrx->mrx_w - 1 ? "\n" : " ");
		}
	}
}

thr_data_t* thr_data_arr_create(io_mrx_t* io_mrx) {
	int n = io_mrx->mrx_h * io_mrx->mrx_w;
	thr_data_t* arr = (thr_data_t*)malloc(sizeof(thr_data_t) * n);
	for (int i = 0; i < n; ++i) {
		arr[i].io_mrx = io_mrx;
		arr[i].i = i / io_mrx->mrx_h;
		arr[i].j = i % io_mrx->mrx_w;
	}
	return arr;
}

void thr_data_arr_destroy(thr_data_t* arr) {
	free(arr);
}

int arithm_mean(int *arr, int size) {
	int res = 0;
	for (int i = 0; i < size; ++i)
		res += arr[i];
	return res / size;
}

int median_val(int *arr, int size) {
	quick_sort(arr, 0, size - 1);
	return (size % 2 == 0) ? (arr[size / 2 - 1] + arr[size / 2]) / 2 : arr[size / 2];
}

void item_filter(thr_data_t* item) {
	int i = max(0, item->i - item->io_mrx->win_h / 2);
	int j = max(0, item->j - item->io_mrx->win_w / 2);
	int end_i = min(item->io_mrx->mrx_h,
		item->i + item->io_mrx->win_h / 2 + 1);
	int end_j = min(item->io_mrx->mrx_w,
		item->j + item->io_mrx->win_w / 2 + 1);
	int count = 0;
	int size = item->io_mrx->win_h * item->io_mrx->win_w;
	int *win_arr = malloc(sizeof(int) * size);
	for (; i < end_i; ++i)
		for (int p = j; p < end_j; ++p)
			win_arr[count++] = item->io_mrx->i_mrx[i][j];
	if (count != size) {
		int ar_mean = arithm_mean(win_arr, count);
		for (int i = count; i < size; ++i)
			win_arr[i] = ar_mean;
	}
	item->io_mrx->o_mrx[item->i][item->j] = median_val(win_arr, size);
	free(win_arr);
}

void* routine(void *arg) {
	thr_data_t* thr_data = (thr_data_t *)arg;
	item_filter(thr_data);
	return NULL;
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Using ./filter num_threads\n");
		return 0;
	}
	int n = atoi(argv[1]);
	int k;
	int p;
	scanf("%d", &k);
	io_mrx_t* io_mrx = io_mrx_create();
	thr_data_t* thr_data_arr = thr_data_arr_create(io_mrx);
	int size = io_mrx->mrx_h * io_mrx->mrx_w;
	n = min(min(max(n, 1), MAX_THREAD_NUM), size);
	pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * n);
	p = k;
	while (k > 0) {
		int i = 0;
		while (i < size) {
			int t = min(size - i, n);
			for (int j = 0; j < t; ++j) {
				pthread_create(&threads[j], NULL, routine, &thr_data_arr[i]);
				++i;
			}
			for (int j = 0; j < t; ++j)
				pthread_join(threads[j], NULL);
		}
		for (int l = 0; l < io_mrx->mrx_h; ++l)
			for (int j = 0; j < io_mrx->mrx_w; ++j)
				io_mrx->i_mrx[l][j] = io_mrx->o_mrx[l][j];
		--k;
	}
	printf("Number of threads: %d\n", n);
	printf("Number of filter: %d\n", p);
	io_mrx_print(io_mrx);
	free(threads);
	thr_data_arr_destroy(thr_data_arr);
	io_mrx_destroy(io_mrx);
}