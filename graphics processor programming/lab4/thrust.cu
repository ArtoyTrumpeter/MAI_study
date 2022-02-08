#include <stdio.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

// Задача: найти максимум в массиве объектов по ключу

struct type {		// Тип элемента массива. Структура из двух полей 
	int key;
	int value;
}; 

struct comparator {												
	__host__ __device__ bool operator()(type a, type b) {		// Функция которая сравнивает объекты на "<"
		return a.key < b.key; 									// operator() - переопределение оператора "()" для экземпляра этой структуры
	}
};

int main() {
	srand(time(NULL));											// Инициализируем генератор случайных чисел в зависимости от времени 
	comparator comp;											
	int i, i_max = -1, n = 10;
	type *arr = (type *)malloc(sizeof(type) * n);
	for(i = 0; i < n; i++) {									// Здесь инициализируем массив и попутно ищем максимум на CPU
		arr[i].key = rand() % 10;
		arr[i].value = rand() % 10;
		if (i_max == -1 || comp(arr[i_max], arr[i]))			
			i_max = i;
		printf("%d ", arr[i].key);
	}
	type *dev_arr;
	cudaMalloc(&dev_arr, sizeof(type) * n);
	cudaMemcpy(dev_arr, arr, sizeof(type) * n, cudaMemcpyHostToDevice); 	// Копируем массив на GPU
	
	thrust::device_ptr<type> p_arr = thrust::device_pointer_cast(dev_arr);	// Трастовские функции принимают свой тип указателей, поэтому выполняем приведение типов.
	thrust::device_ptr<type> res = thrust::max_element(p_arr, p_arr + n, comp);	// Ищем максимум в массиве на GPU

	printf("cpu: %d\ngpu: %d\n", i_max, (int)(res - p_arr));		// Печатаем номер максимального элемента найденого на CPU и GPU
	cudaFree(dev_arr);
	free(arr);
	return 0;
}
