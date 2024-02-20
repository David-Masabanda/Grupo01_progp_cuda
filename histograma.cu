#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <random>
#include <fmt/core.h>
#include <chrono>
namespace ch=std::chrono;

#define NCLASES 256
#define BS 256

std::vector<int> read_file() {
    std::fstream fs("C:/Users/josue/Desktop/Universidad/ProgParalela/Grupal_final/datos4.txt", std::ios::in);
    std::string line;
    std::vector<int> ret;
    while (std::getline(fs, line)) {
        ret.push_back(std::stoi(line));
    }
    fs.close();
    return ret;
}

__global__ void histo_k(int *histo, int *data, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        atomicAdd(&histo[data[i]], 1);
}

int main() {
    // Generar datos aleatorios
    std::vector<int> data = read_file();
    int n = data.size();

    // Inicializar arreglo para el histograma
    int histo[NCLASES] = {0};

    // Definir punteros para la GPU
    int *d_histo;
    int *d_data;
    cudaMalloc((void**)&d_data, n * sizeof(int));
    cudaMalloc((void**)&d_histo, NCLASES * sizeof(int));

    // Copiar datos desde la CPU a la GPU
    auto start = ch::high_resolution_clock::now();

    cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Llamar al kernel
    histo_k<<<(n + BS - 1) / BS, BS>>>(d_histo, d_data, n);

    // Copiar el histograma resultante de la GPU a la CPU
    cudaMemcpy(histo, d_histo, NCLASES * sizeof(int), cudaMemcpyDeviceToHost);


    auto end = std::chrono::high_resolution_clock::now();
    ch::duration<double, std::milli> tiempo = end-start;
    fmt::println("Tiempo CUDA: {}ms", tiempo.count());

    // Liberar memoria en la GPU
    cudaFree(d_data);
    cudaFree(d_histo);

    // Imprimir el histograma resultante
    for (int i = 0; i < NCLASES; ++i) {
        std::cout << "histo[" << i << "]: " << histo[i] << std::endl;
    }

    return 0;
}