#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <random>
#include <fmt/core.h>
#include <cmath>
#include <chrono>
namespace ch=std::chrono;

#define GRUPOS 10

std::vector<int> read_file() {
    std::fstream fs("C:/Users/josue/Desktop/Universidad/ProgParalela/Grupal_final/numeros/datos2.txt", std::ios::in);
    std::string line;
    std::vector<int> ret;
    while (std::getline(fs, line)) {
        ret.push_back(std::stoi(line));
    }
    fs.close();
    return ret;
}

__global__ void histograma(int *histo, int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(&histo[data[idx]], 1);
    }
}

int main() {

    std::vector<int> data = read_file();
    int n = data.size();
    int block = std::ceil((double)255 / GRUPOS);

    std::vector<int> histo(256);
    std::vector<int> final(GRUPOS);

    // Definir punteros para la GPU
    int *d_histo;
    int *d_data;

    cudaMalloc((void**)&d_data, n * sizeof(int));
    cudaMalloc((void**)&d_histo, 256 * sizeof(int));

    // Copiar datos desde la CPU a la GPU
    auto start = ch::high_resolution_clock::now();
    cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Llamar al kernel
    histograma<<<(n + 255) / 256, 256>>>(d_histo, d_data, n);

    // Copiar el histograma resultante de la GPU a la CPU
    cudaMemcpy(histo.data(), d_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria en la GPU
    cudaFree(d_data);
    cudaFree(d_histo);

    int indice= 0;
    for (int i = 0; i < GRUPOS; ++i) {
        int suma = 0;
        for (int j = 0; j < block; ++j) {
            if (indice<256){
                suma += histo[indice++];
            }
        }
        final[i] = suma;
        fmt::println("[{}], [{}]", i, suma);
    }

    auto end = std::chrono::high_resolution_clock::now();
    ch::duration<double, std::milli> tiempo = end - start;
    fmt::println("Tiempo CUDA: {}ms", tiempo.count());

    return 0;
}