# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>NAME : KRISHNA PRASAD S</h3>
<h3>REGISTER NO : 212223230108</h3>
<h3>EX. NO : 3</h3>
<h3>DATE : 29/10/25</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## Program

```cpp
%%writefile sobel_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,
                            unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

        int sumX = 0, sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                unsigned char pixel = srcImage[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        int magnitude = sqrtf(float(sumX * sumX + sumY * sumY));
        magnitude = min(max(magnitude, 0), 255);
        dstImage[y * width + x] = (unsigned char)magnitude;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {

    Mat image = imread("creative2.jpg", IMREAD_COLOR);
    if (image.empty()) {
        printf("Error: Image not found at /content/image.jpg\n");
        return -1;
    }

    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    int width = grayImage.cols;
    int height = grayImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);

    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, grayImage.data, imageSize, cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 blockDim(16, 16);
    dim3 gridSize((width + blockDim.x - 1) / blockDim.x,
                  (height + blockDim.y - 1) / blockDim.y);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockDim>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cudaTime = 0;
    cudaEventElapsedTime(&cudaTime, start, stop);

    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("/content/output_sobel_cuda.jpg", outputImage);

    // OpenCV Sobel timing
    Mat opencvOutput;
    auto startCpu = std::chrono::high_resolution_clock::now();
    Sobel(grayImage, opencvOutput, CV_8U, 1, 1, 3);
    auto endCpu = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(endCpu - startCpu).count();

    imwrite("/content/output_sobel_opencv.jpg", opencvOutput);

    printf("Image Size: %d x %d\n", width, height);
    printf("CUDA Sobel Time: %f ms\n", cudaTime);
    printf("OpenCV Sobel Time: %f ms\n", cpuTime);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    free(h_outputImage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

## Output Explanation

| Original 	|  Output using Cuda |
|:-:	|:-:	|
|  ![archery1](https://github.com/user-attachments/assets/48d62db3-c831-413b-b540-83949f0d4847) | <img width="924" height="482" alt="sobel_output2" src="https://github.com/user-attachments/assets/dabfcddf-ad8d-4015-af14-ffdd10e272bf" />|

| Original 	|  Output using OpenCV |
|:-:	|:-:	|
|  ![archery1](https://github.com/user-attachments/assets/48d62db3-c831-413b-b540-83949f0d4847) |  <img width="950" height="530" alt="sobel_output" src="https://github.com/user-attachments/assets/a9d14a45-f1e3-4f01-b8b9-27efbfbe3905" />|

- **Sample Execution Results**:
<img width="250" height="56" alt="image" src="https://github.com/user-attachments/assets/bc42154f-6124-4fdf-9129-d7e53a82377c" />

- **Graph Analysis**:

<img width="585" height="468" alt="image" src="https://github.com/user-attachments/assets/1cf9bb69-5174-4af7-99f7-abb590647d89" />


## Answers to Questions

1. **Challenges Implementing Sobel for Color Images**:
   - Converting images to grayscale in the kernel increased complexity. Memory management and ensuring correct indexing for color to grayscale conversion required attention.

2. **Influence of Block Size**:
   - Smaller block sizes (e.g., 8x8) were efficient for smaller images but less so for larger ones, where larger blocks (e.g., 32x32) reduced overhead.

3. **CUDA vs. CPU Output Differences**:
   - The CUDA implementation was faster, with minor variations in edge sharpness due to rounding differences. CPU output took significantly more time than the GPU.

4. **Optimization Suggestions**:
   - Use shared memory in the CUDA kernel to reduce global memory access times.
   - Experiment with adaptive block sizes for larger images.

## RESULT:
Thus the program has been executed by using CUDA to Sobel implement Edge detection filter.



