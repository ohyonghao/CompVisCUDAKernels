/*
 * simple_cuda.cu
 *
 *  Created on: Nov 3, 2020
 *      Author: dpost
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <algorithm>

#include "Image.h"
using namespace std;

__host__
void Image::importImage(const Bitmap &bitmap){
    // TODO: Calculate importing padding known as pitch=(width + padding)
    p_width = bitmap.width();
    p_height = bitmap.height();
    p_size = p_width * p_height * 4;

    reloadImage(bitmap);
}

__host__
void Image::reloadImage(const Bitmap &bitmap){
    // Check if we can reuse memory that is allocated
    if( p_width != bitmap.width() ||
        p_height != bitmap.height() || p_size != d_image.size() ){
        p_width = bitmap.width();
        p_height = bitmap.height();
        p_size = p_width * p_height * 4;
        d_image.resize(p_size);
        d_result.resize(p_size);
    }

    // Now copy image over with resizing to [0,1)
    thrust::host_vector<float> h_image(p_size);
    std::transform(bitmap.getBits().cbegin(), bitmap.getBits().cend(), begin(h_image),[](auto in ){
        return in/255.0;
    });

    d_image = h_image;
}
// Source: https://qiita.com/naoyuki_ichimura/items/8c80e67a10d99c2fb53c
inline unsigned int iDivUp( const unsigned int &a, const unsigned int &b ) { return ( a%b != 0 ) ? (a/b+1):(a/b); }

// Global access to our mask
constexpr int MASK_WIDTH = 5;

constexpr unsigned int BLOCKW = 32;
constexpr unsigned int BLOCKH = 32;
constexpr unsigned int TILEW = 4 * 4; // pixels * channels RGBA
constexpr unsigned int CHANNEL = 4;
constexpr unsigned int PIXEL_W = TILEW / CHANNEL;
//constexpr unsigned int PIXEL_H = TILEW;
__constant__ int cd_Mask[MASK_WIDTH][MASK_WIDTH];
__global__
void kBlur(float *d_image, float *d_result, int width, int height, int maskWidth){

    // Threads id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Threads output coordinates
    const int row_o = blockIdx.y * TILEW + ty;
    const int col_o = blockIdx.x * TILEW + tx;

    // Threads input coordinate
    const int row_i = row_o - CHANNEL * (MASK_WIDTH/2);
    const int col_i = col_o - CHANNEL * (MASK_WIDTH/2);
    // grab all our pixels we need
    // Working on 4 pixels at a time
    __shared__ float s_tile[BLOCKW][BLOCKH];

    if(( row_i >= 0 ) && ( row_i < height ) &&
       ( col_i >= 0 ) && ( col_i < width ) ){
        // TODO: Replace width with pitch
        s_tile[ty][tx] = d_image[row_i * width + col_i];
    } else{
        s_tile[ty][tx]= 0.0f;
    }
    __syncthreads();

    float output;

    // Here we are going to be careful to sum
    // only the same color channel, so move by CHANNEL
    // across the X
    if( ty < TILEW && tx < TILEW ){
        for( int i = 0; i < MASK_WIDTH; ++i ){
            for( int j = 0; j < MASK_WIDTH; ++j ){
                output += cd_Mask[i][j] * s_tile[i + ty][j * CHANNEL + tx ];
            }
        }

        // Do not use pitch on output
        if( row_o < height && col_o < width){
            d_result[row_o * width + col_o] = output / (1 << ( (MASK_WIDTH - 1) * 2 ));
            //d_result[row_o * width + col_o] = 1.0;
        }
    }
}
__host__
void Image::CUDABlur(Bitmap &bitmap){
    if( p_size == 0 ){
        importImage(bitmap);
    }else{
        reloadImage(bitmap);
    }

    // launch kernel
    dim3 grid{iDivUp( p_width * CHANNEL * 2, BLOCKW), iDivUp( p_height * 2, BLOCKH)};
    dim3 threadBlock{BLOCKW, BLOCKH};
    kBlur<<<grid, threadBlock >>>(thrust::raw_pointer_cast(&d_image[0]),
                                  thrust::raw_pointer_cast(&d_result[0]),
                                  p_width * CHANNEL,
                                  p_height,
                                  maskwidth);

    cudaDeviceSynchronize();
    exportImage(bitmap);
}

__host__
void Image::exportImage(Bitmap &bitmap){
    thrust::host_vector<float> h_image(p_size);
    h_image = d_result;

    std::transform(begin(h_image), end(h_image), begin(bitmap.getBits()), [](auto in){
        return in * 255.0;
    });
}


__host__
void Image::computeMask(){
    std::vector<int> mask(maskwidth*maskwidth);
    GaussMask(mask);
    // Copy the mask to constant memory
    cudaMemcpyToSymbol(cd_Mask, mask.data(), MASK_WIDTH * sizeof(int));
}


