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
namespace Kernels{

//*****************************************************************************
// Global access to our mask
//*****************************************************************************

constexpr int MASK_WIDTH = 5;
constexpr int MASK_SCALE = 1 << ( ( (MASK_WIDTH - 1)  * 2 ) );

constexpr unsigned int BLOCKW = 32;
constexpr unsigned int BLOCKH = 32;
constexpr unsigned int CHANNEL = 3;
constexpr unsigned int TILEW = ((BLOCKW - (MASK_WIDTH - 1 ) * CHANNEL)/CHANNEL) ; // Number of pixels // should be 6
constexpr unsigned int TILEH = BLOCKH - MASK_WIDTH + 1;


//*****************************************************************************
// Functors
//*****************************************************************************

struct convert: public thrust::unary_function<float, float>
{
__host__ __device__
float operator()(float in){ return in/255.0;}
};


struct revert: public thrust::unary_function<float, float>
{
float operator()(float in){ return in*255.0f;}
};

__host__
Image::Image(const Bitmap &bitmap){
    importImage(bitmap);
}

__host__
void Image::importImage(const Bitmap &bitmap){
    // TODO: Calculate importing padding known as pitch=(width + padding)
    p_bpp = bitmap.bpp();
    p_width = bitmap.width();
    p_height = bitmap.height();
    p_size = p_width * p_height * p_bpp;

    cout << "bpp: " << p_bpp << endl;
    // If 4 channels, then convert to RGB
    cout << "p_size: " << p_size << endl;

    thrust::host_vector<float> h_image{begin(bitmap.getBits()), end(bitmap.getBits())};

    cout << "host_vector.size(): " << h_image.size() << endl;
    d_image = h_image;

    cout << "d_image.size(): " << d_image.size() << endl;

    thrust::transform(d_image.begin(), d_image.end(), d_image.begin(), convert() );

    d_result.resize(d_image.size());
    thrust::fill(d_result.begin(), d_result.end(), 0.5f);

}

// Source: https://qiita.com/naoyuki_ichimura/items/8c80e67a10d99c2fb53c
inline unsigned int iDivUp( const unsigned int &a, const unsigned int &b ) { return ( a%b != 0 ) ? (a/b+1):(a/b); }

__constant__ int cd_Mask[MASK_WIDTH][MASK_WIDTH];
// Width includes channels in it
__global__
void kBlur(float *d_image, float *d_result, int width, int height, int maskWidth){

    // Threads id
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;


    // Threads output coordinates
    const int row_o = blockIdx.y * TILEH + ty;
    const int col_o = blockIdx.x * TILEW * CHANNEL + tx;

    // Threads input coordinate
    const int row_i = row_o - (MASK_WIDTH/2);
    const int col_i = col_o - CHANNEL * (MASK_WIDTH/2);

    // grab all our pixels we need
    __shared__ float s_tile[BLOCKW][BLOCKH];

    if(( row_i >= 0 ) && ( row_i < height ) &&
       ( col_i >= 0 ) && ( col_i < width ) ){
        // TODO: Replace width with pitch
        s_tile[ty][tx] = d_image[row_i * width + col_i];
    } else{
        // If ghost cell set to 0.0f
        s_tile[ty][tx]= 0.0f;
    }

    __syncthreads();

    float output = 0.0f;

    // Only take coordinates used in the tile, not the whole block
    // Here we are going to be careful to sum
    // only the same color channel, so move by CHANNEL
    // across the X
    if( ty < TILEH && tx < TILEW * CHANNEL ){
        for( int i = 0; i < MASK_WIDTH; ++i ){
            for( int j = 0; j < MASK_WIDTH; ++j ){
                output += cd_Mask[i][j] * s_tile[ty + i][tx + j * CHANNEL];
            }
        }
        // Do not use pitch on output
        if( row_o < height && col_o < width){
            d_result[row_o * width + col_o] = output / MASK_SCALE > 1.0f ? 1.0f : output / MASK_SCALE;
        }
    }
}

__host__
void CUDABlur(Bitmap &bitmap){
    Image image{bitmap};
    std::vector<int> mask(MASK_WIDTH*MASK_WIDTH);
    GaussMask(mask);
    // Copy the mask to constant memory
    cudaMemcpyToSymbol(cd_Mask, mask.data(), MASK_WIDTH*MASK_WIDTH * sizeof(int));

    cout << "Calling CUDABlur" << endl;
    cout.flush();

    // launch kernel
    dim3 grid{iDivUp( image.width(), TILEW) * CHANNEL, iDivUp( image.height() , TILEH)};
    dim3 threadBlock{BLOCKW, BLOCKH};


    cout << "GRID_DIM: <" << grid.x << ", " << grid.y << ", " << grid.z << ">" << endl;
    cout << "BLOCK_DIM: <" << threadBlock.x << ", " << threadBlock.y << ", " << threadBlock.z << ">" << endl;
    kBlur<<<grid, threadBlock >>>(image.data(),
                                  image.result(),
                                  image.width() * CHANNEL,
                                  image.height(),
                                  MASK_WIDTH);

    cudaDeviceSynchronize();
    image.exportImage(bitmap);
}

__host__
void Image::exportImage(Bitmap &bitmap){
    cout << "Begin Exporting Image" << endl;
    thrust::host_vector<float> h_image(d_result.begin(), d_result.end());

    std::transform(h_image.begin(), h_image.end(), begin(bitmap.getBits()), revert());
}

}
