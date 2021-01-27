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
#include <thrust/execution_policy.h>

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
constexpr unsigned int WORD_SIZE = 32;

//*****************************************************************************
// Functors
// Therese are used in stl style algorithms. Currently CUDA does not support
// lambda captures so we must make these functors instead.
//*****************************************************************************

struct convert: public thrust::unary_function<float, float>
{
__host__ __device__
float operator()(float in){ return in/255.0f;}
};

struct revert: public thrust::unary_function<float, float>
{
__host__ __device__
float operator()(float in){ return in*255.0f;}
};

// Source: https://qiita.com/naoyuki_ichimura/items/8c80e67a10d99c2fb53c
inline unsigned int iDivUp( const unsigned int &a, const unsigned int &b ) { return ( a%b != 0 ) ? (a/b+1):(a/b); }

__host__
Image::Image(const Bitmap &bitmap){
    importImage(bitmap);
}

__host__
void Image::importImage(const Bitmap &bitmap){
    // Copy out the important properties needed
    prop.channels = bitmap.bpp();
    prop.width = bitmap.width();
    prop.height = bitmap.height();

    const auto bitwidth = prop.width * prop.channels;
    // Calculate the pitch
    prop.pitch = iDivUp(bitwidth, WORD_SIZE) * WORD_SIZE;
    p_size = prop.pitch * prop.height;

    // TODO: If 4 channels, then convert to RGB, this would make prop.channels = 3
    cout << "channels: " << prop.channels << endl;
    cout << "prop.pitch: " << prop.pitch << endl;
    cout << "p_size: " << p_size << endl;

    // Allocate enough space on device
    d_image.resize(p_size);

    // Grab input and output iterators
    auto bits_in  = begin(bitmap.getBits());
    auto bits_out = begin(d_image);

    // Note the input image is not padded and we move with the bitwdith
    // The device image we are padding by moving its iterator with the pitch
    auto i = 0;
    try{
        for( ; i < prop.height; ++i ){
            thrust::copy_n(bits_in, bitwidth, bits_out);
            bits_in  += bitwidth;
            bits_out += prop.pitch;
        }
    }catch(...){
        cout << "Error caught transferring host->device on " << i << "th iteration" << endl;
        auto cudaerr = cudaGetLastError();
        cout << "CudaError: " << cudaerr << endl;
        throw;
    }

    cout << "d_image.size(): " << d_image.size() << endl;
    d_result.resize(d_image.size());

    // Scale the image on the device
    thrust::transform(thrust::device, d_image.begin(), d_image.end(), d_image.begin(), convert() );
}

__host__
void Image::exportImage(Bitmap &bitmap){
    cout << "Begin Exporting Image" << endl;
    thrust::host_vector<float> h_image(d_image.begin(), d_image.end());

    auto bits_in = begin(h_image);
    auto bits_out = begin(bitmap.getBits());

    for( auto i = 0; i < prop.height; ++i){
        std::transform( bits_in, bits_in + prop.width * prop.channels, bits_out, revert() );
        bits_in  += prop.pitch;
        bits_out += prop.width * prop.channels;
    }
}

__constant__ int cd_Mask[MASK_WIDTH][MASK_WIDTH];
// Width includes channels in it
__global__
void kBlur(float *d_image, float *d_result, int width, int height, int maskWidth, int pitch){

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
        s_tile[ty][tx] = d_image[row_i * pitch + col_i];
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
            d_result[row_o * pitch + col_o] = output / MASK_SCALE > 1.0f ? 1.0f : output / MASK_SCALE;
        }
    }
}

__host__
void CUDABlur(Image &image, size_t iterations){
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
    for( auto i = 0; i < iterations; ++i ){
        kBlur<<<grid, threadBlock >>>(image.data(),
                                      image.result(),
                                      image.width() * CHANNEL,
                                      image.height(),
                                      MASK_WIDTH,
                                      image.pitch());

        cudaDeviceSynchronize();
        image.swap_work();
    }
}


}
