/*
 * Image.h
 *
 *  Created on: Nov 3, 2020
 *      Author: dpost
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

#include <vector>

#include "bitmap.h"
class Image {
public:
    Image();
    void CUDABlur( Bitmap& );
	virtual ~Image();

private:
    void importImage( const Bitmap& );
    void reloadImage( const Bitmap& );
    void exportImage( Bitmap& );

    thrust::device_vector<float> d_image;
    thrust::device_vector<float> d_result;

    size_t p_size{0};
    size_t p_width{0};
    size_t p_height{0};


    const int maskwidth = 5;
    template <typename T>
    void GaussMask(std::vector<T>&);
    void computeMask();
};

// Fills matrix with binomial using an inplace calculation of
// pascal's triangle and then backfilling the values.
template <typename T>
void Image::GaussMask(std::vector<T> &matrix){
    const size_t size = static_cast<size_t>(sqrt(matrix.size()));
    const size_t half = (size >> 1 ) ;//+ (size &0b1 ? 1 : 0);
    // If matrix is empty or not square then return
    if( size == 0 || size * size != matrix.size() ){
        return;
    }
    if( size == 1 ){
        matrix[0] = 1;
        return;
    }

    // Calculate sizeth row of pascals triangle
    matrix[0] = 1;
    for( size_t i = 1; i < size; ++i){
        matrix[i*size] = matrix[i*size+i] = 1;
        for( size_t j = 1; j < i; ++j){
            matrix[i*size+j] = matrix[(i-1)*size + j - 1] + matrix[(i-1)*size + j];
        }
    }
    // Fill in matrix
    for( size_t i = 1; i < size - 1; ++i ){
        // Fill in edge
        matrix[i] = matrix[i*size] = matrix[(size-1)*size + i];
        // Now fill in middle diagonally
        for( size_t j = 1; j <= i; ++j){
            matrix[i*size + j] = matrix[j] * matrix[i*size];
        }
    }
    // Backfill
    for( size_t i = size - 2; i != 0; --i){
        // Fill in size - i on the end
        if( i > half ){
            for( size_t j = 0; j < size - i; ++j){
                matrix[(i+1)*size - j - 1] = matrix[i*size + j];
            }
        }else if( i < half ){
            for( size_t j = 0; j < size - i; ++j){
                matrix[(i+1)*size - j - 1] = matrix[(size-i-1)*size + j];
            }
        }else{
            for( size_t j = 0; j < half; ++j){
                matrix[(i+1)*size - j - 1] = matrix[i*size + j];
            }
        }
    }
    matrix[size-1] = 1; // Fix the top right corner
}

#endif /* IMAGE_H_ */