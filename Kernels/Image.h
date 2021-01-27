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

#include "../ImageUtils/bitmap.h"
namespace Kernels {

// Properties to be used for image, this was done this way mostly
// due  to the example in the book. Something to check would be
// sending the struct through the kernel call. So far I have only
// sent C++ basic types
struct ImageProperties{
    int width;
    int height;
    int pitch;
    int channels;
};

// Holds an image buffered in device memory that has been padded with a
// second copy of the result.
// Exmaple:
//      Image image( bitmap );
//      CUDABlur( image );
//      image.exportImage( bitmap );
class Image {
public:
    Image();
    Image(const Bitmap & );
	virtual ~Image();

    void importImage(const Bitmap &);
    void exportImage(Bitmap &);

    auto size()      {return p_size;}
    auto width()     {return prop.width;}
    auto height()    {return prop.height;}
    auto pitch()     {return prop.pitch;}
    auto channels()  {return prop.channels;}

    // Returns a raw CUDA pointer to be used in kernels
    // The image is stored scaled to [0,1]. It is up to the
    // caller to move the result to data with swap_work() after
    // a kernel has finished executing.
    auto data()      {return thrust::raw_pointer_cast(&d_image[0]);}
    auto result()    {return thrust::raw_pointer_cast(&d_result[0]);}

    // Swaps our pointers to data and result so we can do work on this without
    // needing to export and import the image incurring the memory transfer costs.
    void swap_work() {swap(d_image, d_result);}
private:

    thrust::device_vector<float> d_image;
    thrust::device_vector<float> d_result;

    ImageProperties prop{0,0,0,0};
    size_t p_size{0};
};

// Performs a gaussian blur operation on the provided Bitmap for the number
// of given iterations. The given iterations guarntees that it will not be
// transferred back to host memory until after the complete openation is done.
void CUDABlur( Image&, size_t iterations = 1 );

// Fills matrix with binomial using an inplace calculation of
// pascal's triangle and then backfilling the values.
// When used with integer types this avoids using the FPU
template <typename T>
void GaussMask(std::vector<T> &matrix){
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

}
#endif /* IMAGE_H_ */
