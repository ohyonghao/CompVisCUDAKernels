# CompVisCUDAKernels
This project explores the use of CUDA for writing custom kernels. It will probably become a hierarchy of multiple projects.

## Project Layout
This is layed out as a single Qt project for now.

MainWindow -- creates the UI
ImageDisplay -- handles the display of the image and coordinates with the image processor for signals
ImageProcessor -- Receives requests for filters to be ran and queues them to run in a background thread
bitmap -- Reads a bitmap file and extracts the information and bitmap data
jarvisMarch -- Implementation of Marching Squares algorithm for edge and polygon detection
point -- implements a convex hull algorithm given a set of points
Image -- A CUDA implementation of filters

## Planned Changes

* The Qt GUI portion will be separated out into its own project 
* Image files will become their own library.
* A CLI tester program will be created to run the filters on an image

-- /root
    -- GUI Test Project
    -- CLI Test program
    -- Image Library
    -- Filter Library

## Compiling
Currently the GUI project requires Qt4 or higher. This also requires C++17, and CUDA.

## TODO

* Above mentioned restructuring
* Updates to README.md to include dependency versions
* Instructions on installing dependiencies and making
