# CompVisCUDAKernels
This project explores the use of CUDA for writing custom kernels. It will probably become a hierarchy of multiple projects.

## Project Layout
This is layed out as a single Qt project for now.


-- /root
    -- Pixelater -- Qt Test program
        -- MainWindow -- creates the UI
        -- ImageDisplay -- handles the display of the image and coordinates with the image processor for signals
        -- ImageProcessor -- Receives requests for filters to be ran and queues them to run in a background thread
    -- CLI Test program
        -- KernelTester -- runs from command line
    -- Image Library -- Various Image related tools
        -- bitmap -- Reads a bitmap file and extracts the information and bitmap data
        -- jarvisMarch -- Implementation of Marching Squares algorithm for edge and polygon detection
        -- point -- implements a convex hull algorithm given a set of points
    -- Filter Library
        -- Image -- A CUDA implementation of filters
## Planned Changes

* Fix kernel to work properly in CUDA
* Generalize kernel launch parameters for image
* Generalize kernel launch for hardware
* Include Makefile for non-qt components

## Compiling
Currently the GUI project requires Qt4 or higher. This also requires C++17, and CUDA.

All projects can be compiled with Qt Creator, and each project has dependencies setup correctly.

## TODO

* Updates to README.md to include dependency versions
* Instructions on installing dependiencies and making
