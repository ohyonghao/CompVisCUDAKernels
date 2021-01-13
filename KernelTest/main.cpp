#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "bitmap.h"
#include "Image.h"
using namespace std;
using namespace Kernels;
void printUsage(){
    cout << "Usage: ./KernelTest <image.bmp> <filter>" << endl;
}

int main( int argc, char** argv)
{
    // Read in command line parameters
    if( argc < 3 ){
            printUsage();
            return 0;
    }

    try{
        const string ext = ".bmp";
        const string out_append = "_out" + ext;
        string infile = argv[1];
        string outfile = infile;
        auto pos = outfile.find(ext, 0u);
        if( pos == string::npos ){
            cout << "Invalid file type, only bmp is supported for now." << endl;
            return 0;
        }
        outfile.replace(pos, ext.length(), out_append);
        string flag = argv[2];

        ifstream in;
        Bitmap image;
        ofstream out;

        in.open(infile, ios::binary);
        if( !in.is_open() ){
            cout << "Could not open file." << endl;
            return 0;
        }
        in >> image;
        in.close();

        if(flag == "-g"){
            cout << "Running CPU Kernel" << endl;
            pixelate(image);
        }
        else if(flag == "-G"){
            cout << "Running CUDA Kernel" << endl;
            auto start = chrono::high_resolution_clock::now();
            CUDABlur(image, 1000);
            auto end = chrono::high_resolution_clock::now();
            cout << "CUDA Kernel done in " << chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "s" << endl;
        }else{
            cout << "Invalid Flag: " << flag << endl;
            printUsage();
            return 0;
        }

        cout << "Writing result to " << outfile << endl;
        out.open(outfile, ios::binary);
        out << image;
        out.close();
    }
    catch(...)
    {
        cout << "Error: an uncoaught exception occured." << endl;
    }

    return 0;
}
