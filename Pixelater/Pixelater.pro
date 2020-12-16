#-------------------------------------------------
#
# Project created by QtCreator 2019-03-10T17:08:48
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Pixelater

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

QMAKE_CXXFLAGS += -O3
# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++17 O2

SOURCES += \
        main.cpp \
        MainWindow.cpp \
        ImageDisplay.cpp \
        ImageProcessor.cpp


HEADERS += \
        MainWindow.h \
        ImageDisplay.h\
        ImageProcessor.h
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

#-----------------------------------------------------
# Library CONFIGURATION
#-----------------------------------------------------
#win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../ImageUtils/release/ -lImageUtils
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../debug/ -lImageUtils
#else:unix:
LIBS += -L$$OUT_PWD/../lib/debug -lImageUtils \
        -lKernels

INCLUDEPATH += \
        $$PWD/../ImageUtils \
        $$PWD/../Kernels
DEPENDPATH += \
        $$PWD/../ImageUtils \
        $$PWD/../Kernels


#win32:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../../ImageUtils/release/ImageUtils.lib
#else:win32:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../../ImageUtils/debug/ImageUtils.lib
#else:unix:
PRE_TARGETDEPS += $$OUT_PWD/../lib/debug/libImageUtils.a \
                  $$OUT_PWD/../lib/debug/libKernels.a

#-----------------------------------------------------
# CUDA CONFIGURATION
#-----------------------------------------------------

CUDA_OBJECTS_DIR = OBJECTS_DIR/../cuda

#QMAKE_CXXFLAGS += -xcuda
## CUDA_SOURCES - the source (generally .cu) files for nvcc. No spaces in path names
CUDA_SOURCES +=
#Image.cu

# CUDA settings
SYSTEM_NAME = x86_64                   # Depending on your system either 'Win32', 'x64', or 'Win64'
## SYSTEM_TYPE - compiling for 32 or 64 bit architecture
SYSTEM_TYPE = 64

## CUDA_COMPUTE_ARCH - This will enable nvcc to compiler appropriate architecture specific code for different compute versions.
## Multiple architectures can be requested by using a space to seperate. example:

CUDA_COMPUTE_ARCH = 75

## CUDA_DEFINES - The seperate defines needed for the cuda device and host methods
CUDA_DEFINES +=

## CUDA_DIR - the directory of cuda such that CUDA\<version-number\ contains the bin, lib, src and include folders
CUDA_DIR= /usr/local/cuda-11.1

## CUDA_LIBS - the libraries to link
CUDA_LIBS= -lcuda -lcudart
CUDA_LIBS_DIR=$$CUDA_DIR/lib64

## CUDA_INC - all includes needed by the cuda files (such as CUDA\<version-number\include)
CUDA_INC+= $$CUDA_DIR/include

## NVCC_OPTIONS - any further options for the compiler
NVCC_OPTIONS += -O2 #--use_fast_math --ptxas-options=-v

## correctly formats CUDA_COMPUTE_ARCH to CUDA_ARCH with code gen flags
## resulting format example: -gencode arch=compute_20,code=sm_20
for(_a, CUDA_COMPUTE_ARCH):{
    formatted_arch =$$join(_a,'',' -gencode arch=compute_',',code=sm_$$_a')
    CUDA_ARCH += $$formatted_arch
}

## correctly formats CUDA_DEFINES for nvcc
for(_defines, CUDA_DEFINES):{
    formatted_defines += -D$$_defines
}
CUDA_DEFINES = $$formatted_defines

#nvcc config
CONFIG(debug, debug|release) {
        #Debug settings
        CUDA_OBJECTS_DIR = cudaobj/$$SYSTEM_NAME/Debug
        cuda_d.input = CUDA_SOURCES
        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda_d.commands = $$CUDA_DIR/bin/nvcc -g -G -lineinfo --std=c++17 -D_DEBUG $$CUDA_DEFINES $$NVCC_OPTIONS -I $$CUDA_INC -L$$CUDA_LIBS_DIR $$CUDA_LIBS --machine $$SYSTEM_TYPE $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda_d.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
        # Release settings
        CUDA_OBJECTS_DIR = cudaobj/$$SYSTEM_NAME/Release
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda.commands = $$CUDA_DIR/bin/nvcc --std=c++17  $$CUDA_DEFINES $$NVCC_OPTIONS -I $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda
}

LIBS += -L$$CUDA_LIBS_DIR $$CUDA_LIBS

INCLUDEPATH += $$CUDA_DIR/targets/x86_64-linux/include

