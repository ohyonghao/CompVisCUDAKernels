CONFIG -= qt

TEMPLATE = lib
DEFINES += IMAGEUTILS_LIBRARY

CONFIG += c++17 staticlib

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    BitmapIterator.cpp \
    ImageUtils.cpp \
    bitmap.cpp

HEADERS += \
    BitmapIterator.h \
    ImageUtils_global.h \
    ImageUtils.h \
    bitmap.h \
    jarvisMarch.hpp \
    point.hpp

# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target


#headersDataFiles.path = $$[QT_INSTALL_HEADERS]/ImageUtils/
#headersDataFiles.files = $$PWD/*.h
#INSTALLS += headersDataFiles


CONFIG(debug, debug|release){
    DESTDIR = ../lib/debug
}
else{
    DESTDIR = ../lib/release
}
