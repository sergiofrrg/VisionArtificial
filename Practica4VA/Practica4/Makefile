#############################################################################
# Makefile for building: Practica4
# Generated by qmake (2.01a) (Qt 4.8.1) on: Fri May 16 12:52:47 2014
# Project:  Practica4.pro
# Template: app
# Command: /usr/bin/qmake-qt4 -spec /usr/share/qt4/mkspecs/linux-g++ -o Makefile Practica4.pro
#############################################################################

####### Compiler, tools and options

CC            = gcc
CXX           = g++
DEFINES       = -DQT_WEBKIT
CFLAGS        = -pipe -O2 -Wall -W $(DEFINES)
CXXFLAGS      = -pipe -O2 -Wall -W $(DEFINES)
INCPATH       = -I/usr/share/qt4/mkspecs/linux-g++ -I. -I/home/aescudero/VisionArtificial/Practica4VA/Practica4/../../../../usr/local/lib -I/home/aescudero/VisionArtificial/Practica4VA/Practica4/../../../../usr/local/include -I/home/aescudero/VisionArtificial/Practica4VA/Practica4/../../../../usr/local/lib -I/home/aescudero/VisionArtificial/Practica4VA/Practica4/../../../../usr/local/lib -I/home/aescudero/VisionArtificial/Practica4VA/Practica4/../../../../usr/local/lib -I/home/aescudero/VisionArtificial/Practica4VA/Practica4/../../../../usr/local/lib -I/usr/local/include -I/usr/local/include
LINK          = g++
LFLAGS        = -Wl,-O1
LIBS          = $(SUBLIBS)   -L/home/aescudero/VisionArtificial/Practica4VA/Practica4/../../../../usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_flann -lopencv_legacy -L/home/aescudero/VisionArtificial/Practica4VA/Practica4/../../../../../usr/local/lib/ -lopencv_objdetect 
AR            = ar cqs
RANLIB        = 
QMAKE         = /usr/bin/qmake-qt4
TAR           = tar -cf
COMPRESS      = gzip -9f
COPY          = cp -f
SED           = sed
COPY_FILE     = $(COPY)
COPY_DIR      = $(COPY) -r
STRIP         = strip
INSTALL_FILE  = install -m 644 -p
INSTALL_DIR   = $(COPY_DIR)
INSTALL_PROGRAM = install -m 755 -p
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p

####### Output directory

OBJECTS_DIR   = ./

####### Files

SOURCES       = main.cpp \
		subspace.cpp \
		decomposition.cpp \
		helper.cpp \
		filtro.cpp 
OBJECTS       = main.o \
		subspace.o \
		decomposition.o \
		helper.o \
		filtro.o
DIST          = /usr/share/qt4/mkspecs/common/unix.conf \
		/usr/share/qt4/mkspecs/common/linux.conf \
		/usr/share/qt4/mkspecs/common/gcc-base.conf \
		/usr/share/qt4/mkspecs/common/gcc-base-unix.conf \
		/usr/share/qt4/mkspecs/common/g++-base.conf \
		/usr/share/qt4/mkspecs/common/g++-unix.conf \
		/usr/share/qt4/mkspecs/qconfig.pri \
		/usr/share/qt4/mkspecs/modules/qt_webkit_version.pri \
		/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/usr/share/qt4/mkspecs/features/qt_config.prf \
		/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/usr/share/qt4/mkspecs/features/default_pre.prf \
		/usr/share/qt4/mkspecs/features/release.prf \
		/usr/share/qt4/mkspecs/features/default_post.prf \
		/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf \
		/usr/share/qt4/mkspecs/features/warn_on.prf \
		/usr/share/qt4/mkspecs/features/resources.prf \
		/usr/share/qt4/mkspecs/features/uic.prf \
		/usr/share/qt4/mkspecs/features/yacc.prf \
		/usr/share/qt4/mkspecs/features/lex.prf \
		/usr/share/qt4/mkspecs/features/include_source_dir.prf \
		Practica4.pro
QMAKE_TARGET  = Practica4
DESTDIR       = 
TARGET        = Practica4

first: all
####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

all: Makefile $(TARGET)

$(TARGET):  $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS)

Makefile: Practica4.pro  /usr/share/qt4/mkspecs/linux-g++/qmake.conf /usr/share/qt4/mkspecs/common/unix.conf \
		/usr/share/qt4/mkspecs/common/linux.conf \
		/usr/share/qt4/mkspecs/common/gcc-base.conf \
		/usr/share/qt4/mkspecs/common/gcc-base-unix.conf \
		/usr/share/qt4/mkspecs/common/g++-base.conf \
		/usr/share/qt4/mkspecs/common/g++-unix.conf \
		/usr/share/qt4/mkspecs/qconfig.pri \
		/usr/share/qt4/mkspecs/modules/qt_webkit_version.pri \
		/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/usr/share/qt4/mkspecs/features/qt_config.prf \
		/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/usr/share/qt4/mkspecs/features/default_pre.prf \
		/usr/share/qt4/mkspecs/features/release.prf \
		/usr/share/qt4/mkspecs/features/default_post.prf \
		/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf \
		/usr/share/qt4/mkspecs/features/warn_on.prf \
		/usr/share/qt4/mkspecs/features/resources.prf \
		/usr/share/qt4/mkspecs/features/uic.prf \
		/usr/share/qt4/mkspecs/features/yacc.prf \
		/usr/share/qt4/mkspecs/features/lex.prf \
		/usr/share/qt4/mkspecs/features/include_source_dir.prf
	$(QMAKE) -spec /usr/share/qt4/mkspecs/linux-g++ -o Makefile Practica4.pro
/usr/share/qt4/mkspecs/common/unix.conf:
/usr/share/qt4/mkspecs/common/linux.conf:
/usr/share/qt4/mkspecs/common/gcc-base.conf:
/usr/share/qt4/mkspecs/common/gcc-base-unix.conf:
/usr/share/qt4/mkspecs/common/g++-base.conf:
/usr/share/qt4/mkspecs/common/g++-unix.conf:
/usr/share/qt4/mkspecs/qconfig.pri:
/usr/share/qt4/mkspecs/modules/qt_webkit_version.pri:
/usr/share/qt4/mkspecs/features/qt_functions.prf:
/usr/share/qt4/mkspecs/features/qt_config.prf:
/usr/share/qt4/mkspecs/features/exclusive_builds.prf:
/usr/share/qt4/mkspecs/features/default_pre.prf:
/usr/share/qt4/mkspecs/features/release.prf:
/usr/share/qt4/mkspecs/features/default_post.prf:
/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf:
/usr/share/qt4/mkspecs/features/warn_on.prf:
/usr/share/qt4/mkspecs/features/resources.prf:
/usr/share/qt4/mkspecs/features/uic.prf:
/usr/share/qt4/mkspecs/features/yacc.prf:
/usr/share/qt4/mkspecs/features/lex.prf:
/usr/share/qt4/mkspecs/features/include_source_dir.prf:
qmake:  FORCE
	@$(QMAKE) -spec /usr/share/qt4/mkspecs/linux-g++ -o Makefile Practica4.pro

dist: 
	@$(CHK_DIR_EXISTS) .tmp/Practica41.0.0 || $(MKDIR) .tmp/Practica41.0.0 
	$(COPY_FILE) --parents $(SOURCES) $(DIST) .tmp/Practica41.0.0/ && (cd `dirname .tmp/Practica41.0.0` && $(TAR) Practica41.0.0.tar Practica41.0.0 && $(COMPRESS) Practica41.0.0.tar) && $(MOVE) `dirname .tmp/Practica41.0.0`/Practica41.0.0.tar.gz . && $(DEL_FILE) -r .tmp/Practica41.0.0


clean:compiler_clean 
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core


####### Sub-libraries

distclean: clean
	-$(DEL_FILE) $(TARGET) 
	-$(DEL_FILE) Makefile


check: first

compiler_rcc_make_all:
compiler_rcc_clean:
compiler_uic_make_all:
compiler_uic_clean:
compiler_image_collection_make_all: qmake_image_collection.cpp
compiler_image_collection_clean:
	-$(DEL_FILE) qmake_image_collection.cpp
compiler_yacc_decl_make_all:
compiler_yacc_decl_clean:
compiler_yacc_impl_make_all:
compiler_yacc_impl_clean:
compiler_lex_make_all:
compiler_lex_clean:
compiler_clean: 

####### Compile

main.o: main.cpp /usr/local/include/opencv2/core/core.hpp \
		/usr/local/include/opencv2/core/types_c.h \
		/usr/local/include/opencv2/core/version.hpp \
		/usr/local/include/opencv2/core/operations.hpp \
		/usr/local/include/opencv2/core/mat.hpp \
		/usr/local/include/opencv2/highgui/highgui.hpp \
		/usr/local/include/opencv2/highgui/highgui_c.h \
		/usr/local/include/opencv2/core/core_c.h \
		/usr/local/include/opencv2/imgproc/imgproc.hpp \
		/usr/local/include/opencv2/imgproc/types_c.h \
		/usr/local/include/opencv2/features2d/features2d.hpp \
		/usr/local/include/opencv2/flann/miniflann.hpp \
		/usr/local/include/opencv2/flann/defines.h \
		/usr/local/include/opencv2/flann/config.h \
		/usr/local/include/opencv2/flann/flann.hpp \
		/usr/local/include/opencv2/flann/flann_base.hpp \
		/usr/local/include/opencv2/flann/general.h \
		/usr/local/include/opencv2/flann/matrix.h \
		/usr/local/include/opencv2/flann/params.h \
		/usr/local/include/opencv2/flann/any.h \
		/usr/local/include/opencv2/flann/saving.h \
		/usr/local/include/opencv2/flann/nn_index.h \
		/usr/local/include/opencv2/flann/result_set.h \
		/usr/local/include/opencv2/flann/all_indices.h \
		/usr/local/include/opencv2/flann/kdtree_index.h \
		/usr/local/include/opencv2/flann/dynamic_bitset.h \
		/usr/local/include/opencv2/flann/dist.h \
		/usr/local/include/opencv2/flann/heap.h \
		/usr/local/include/opencv2/flann/allocator.h \
		/usr/local/include/opencv2/flann/random.h \
		/usr/local/include/opencv2/flann/kdtree_single_index.h \
		/usr/local/include/opencv2/flann/kmeans_index.h \
		/usr/local/include/opencv2/flann/logger.h \
		/usr/local/include/opencv2/flann/composite_index.h \
		/usr/local/include/opencv2/flann/linear_index.h \
		/usr/local/include/opencv2/flann/hierarchical_clustering_index.h \
		/usr/local/include/opencv2/flann/lsh_index.h \
		/usr/local/include/opencv2/flann/lsh_table.h \
		/usr/local/include/opencv2/flann/autotuned_index.h \
		/usr/local/include/opencv2/flann/ground_truth.h \
		/usr/local/include/opencv2/flann/index_testing.h \
		/usr/local/include/opencv2/flann/timer.h \
		/usr/local/include/opencv2/flann/sampling.h \
		/usr/local/include/opencv2/legacy/legacy.hpp \
		/usr/local/include/opencv2/imgproc/imgproc_c.h \
		/usr/local/include/opencv2/calib3d/calib3d.hpp \
		/usr/local/include/opencv2/ml/ml.hpp \
		/usr/local/include/opencv2/objdetect/objdetect.hpp \
		subspace.hpp \
		filtro.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o main.o main.cpp

subspace.o: subspace.cpp subspace.hpp \
		/usr/local/include/opencv2/core/core.hpp \
		/usr/local/include/opencv2/core/types_c.h \
		/usr/local/include/opencv2/core/version.hpp \
		/usr/local/include/opencv2/core/operations.hpp \
		/usr/local/include/opencv2/core/mat.hpp \
		decomposition.hpp \
		helper.hpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o subspace.o subspace.cpp

decomposition.o: decomposition.cpp decomposition.hpp \
		/usr/local/include/opencv2/core/core.hpp \
		/usr/local/include/opencv2/core/types_c.h \
		/usr/local/include/opencv2/core/version.hpp \
		/usr/local/include/opencv2/core/operations.hpp \
		/usr/local/include/opencv2/core/mat.hpp \
		helper.hpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o decomposition.o decomposition.cpp

helper.o: helper.cpp /usr/local/include/opencv2/opencv.hpp \
		/usr/local/include/opencv2/core/core_c.h \
		/usr/local/include/opencv2/core/types_c.h \
		/usr/local/include/opencv2/core/core.hpp \
		/usr/local/include/opencv2/core/version.hpp \
		/usr/local/include/opencv2/core/operations.hpp \
		/usr/local/include/opencv2/core/mat.hpp \
		/usr/local/include/opencv2/flann/miniflann.hpp \
		/usr/local/include/opencv2/flann/defines.h \
		/usr/local/include/opencv2/flann/config.h \
		/usr/local/include/opencv2/imgproc/imgproc_c.h \
		/usr/local/include/opencv2/imgproc/types_c.h \
		/usr/local/include/opencv2/imgproc/imgproc.hpp \
		/usr/local/include/opencv2/photo/photo.hpp \
		/usr/local/include/opencv2/photo/photo_c.h \
		/usr/local/include/opencv2/video/video.hpp \
		/usr/local/include/opencv2/video/tracking.hpp \
		/usr/local/include/opencv2/video/background_segm.hpp \
		/usr/local/include/opencv2/features2d/features2d.hpp \
		/usr/local/include/opencv2/objdetect/objdetect.hpp \
		/usr/local/include/opencv2/calib3d/calib3d.hpp \
		/usr/local/include/opencv2/ml/ml.hpp \
		/usr/local/include/opencv2/highgui/highgui_c.h \
		/usr/local/include/opencv2/highgui/highgui.hpp \
		/usr/local/include/opencv2/contrib/contrib.hpp \
		/usr/local/include/opencv2/contrib/retina.hpp \
		/usr/local/include/opencv2/contrib/openfabmap.hpp \
		helper.hpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o helper.o helper.cpp

filtro.o: filtro.cpp filtro.h \
		/usr/local/include/opencv2/core/core.hpp \
		/usr/local/include/opencv2/core/types_c.h \
		/usr/local/include/opencv2/core/version.hpp \
		/usr/local/include/opencv2/core/operations.hpp \
		/usr/local/include/opencv2/core/mat.hpp \
		/usr/local/include/opencv2/highgui/highgui.hpp \
		/usr/local/include/opencv2/highgui/highgui_c.h \
		/usr/local/include/opencv2/core/core_c.h \
		/usr/local/include/opencv2/imgproc/imgproc.hpp \
		/usr/local/include/opencv2/imgproc/types_c.h \
		/usr/local/include/opencv2/features2d/features2d.hpp \
		/usr/local/include/opencv2/flann/miniflann.hpp \
		/usr/local/include/opencv2/flann/defines.h \
		/usr/local/include/opencv2/flann/config.h \
		/usr/local/include/opencv2/flann/flann.hpp \
		/usr/local/include/opencv2/flann/flann_base.hpp \
		/usr/local/include/opencv2/flann/general.h \
		/usr/local/include/opencv2/flann/matrix.h \
		/usr/local/include/opencv2/flann/params.h \
		/usr/local/include/opencv2/flann/any.h \
		/usr/local/include/opencv2/flann/saving.h \
		/usr/local/include/opencv2/flann/nn_index.h \
		/usr/local/include/opencv2/flann/result_set.h \
		/usr/local/include/opencv2/flann/all_indices.h \
		/usr/local/include/opencv2/flann/kdtree_index.h \
		/usr/local/include/opencv2/flann/dynamic_bitset.h \
		/usr/local/include/opencv2/flann/dist.h \
		/usr/local/include/opencv2/flann/heap.h \
		/usr/local/include/opencv2/flann/allocator.h \
		/usr/local/include/opencv2/flann/random.h \
		/usr/local/include/opencv2/flann/kdtree_single_index.h \
		/usr/local/include/opencv2/flann/kmeans_index.h \
		/usr/local/include/opencv2/flann/logger.h \
		/usr/local/include/opencv2/flann/composite_index.h \
		/usr/local/include/opencv2/flann/linear_index.h \
		/usr/local/include/opencv2/flann/hierarchical_clustering_index.h \
		/usr/local/include/opencv2/flann/lsh_index.h \
		/usr/local/include/opencv2/flann/lsh_table.h \
		/usr/local/include/opencv2/flann/autotuned_index.h \
		/usr/local/include/opencv2/flann/ground_truth.h \
		/usr/local/include/opencv2/flann/index_testing.h \
		/usr/local/include/opencv2/flann/timer.h \
		/usr/local/include/opencv2/flann/sampling.h \
		/usr/local/include/opencv2/legacy/legacy.hpp \
		/usr/local/include/opencv2/imgproc/imgproc_c.h \
		/usr/local/include/opencv2/calib3d/calib3d.hpp \
		/usr/local/include/opencv2/ml/ml.hpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o filtro.o filtro.cpp

####### Install

install:   FORCE

uninstall:   FORCE

FORCE:

