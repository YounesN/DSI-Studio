# -------------------------------------------------
# Project created by QtCreator 2011-01-20T20:02:59
# -------------------------------------------------
QT += core \
    gui \
    opengl \
    printsupport
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TARGET = dsi_studio
TEMPLATE = app

win32* {
#change to the directory that contains boost library
INCLUDEPATH += ../include
RC_FILE = dsi_studio.rc
}

linux* {
QMAKE_CXXFLAGS += -fpermissive
LIBS += -lboost_thread \
        -lboost_program_options \
        -lGLU \
        -lz
}

mac{

#change to the directory that contains boost library
INCLUDEPATH += /Users/frankyeh/include
LIBS += -L/Users/frankyeh/include/lib -lboost_system \
        -L/Users/frankyeh/include/lib -lboost_thread \
        -L/Users/frankyeh/include/lib -lboost_program_options \
        -L/Users/frankyek/include/lib -litpp_debug_win32 \
        -lz
ICON = dsi_studio.icns
}


# you may need to change the include directory of boost library
INCLUDEPATH += libs \
    libs/dsi \
    libs/tracking \
    libs/mapping
HEADERS += mainwindow.h \
    dicom/dicom_parser.h \
    dicom/dwi_header.hpp \
    libs/dsi/tessellated_icosahedron.hpp \
    libs/dsi/space_mapping.hpp \
    libs/dsi/sh_process.hpp \
    libs/dsi/sample_model.hpp \
    libs/dsi/racian_noise.hpp \
    libs/dsi/qbi_process.hpp \
    libs/dsi/odf_process.hpp \
    libs/dsi/odf_deconvolusion.hpp \
    libs/dsi/odf_decomposition.hpp \
    libs/dsi/mix_gaussian_model.hpp \
    libs/dsi/layout.hpp \
    libs/dsi/image_model.hpp \
    libs/dsi/gqi_process.hpp \
    libs/dsi/gqi_mni_reconstruction.hpp \
    libs/dsi/dti_process.hpp \
    libs/dsi/dsi_process.hpp \
    libs/dsi/basic_voxel.hpp \
    libs/dsi_interface_static_link.h \
    SliceModel.h \
    tracking/tracking_window.h \
    reconstruction/reconstruction_window.h \
    tracking/slice_view_scene.h \
    opengl/glwidget.h \
    libs/tracking/tracking_method.hpp \
    libs/tracking/roi.hpp \
    libs/tracking/interpolation_process.hpp \
    libs/tracking/fib_data.hpp \
    libs/tracking/basic_process.hpp \
    libs/tracking/tract_cluster.hpp \
    tracking/region/regiontablewidget.h \
    tracking/region/Regions.h \
    tracking/region/RegionModel.h \
    libs/tracking/tract_model.hpp \
    tracking/tract/tracttablewidget.h \
    opengl/renderingtablewidget.h \
    qcolorcombobox.h \
    libs/tracking/tracking_thread.hpp \
    libs/prog_interface_static_link.h \
    simulation.h \
    reconstruction/vbcdialog.h \
    libs/mapping/atlas.hpp \
    libs/mapping/fa_template.hpp \
    plot/qcustomplot.h \
    view_image.h \
    libs/vbc/vbc_database.h \
    libs/gzip_interface.hpp \
    libs/dsi/racian_noise.hpp \
    libs/dsi/mix_gaussian_model.hpp \
    libs/dsi/layout.hpp \
    manual_alignment.h \
    tracking/vbc_dialog.hpp \
    tracking/tract_report.hpp \
    tracking/color_bar_dialog.hpp \
    tracking/connectivity_matrix_dialog.h \
    tracking/atlasdialog.h \
    dicom/motion_dialog.hpp \
    libs/dsi/ica_process.hpp \
    libs/dsi/sig_process.hpp \
    libs/dsi/icabsm_process.hpp \
    libs/dsi/icaidsi_process.hpp \
    libs/dsi/icabsm_process_gpu.hpp

FORMS += mainwindow.ui \
    tracking/tracking_window.ui \
    reconstruction/reconstruction_window.ui \
    dicom/dicom_parser.ui \
    simulation.ui \
    reconstruction/vbcdialog.ui \
    view_image.ui \
    manual_alignment.ui \
    tracking/vbc_dialog.ui \
    tracking/tract_report.ui \
    tracking/color_bar_dialog.ui \
    tracking/connectivity_matrix_dialog.ui \
    tracking/atlasdialog.ui \
    dicom/motion_dialog.ui
RESOURCES += \
    icons.qrc
SOURCES += main.cpp \
    mainwindow.cpp \
    dicom/dicom_parser.cpp \
    dicom/dwi_header.cpp \
    libs/utility/prog_interface.cpp \
    libs/dsi/sample_model.cpp \
    libs/dsi/dsi_interface_imp.cpp \
    libs/tracking/interpolation_process.cpp \
    libs/tracking/tract_cluster.cpp \
    SliceModel.cpp \
    tracking/tracking_window.cpp \
    reconstruction/reconstruction_window.cpp \
    tracking/slice_view_scene.cpp \
    opengl/glwidget.cpp \
    tracking/region/regiontablewidget.cpp \
    tracking/region/Regions.cpp \
    tracking/region/RegionModel.cpp \
    libs/tracking/tract_model.cpp \
    tracking/tract/tracttablewidget.cpp \
    opengl/renderingtablewidget.cpp \
    qcolorcombobox.cpp \
    cmd/trk.cpp \
    cmd/rec.cpp \
    simulation.cpp \
    reconstruction/vbcdialog.cpp \
    cmd/src.cpp \
    libs/mapping/atlas.cpp \
    libs/mapping/fa_template.cpp \
    plot/qcustomplot.cpp \
    cmd/ana.cpp \
    view_image.cpp \
    libs/vbc/vbc_database.cpp \
    manual_alignment.cpp \
    tracking/vbc_dialog.cpp \
    tracking/tract_report.cpp \
    tracking/color_bar_dialog.cpp \
    cmd/exp.cpp \
    tracking/connectivity_matrix_dialog.cpp \
    libs/dsi/tessellated_icosahedron.cpp \
    cmd/atl.cpp \
    tracking/atlasdialog.cpp \
    dicom/motion_dialog.cpp \
    cmd/cnt.cpp \
    cmd/vis.cpp

OTHER_FILES += \
    options.txt

# Boost Library

QMAKE_LIBDIR += $$PWD/stage/lib\

# IT++ library
win32: LIBS += -L$$PWD/itpp/lib/ -litpp_debug_win32

INCLUDEPATH += $$PWD/itpp
DEPENDPATH += $$PWD/itpp

win32:!win32-g++: PRE_TARGETDEPS += $$PWD/itpp/lib/itpp_debug_win32.lib
else:win32-g++: PRE_TARGETDEPS += $$PWD/itpp/lib/libitpp_debug_win32.a

# LAPACK library

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/itpp/lib/ -llapack_win32_MT
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/itpp/lib/ -llapack_win32_MTd

INCLUDEPATH += $$PWD/itpp
DEPENDPATH += $$PWD/itpp

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/itpp/lib/liblapack_win32_MT.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/itpp/lib/liblapack_win32_MTd.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/itpp/lib/lapack_win32_MT.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/itpp/lib/lapack_win32_MTd.lib

# BLAS library

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/itpp/lib/ -lblas_win32_MT
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/itpp/lib/ -lblas_win32_MTd

INCLUDEPATH += $$PWD/itpp
DEPENDPATH += $$PWD/itpp

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/itpp/lib/libblas_win32_MT.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/itpp/lib/libblas_win32_MTd.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/itpp/lib/blas_win32_MT.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/itpp/lib/blas_win32_MTd.lib


# LEVMAR library

win32: LIBS += -L$$PWD/levmar/lib/ -llevmar

INCLUDEPATH += $$PWD/levmar
DEPENDPATH += $$PWD/levmar

win32:!win32-g++: PRE_TARGETDEPS += $$PWD/levmar/lib/levmar.lib
else:win32-g++: PRE_TARGETDEPS += $$PWD/levmar/lib/liblevmar.a

# Armadillo library

INCLUDEPATH += $$PWD/armadillo/include

# CUDA Configuration
DESTDIR = debug
OBJECTS_DIR = debug/obj           # directory where .obj files will be saved
CUDA_OBJECTS_DIR = debug/obj      # directory where .obj  of cuda file will be saved
# This makes the .cu files appear in your project
OTHER_FILES += cudaKernels.cu      # this is your cu file need to compile

# CUDA settings <-- may change depending on your system (i think you missed this)
CUDA_SOURCES += cudaKernels.cu   # let NVCC know which file you want to compile CUDA NVCC

CUDA_DIR = C:/cuda
SYSTEM_NAME = Win32         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 32            # '32' or '64', depending on your system
CUDA_ARCH = sm_52           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS += --use_fast_math # default setting

# include paths
INCLUDEPATH += $$CUDA_DIR/include\

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME\

# Add the necessary libraries
CUDA_LIBS= -lcuda -lcudart
#add quotation for those directories contain space (Windows required)
#CUDA_INC +=$$join(INCLUDEPATH,'" -I"','-I"','"')
CUDA_INC = -I"C:/cuda/include"\
           -I"C:/Users/Younes/dsi-studio"\
           #-I"C:/Users/Younes/dsi-studio/itpp"\
           -I"C:/Users/Younes/dsi-studio/levmar"

LIBS += $$CUDA_LIBS
#nvcc config
# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

CONFIG(debug, debug|release) {
    #Debug settings
    # Debug mode
    cuda_d.input    = CUDA_SOURCES
    cuda_d.output   = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                      --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                      --compile -cudart static -g -DWIN32 -D_MBCS \
                      -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                      -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
     # Release settings
     cuda.input    = CUDA_SOURCES
     cuda.output   = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
     cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                    --compile -cudart static -DWIN32 -D_MBCS \
                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
     cuda.dependency_type = TYPE_C
     QMAKE_EXTRA_COMPILERS += cuda
}
