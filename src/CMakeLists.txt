# OpenCV
FIND_PACKAGE ( OpenCV REQUIRED )
# CUDA
FIND_PACKAGE ( CUDA REQUIRED )

set( CMAKE_CXX_FLAGS "-std=c++11 -O0" )

CUDA_ADD_LIBRARY ( cufeature STATIC cuFeature.cu)

CUDA_ADD_LIBRARY ( cuscore STATIC cuScore.cu)
TARGET_LINK_LIBRARIES ( cuscore 
    ${CUDA_cusparse_LIBRARY} )

ADD_LIBRARY ( cuvocabulary STATIC cuVocabulary.cpp )

ADD_LIBRARY ( loopcheck STATIC LoopCheck.cpp)
TARGET_LINK_LIBRARIES( loopcheck
    cuscore )

ADD_LIBRARY ( cluster STATIC Cluster.cpp )

ADD_LIBRARY ( vocabulary STATIC Vocabulary.cpp )
TARGET_LINK_LIBRARIES( vocabulary
    cuvocabulary
    cufeature
    cluster
    ${OpenCV_LIBS} )    