# OpenCV
FIND_PACKAGE ( OpenCV REQUIRED )

set( CMAKE_CXX_FLAGS "-std=c++11 -O0" )

CUDA_ADD_LIBRARY ( cufeature STATIC cuFeature.cu)
# TARGET_LINK_LIBRARIES ( cufeature )

CUDA_ADD_LIBRARY ( cuscore STATIC cuScore.cu)
# TARGET_LINK_LIBRARIES ( cuscore )

ADD_LIBRARY ( cuvocabulary STATIC cuVocabulary.cpp )
# TARGET_LINK_LIBRARIES ( cuvocabulary )

ADD_LIBRARY ( cluster STATIC Cluster.cpp )
# TARGET_LINK_LIBRARIES ( cluster )

ADD_LIBRARY ( vocabulary STATIC Vocabulary.cpp )
TARGET_LINK_LIBRARIES( vocabulary
    cuvocabulary
    cufeature
    cluster
    ${OpenCV_LIBS} )