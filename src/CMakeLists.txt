# OpenCV
FIND_PACKAGE ( OpenCV REQUIRED )

set( CMAKE_CXX_FLAGS "-std=c++11 -O0" )

CUDA_ADD_LIBRARY ( cubowvector STATIC cuBowVector.cu)
TARGET_LINK_LIBRARIES ( cubowvector )

ADD_LIBRARY ( cuvocabulary STATIC cuVocabulary.c )
TARGET_LINK_LIBRARIES ( cuvocabulary )

ADD_LIBRARY ( bowvector STATIC BowVector.cpp )
TARGET_LINK_LIBRARIES ( bowvector 
    cuvocabulary
    ${OpenCV_LIBS} )

ADD_LIBRARY ( cluster STATIC Cluster.cpp )
TARGET_LINK_LIBRARIES ( cluster
    bowvector )

ADD_LIBRARY ( vocabulary STATIC Vocabulary.cpp )
TARGET_LINK_LIBRARIES( vocabulary
    cubowvector
    cluster
    bowvector
    ${OpenCV_LIBS} )

ADD_EXECUTABLE ( main main.cpp )
TARGET_LINK_LIBRARIES( main
    cubowvector
    cluster
    vocabulary )