LINK_LIBRARIES( ${PROJECT_SOURCE_DIR}/lib )

INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/src )

set( CMAKE_CXX_FLAGS "-std=c++11 -O0" )

ADD_EXECUTABLE ( maintest mainTest.cpp )
TARGET_LINK_LIBRARIES( maintest 
    # cufeature
    # cluster
    vocabulary )

ADD_EXECUTABLE ( featuretest featureTest.cpp )
TARGET_LINK_LIBRARIES( featuretest
    cuscore
    cuvocabulary )