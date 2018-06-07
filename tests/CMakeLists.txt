LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/src )

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

ADD_EXECUTABLE ( maintest mainTest.cpp )
TARGET_LINK_LIBRARIES( maintest 
    vocabulary )

ADD_EXECUTABLE ( featuretest featureTest.cpp )
TARGET_LINK_LIBRARIES( featuretest
    loopcheck )

ADD_EXECUTABLE ( turtest turTest.cpp )
TARGET_LINK_LIBRARIES( turtest 
    loopcheck
    vocabulary )

ADD_EXECUTABLE ( traintree trainTree.cpp )
TARGET_LINK_LIBRARIES( traintree
    vocabulary )

ADD_EXECUTABLE ( bowvec BoWvec.cpp )
TARGET_LINK_LIBRARIES( bowvec
    vocabulary )