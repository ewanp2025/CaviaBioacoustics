cmake_minimum_required(VERSION 3.16)
project(CaviaBioacoustics VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 6.2 REQUIRED COMPONENTS Quick Multimedia)

qt_add_executable(appCaviaBioacoustics
    main.cpp
)

target_link_libraries(appCaviaBioacoustics
    PRIVATE Qt6::Quick Qt6::Multimedia
)
