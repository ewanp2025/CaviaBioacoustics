cmake_minimum_required(VERSION 3.21)
project(CaviaBioacoustics VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 6.7 REQUIRED COMPONENTS Quick Multimedia)

qt_add_executable(appCaviaBioacoustics
    main.cpp
)

target_link_libraries(appCaviaBioacoustics
    PRIVATE Qt6::Quick Qt6::Multimedia
)

# Android settings
if(ANDROID)
    set_target_properties(appCaviaBioacoustics PROPERTIES
        QT_ANDROID_PACKAGE_NAME "com.caviabioacoustics.app"
        QT_ANDROID_APP_NAME "Cavia Bioacoustics"
        QT_ANDROID_MIN_SDK_VERSION 28
        QT_ANDROID_TARGET_SDK_VERSION 34
        QT_ANDROID_COMPILE_SDK_VERSION 34
        QT_ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android"
    )
endif()
