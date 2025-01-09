cmake_minimum_required(VERSION 3.30)


#set(VCPKG_DIR "D:/Software/")
#
#set(CMAKE_TOOLCHAIN_FILE "${VCPKG_DIR}vcpkg/scripts/buildsystems/vcpkg.cmake")
#
#set(VCPKG_TARGET_TRIPLET "x64-windows-static")


project(NN)

set(CMAKE_CXX_STANDARD 20)
add_subdirectory(Train)
add_subdirectory(Test)


# openCV vcpkg
#find_package(spdlog CONFIG REQUIRED)
#target_link_libraries(NN_test PRIVATE spdlog::spdlog_header_only)