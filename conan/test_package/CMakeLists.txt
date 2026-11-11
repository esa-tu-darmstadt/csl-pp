cmake_minimum_required(VERSION 3.25)
project(CSLMLIRInstalledConsumer LANGUAGES CXX)
if(CSL_CONSUMER_SHARED)
  # A downstream shared-library consumer may have no Squashfs package: Conan
  # removes this private static dependency after it has been linked into CSL.
  set(CMAKE_DISABLE_FIND_PACKAGE_Squashfs TRUE)
endif()
find_package(CSLMLIR CONFIG REQUIRED)
add_executable(csl-package-consumer main.cpp)
target_link_libraries(csl-package-consumer PRIVATE
  CSLMLIR::MLIRCSLDialect CSLMLIR::MLIRParsedCSLDialect
  CSLMLIR::MLIRTypedCSLDialect)
