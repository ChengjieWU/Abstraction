cmake_minimum_required(VERSION 3.15)


aux_source_directory(hand-isomorphism/src HAND_ISOMORPHISM_SRC)
pybind11_add_module(index index.cpp ${HAND_ISOMORPHISM_SRC})