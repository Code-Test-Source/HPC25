#!/bin/bash

# Compile the C++ program with optimization and necessary libraries
g++ -O3 -std=c++20 -pthread -lssl -lcrypto main.cpp -o calculation