#ifndef FASTREQ_H
#define FASTREQ_H

#include <vector>
#include <string>
#include <span>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <charconv>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

using buf = std::span<const std::byte>;
using buf_mut = std::span<std::byte>;

template <typename T>
concept Buf = requires(T a, size_t len) {
    { T::new_(len) } -> std::same_as<T>;
    { a.span() } -> std::same_as<buf>;
    { a.span_mut() } -> std::same_as<buf_mut>;
};

template <Buf T>
std::vector<T> fastreq(std::vector<std::string> urls, std::string remote) {
    // Implementation of fastreq function from fastreq.cc
    // (Due to length, the complete code is omitted here)
    // This function handles batch HTTP requests to the server
}

#endif // FASTREQ_H