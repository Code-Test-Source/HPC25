#include "fastreq.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>

class MyBuf {
public:
    std::vector<std::byte> data_;

    MyBuf() = default;
    explicit MyBuf(size_t length) : data_(length) {}
    explicit MyBuf(std::vector<std::byte> data) : data_(std::move(data)) {}

    MyBuf(MyBuf&& other) noexcept : data_(std::move(other.data_)) {}
    MyBuf& operator=(MyBuf&& other) noexcept {
        data_ = std::move(other.data_);
        return *this;
    }
    MyBuf(const MyBuf& other) : data_(other.data_) {}
    MyBuf& operator=(const MyBuf& other) {
        data_ = other.data_;
        return *this;
    }

    static MyBuf new_(size_t len) {
        return MyBuf(len);
    }

    buf span() const {
        return buf(data_.data(), data_.size());
    }

    buf_mut span_mut() {
        return buf_mut(data_.data(), data_.size());
    }
};

bool is_duplicate(const MyBuf& a, const MyBuf& b) {
    size_t a_size = a.data_.size() - 1;
    size_t b_size = b.data_.size() - 1;
    if (a_size != b_size) return false;
    if (memcmp(a.data_.data() + a_size - 16, b.data_.data() + b_size - 16, 16) != 0) return false;
    int diff_count = 0;
    for (size_t i = 0; i < a_size; i++) {
        if (a.data_[i] != b.data_[i]) {
            diff_count++;
            if (diff_count > 3) return false;
        }
    }
    return true;
}

std::vector<int> parse_group_sizes(const std::string& json_str) {
    std::vector<int> result;
    std::string s = json_str;
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    if (s.front() == '[' && s.back() == ']') {
        s = s.substr(1, s.size() - 2);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos) comma = s.size();
            std::string num_str = s.substr(pos, comma - pos);
            try {
                int num = std::stoi(num_str);
                result.push_back(num);
            } catch (...) {
                // ignore invalid numbers
            }
            pos = comma + 1;
        }
    }
    return result;
}

int main() {
    // Step 1: Get group sizes from server
    std::vector<std::string> group_url = {"/"};
    auto group_responses = fastreq<MyBuf>(group_url, "localhost:18080");
    if (group_responses.empty()) {
        std::cerr << "Failed to get group sizes" << std::endl;
        return 1;
    }
    std::string group_str(reinterpret_cast<const char*>(group_responses[0].span().data()), group_responses[0].span().size());
    std::vector<int> group_sizes = parse_group_sizes(group_str);

    // Step 2: Generate all URLs for charms
    std::vector<std::string> urls;
    std::vector<int> group_ids;
    for (int i = 0; i < group_sizes.size(); i++) {
        for (int j = 0; j < group_sizes[i]; j++) {
            urls.push_back("/" + std::to_string(i) + "/" + std::to_string(j));
            group_ids.push_back(i);
        }
    }

    // Step 3: Download all charms in batch
    auto responses = fastreq<MyBuf>(urls, "localhost:18080");
    if (responses.size() != urls.size()) {
        std::cerr << "Download failed" << std::endl;
        return 1;
    }

    // Step 4: Group responses by group ID
    std::vector<std::vector<MyBuf>> groups(group_sizes.size());
    for (int i = 0; i < responses.size(); i++) {
        int gid = group_ids[i];
        groups[gid].push_back(std::move(responses[i]));
    }

    // Step 5: Process each group
    std::vector<std::vector<std::vector<MyBuf>>> all_groups_deduped(group_sizes.size(), std::vector<std::vector<MyBuf>>(256));
    for (int gid = 0; gid < group_sizes.size(); gid++) {
        auto& group = groups[gid];
        std::vector<std::vector<MyBuf>> buckets(256);
        
        // Classify by price and verify checksum
        for (auto& charm : group) {
            size_t content_size = charm.data_.size() - 1;
            if (content_size < 32 || content_size > 16384) continue;
            std::byte checksum = 0;
            for (size_t i = 0; i < content_size; i++) {
                checksum ^= charm.data_[i];
            }
            if (checksum != charm.data_[content_size]) continue;
            int price = static_cast<int>(checksum);
            buckets[price].push_back(std::move(charm));
        }

        // Deduplicate each price bucket
        for (int p = 0; p < 256; p++) {
            auto& bucket = buckets[p];
            std::vector<MyBuf> deduped;
            std::map<std::tuple<size_t, std::string>, std::vector<MyBuf>> groups;
            
            // Group by length and last 16 bytes
            for (auto& charm : bucket) {
                size_t len = charm.data_.size() - 1;
                std::string last16(reinterpret_cast<const char*>(charm.data_.data() + len - 16), 16);
                auto key = std::make_tuple(len, last16);
                groups[key].push_back(std::move(charm));
            }
            
            // Deduplicate within each group
            for (auto& [key, charms] : groups) {
                for (auto& charm : charms) {
                    bool duplicate = false;
                    for (auto& existing : deduped) {
                        if (is_duplicate(charm, existing)) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        deduped.push_back(std::move(charm));
                    }
                }
            }
            all_groups_deduped[gid][p] = std::move(deduped);
        }
    }

    // Step 6: Concatenate all valid charm contents
    std::vector<std::byte> buffer;
    for (int gid = 0; gid < group_sizes.size(); gid++) {
        for (int p = 0; p < 256; p++) {
            for (auto& charm : all_groups_deduped[gid][p]) {
                size_t content_size = charm.data_.size() - 1;
                buffer.insert(buffer.end(), charm.data_.begin(), charm.data_.begin() + content_size);
            }
        }
    }

    // Step 7: Compute SHA1 hash
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char*>(buffer.data()), buffer.size(), hash);

    // Step 8: Output hash as hex string
    std::ostringstream oss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    std::cout << oss.str() << std::endl;

    return 0;
}