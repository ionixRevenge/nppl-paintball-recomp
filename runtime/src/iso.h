#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct IsoEntry {
    uint32_t lsn = 0, size = 0;
    bool is_dir = false;
    std::string name;
};

bool iso_open(const std::string &file_or_dir);   // a .iso, or a directory containing one
bool iso_lookup(const std::string &path, IsoEntry &out);
std::vector<IsoEntry> iso_list(const IsoEntry &dir);
bool iso_read_bytes(const IsoEntry &e, uint64_t pos, void *dst, size_t n);
bool iso_read_sectors(uint32_t lsn, uint32_t count, void *dst);
