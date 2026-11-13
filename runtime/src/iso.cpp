// ISO9660 reader for the user's own disc image (game_iso/*.iso).
// Serves file lookups for cdrom0: paths and raw sector reads for the CD/DVD library.
#include "iso.h"
#include "rt_internal.h"
#include <algorithm>
#include <cctype>
#include <filesystem>

namespace fs = std::filesystem;

static FILE *g_iso;
static IsoEntry g_root;

static bool read_raw(uint64_t off, void *dst, size_t n) {
    if (!g_iso) return false;
    _fseeki64(g_iso, (long long)off, SEEK_SET);
    return fread(dst, 1, n, g_iso) == n;
}

bool iso_read_sectors(uint32_t lsn, uint32_t count, void *dst) {
    return read_raw((uint64_t)lsn * 2048, dst, (size_t)count * 2048);
}

bool iso_read_bytes(const IsoEntry &e, uint64_t pos, void *dst, size_t n) {
    if (pos + n > e.size) return false;
    return read_raw((uint64_t)e.lsn * 2048 + pos, dst, n);
}

static std::string upper(std::string s) {
    for (auto &ch : s) ch = (char)toupper((unsigned char)ch);
    return s;
}

std::vector<IsoEntry> iso_list(const IsoEntry &dir) {
    std::vector<IsoEntry> out;
    std::vector<uint8_t> d(dir.size);
    if (!read_raw((uint64_t)dir.lsn * 2048, d.data(), d.size())) return out;
    size_t i = 0;
    while (i < d.size()) {
        uint8_t len = d[i];
        if (len == 0) { i = (i / 2048 + 1) * 2048; continue; }
        const uint8_t *r = &d[i];
        uint8_t nlen = r[32];
        std::string name((const char *)r + 33, nlen);
        if (!(nlen == 1 && (name[0] == 0 || name[0] == 1))) {
            IsoEntry e;
            memcpy(&e.lsn, r + 2, 4);
            memcpy(&e.size, r + 10, 4);
            e.is_dir = (r[25] & 2) != 0;
            e.name = name;
            auto semi = e.name.find(';');
            if (semi != std::string::npos) e.name.resize(semi);
            if (!e.name.empty() && e.name.back() == '.') e.name.pop_back();
            out.push_back(e);
        }
        i += len;
    }
    return out;
}

// path: "cdrom0:\DATA\FILE.EXT;1", "\DATA\FILE.EXT", "DATA/FILE.EXT"... (case-insensitive)
bool iso_lookup(const std::string &path_in, IsoEntry &out) {
    if (!g_iso) return false;
    std::string p = path_in;
    auto colon = p.find(':');
    if (colon != std::string::npos) p = p.substr(colon + 1);
    auto semi = p.find(';');
    if (semi != std::string::npos) p.resize(semi);
    for (auto &ch : p) if (ch == '\\') ch = '/';
    IsoEntry cur = g_root;
    size_t i = 0;
    while (i < p.size()) {
        while (i < p.size() && p[i] == '/') i++;
        if (i >= p.size()) break;
        size_t j = p.find('/', i);
        std::string part = upper(p.substr(i, j == std::string::npos ? std::string::npos : j - i));
        i = j == std::string::npos ? p.size() : j;
        if (!cur.is_dir) return false;
        bool found = false;
        for (auto &e : iso_list(cur))
            if (upper(e.name) == part) { cur = e; found = true; break; }
        if (!found) return false;
    }
    out = cur;
    return true;
}

bool iso_open(const std::string &where) {
    std::string path = where;
    std::error_code ec;
    if (fs::is_directory(where, ec)) {
        path.clear();
        for (auto &f : fs::directory_iterator(where, ec)) {
            auto ext = f.path().extension().string();
            for (auto &ch : ext) ch = (char)tolower((unsigned char)ch);
            if (ext == ".iso") { path = f.path().string(); break; }
        }
        if (path.empty()) { rt_log(LOG_ERR, "no .iso found in %s", where.c_str()); return false; }
    }
    g_iso = fopen(path.c_str(), "rb");
    if (!g_iso) { rt_log(LOG_ERR, "cannot open %s", path.c_str()); return false; }
    uint8_t pvd[2048];
    if (!read_raw(16 * 2048, pvd, sizeof pvd) || memcmp(pvd + 1, "CD001", 5)) {
        rt_log(LOG_ERR, "%s is not an ISO9660 image", path.c_str());
        fclose(g_iso); g_iso = nullptr; return false;
    }
    memcpy(&g_root.lsn, pvd + 156 + 2, 4);
    memcpy(&g_root.size, pvd + 156 + 10, 4);
    g_root.is_dir = true;
    g_root.name = "";
    rt_log(LOG_INFO, "disc image: %s (%.32s)", path.c_str(), (const char *)pvd + 40);
    return true;
}
