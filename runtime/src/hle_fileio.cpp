// Native replacement for the SCE fileio client (sceOpen/sceRead/... over SIF).
//   cdrom0:  -> files inside the user's ISO (read-only)
//   host0:/host: -> ./host/ next to the executable (devkit host filesystem, e.g. debug.log)
// Results follow the IOP ioman conventions: fd >= 0 or a negative errno.
#include "rt_internal.h"
#include "iso.h"
#include <filesystem>
#include <map>
#include <string>

namespace fs = std::filesystem;

enum { E_NOENT = -2, E_IO = -5, E_BADF = -9, E_NOMEM = -12, E_ACCES = -13, E_NOTDIR = -20, E_INVAL = -22, E_MFILE = -24 };
enum { O_RD = 1, O_WR = 2, O_APPEND = 0x100, O_CREAT = 0x200, O_TRUNC = 0x400 };

struct Fd {
    enum Kind { NONE, ISO_FILE, ISO_DIR, HOST_FILE } kind = NONE;
    IsoEntry ent;
    uint64_t pos = 0;
    FILE *host = nullptr;
    std::vector<IsoEntry> dir;
    size_t dir_i = 0;
    std::string path;
};
static std::map<int, Fd> g_fds;

static uint32_t A(R5900 *c, int n) { return (uint32_t)c->r[4 + n].ud[0]; }
static void RET(R5900 *c, int64_t v) { c->r[2].ud[0] = (uint64_t)v; }

static std::string guest_str(uint32_t a) {
    std::string s;
    for (uint32_t k = 0; k < 1024; k++) { char ch = (char)LB(a + k); if (!ch) break; s += ch; }
    return s;
}

static int alloc_fd() {
    for (int i = 3; i < 128; i++) if (!g_fds.count(i)) return i;
    return E_MFILE;
}

static std::string device_of(const std::string &p) {
    auto colon = p.find(':');
    if (colon == std::string::npos) return "";
    std::string d = p.substr(0, colon);
    for (auto &ch : d) ch = (char)tolower((unsigned char)ch);
    while (!d.empty() && isdigit((unsigned char)d.back())) d.pop_back();
    return d;
}

static fs::path host_path(const std::string &p) {
    std::string rel = p.substr(p.find(':') + 1);
    for (auto &ch : rel) if (ch == '\\') ch = '/';
    while (!rel.empty() && rel[0] == '/') rel.erase(0, 1);
    if (rel.size() > 2 && rel[1] == ':') rel.erase(0, 2);   // "c:/2002/..." devkit paths
    while (!rel.empty() && rel[0] == '/') rel.erase(0, 1);
    return fs::path("host") / rel;
}

// iox_stat_t (64 bytes): mode, attr, size, ctime[8], atime[8], mtime[8], hisize, private[6]
static const uint32_t STAT_SIZE = 64;
static void write_stat(uint32_t p, const IsoEntry &e) {
    for (uint32_t k = 0; k < STAT_SIZE; k++) SB(p + k, 0);
    // iomanX (IOPRP 2.7) mode bits: SCE_STM_FDIR 0x1000 / SCE_STM_FREG 0x2000, r-x for all
    SW(p + 0, e.is_dir ? (0x1000 | 0x16D) : (0x2000 | 0x124));
    SW(p + 8, e.size);
}

extern "C" {

void hle_sceOpen(R5900 *c) {
    std::string path = guest_str(A(c, 0));
    uint32_t flags = A(c, 1);
    std::string dev = device_of(path);
    int fd = alloc_fd();
    if (fd < 0) { RET(c, fd); return; }
    Fd f; f.path = path;
    if (dev == "cdrom" || dev == "") {
        if (flags & O_WR) { RET(c, E_ACCES); return; }
        if (!iso_lookup(path, f.ent) || f.ent.is_dir) {
            rt_log(LOG_WARN, "sceOpen(%s): not found", path.c_str());
            RET(c, E_NOENT); return;
        }
        f.kind = Fd::ISO_FILE;
    } else if (dev == "host") {
        fs::path hp = host_path(path);
        std::error_code ec; fs::create_directories(hp.parent_path(), ec);
        const char *mode = (flags & O_WR) ? ((flags & O_APPEND) ? "ab" : ((flags & O_TRUNC) || (flags & O_CREAT)) ? "wb" : "r+b") : "rb";
        f.host = fopen(hp.string().c_str(), mode);
        if (!f.host) { RET(c, E_NOENT); return; }
        f.kind = Fd::HOST_FILE;
    } else {
        rt_log(LOG_WARN, "sceOpen(%s): unsupported device", path.c_str());
        RET(c, E_NOENT); return;
    }
    rt_log(LOG_INFO, "sceOpen(%s, %X) -> %d", path.c_str(), flags, fd);
    g_fds[fd] = std::move(f);
    RET(c, fd);
}

void hle_sceClose(R5900 *c) {
    int fd = (int)A(c, 0);
    auto it = g_fds.find(fd);
    if (it == g_fds.end()) { RET(c, E_BADF); return; }
    if (it->second.host) fclose(it->second.host);
    g_fds.erase(it);
    RET(c, 0);
}

void hle_sceRead(R5900 *c) {
    int fd = (int)A(c, 0); uint32_t buf = A(c, 1); int32_t n = (int32_t)A(c, 2);
    auto it = g_fds.find(fd);
    if (it == g_fds.end() || n < 0) { RET(c, E_BADF); return; }
    Fd &f = it->second;
    std::vector<uint8_t> tmp(n);
    size_t got = 0;
    if (f.kind == Fd::ISO_FILE) {
        uint64_t left = f.ent.size > f.pos ? f.ent.size - f.pos : 0;
        got = (size_t)std::min<uint64_t>(left, (uint64_t)n);
        if (got && !iso_read_bytes(f.ent, f.pos, tmp.data(), got)) { RET(c, E_IO); return; }
    } else if (f.kind == Fd::HOST_FILE) {
        got = fread(tmp.data(), 1, n, f.host);
    } else { RET(c, E_BADF); return; }
    for (size_t k = 0; k < got; k++) SB(buf + (uint32_t)k, tmp[k]);
    f.pos += got;
    RET(c, (int64_t)got);
}

void hle_sceWrite(R5900 *c) {
    int fd = (int)A(c, 0); uint32_t buf = A(c, 1); int32_t n = (int32_t)A(c, 2);
    if (fd == 1 || fd == 2) {  // tty
        std::string s;
        for (int32_t k = 0; k < n; k++) s += (char)LB(buf + k);
        while (!s.empty() && (s.back() == 10 || s.back() == 13)) s.pop_back();
        rt_log(LOG_INFO, "TTY: %s", s.c_str());
        RET(c, n); return;
    }
    auto it = g_fds.find(fd);
    if (it == g_fds.end() || it->second.kind != Fd::HOST_FILE) { RET(c, E_BADF); return; }
    std::vector<uint8_t> tmp(n);
    for (int32_t k = 0; k < n; k++) tmp[k] = LB(buf + k);
    size_t w = fwrite(tmp.data(), 1, n, it->second.host);
    fflush(it->second.host);
    RET(c, (int64_t)w);
}

void hle_sceLseek(R5900 *c) {
    int fd = (int)A(c, 0); int32_t off = (int32_t)A(c, 1); int whence = (int)A(c, 2);
    auto it = g_fds.find(fd);
    if (it == g_fds.end()) { RET(c, E_BADF); return; }
    Fd &f = it->second;
    if (f.kind == Fd::HOST_FILE) {
        fseek(f.host, off, whence);
        RET(c, ftell(f.host)); return;
    }
    int64_t base = whence == 0 ? 0 : whence == 1 ? (int64_t)f.pos : (int64_t)f.ent.size;
    int64_t np = base + off;
    if (np < 0) { RET(c, E_INVAL); return; }
    f.pos = (uint64_t)np;
    RET(c, np);
}

void hle_sceIoctl(R5900 *c) {
    if (rt_once(0x1200000000ull | A(c, 1))) rt_log(LOG_INFO, "sceIoctl(fd %d, req %X)", A(c, 0), A(c, 1));
    RET(c, 0);
}

void hle_sceDopen(R5900 *c) {
    std::string path = guest_str(A(c, 0));
    int fd = alloc_fd();
    if (fd < 0) { RET(c, fd); return; }
    Fd f; f.path = path;
    if (!iso_lookup(path, f.ent) || !f.ent.is_dir) {
        rt_log(LOG_WARN, "sceDopen(%s): not found", path.c_str());
        RET(c, E_NOENT); return;
    }
    f.kind = Fd::ISO_DIR;
    f.dir = iso_list(f.ent);
    rt_log(LOG_INFO, "sceDopen(%s) -> %d (%zu entries)", path.c_str(), fd, f.dir.size());
    g_fds[fd] = std::move(f);
    RET(c, fd);
}

void hle_sceDread(R5900 *c) {
    int fd = (int)A(c, 0); uint32_t ent = A(c, 1);
    if (rt_once(0x1300000000ull)) rt_log(LOG_INFO, "sceDread(a0=%08X, a1=%08X, a2=%08X)", A(c, 0), A(c, 1), A(c, 2));
    auto it = g_fds.find(fd);
    if (it == g_fds.end() || it->second.kind != Fd::ISO_DIR) { RET(c, E_BADF); return; }
    Fd &f = it->second;
    if (f.dir_i >= f.dir.size()) { RET(c, 0); return; }
    const IsoEntry &e = f.dir[f.dir_i++];
    // iox_dirent_t: iox_stat_t (64 bytes) + name[256] + private
    write_stat(ent, e);
    for (size_t k = 0; k < 256; k++) SB(ent + STAT_SIZE + (uint32_t)k, k < e.name.size() ? (uint8_t)e.name[k] : 0);
    RET(c, (int64_t)e.name.size());
}

void hle_sceDclose(R5900 *c) {
    int fd = (int)A(c, 0);
    if (!g_fds.erase(fd)) { RET(c, E_BADF); return; }
    RET(c, 0);
}

void hle_sceGetstat(R5900 *c) {
    std::string path = guest_str(A(c, 0));
    IsoEntry e;
    if (device_of(path) == "host") {
        std::error_code ec;
        fs::path hp = host_path(path);
        if (!fs::exists(hp, ec)) { RET(c, E_NOENT); return; }
        e.is_dir = fs::is_directory(hp, ec);
        e.size = e.is_dir ? 0 : (uint32_t)fs::file_size(hp, ec);
    } else if (!iso_lookup(path, e)) {
        RET(c, E_NOENT); return;
    }
    write_stat(A(c, 1), e);
    RET(c, 0);
}

} // extern "C"
