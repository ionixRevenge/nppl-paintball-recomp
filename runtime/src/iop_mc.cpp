// mcserv (sid 0x80000400) backed by a host folder: memcard/port0 is memory card slot 1.
//
// libmc 2.7 ("XMC" protocol). Requests are one of two structures:
//   desc (48 bytes):   fd, port, slot, size, offset, origin, buffer, param, data[16]
//   name (1044 bytes): port, slot, flags, maxent, table/curdir, name[1024]
// The reply is one word: the function result. Data goes straight into EE memory.
#include "rt_internal.h"
#include "sif.h"
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

enum {
    MC_GETINFO = 1, MC_OPEN = 2, MC_CLOSE = 3, MC_SEEK = 4, MC_READ = 5, MC_WRITE = 6, MC_FLUSH = 10,
    MC_CHDIR = 12, MC_GETDIR = 13, MC_SETINFO = 14, MC_DELETE = 15, MC_FORMAT = 16, MC_UNFORMAT = 17,
    MC_GETENTSPACE = 18, MC_INIT = 254
};
enum { RES_NOENTRY = -4, RES_DENIED = -5, RES_NOTEMPTY = -6, RES_UPLIMIT = -7, RES_NOCARD = -10 };
enum { O_RD = 1, O_WR = 2, O_CREAT = 0x200, O_TRUNC = 0x400, O_DIR = 0x40 };

static fs::path g_root;               // memcard/port0
static std::string g_cwd = "/";
static bool g_seen_card;
struct McFile { FILE *f; std::string path; };
static std::map<int, McFile> g_files;
static int g_next_fd = 1;

struct DirEnt { std::string name; bool dir; uint32_t size; fs::file_time_type mtime; };
static std::vector<DirEnt> g_dir_pending;   // GetDir continuation

static uint8_t *ee_ptr(uint32_t a) { return g_ram + (a & 0x01FFFFFF); }

// Card path -> host path. Handles absolute and cwd-relative paths, "." and "..".
static std::string card_path(const char *name) {
    std::string in = name, base = (in.size() && in[0] == '/') ? "" : g_cwd;
    std::vector<std::string> parts;
    std::string full = base + "/" + in, cur;
    for (size_t i = 0; i <= full.size(); i++) {
        if (i == full.size() || full[i] == '/') {
            if (cur == "..") { if (!parts.empty()) parts.pop_back(); }
            else if (!cur.empty() && cur != ".") parts.push_back(cur);
            cur.clear();
        } else cur += full[i];
    }
    std::string out;
    for (auto &p : parts) out += "/" + p;
    return out.empty() ? "/" : out;
}
static fs::path host_path(const std::string &cp) { return g_root / fs::path(cp.substr(1)); }

static bool wild_match(const char *p, const char *s) {
    if (!*p) return !*s;
    if (*p == '*') return wild_match(p + 1, s) || (*s && wild_match(p, s + 1));
    if (*s && (*p == '?' || *p == *s)) return wild_match(p + 1, s + 1);
    return false;
}

static void put_date(uint8_t *d, fs::file_time_type t) {
    auto sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(t - fs::file_time_type::clock::now() +
                                                                                   std::chrono::system_clock::now());
    time_t tt = std::chrono::system_clock::to_time_t(sys);
    struct tm lt; localtime_s(&lt, &tt);
    d[0] = 0; d[1] = (uint8_t)lt.tm_sec; d[2] = (uint8_t)lt.tm_min; d[3] = (uint8_t)lt.tm_hour;
    d[4] = (uint8_t)lt.tm_mday; d[5] = (uint8_t)(lt.tm_mon + 1);
    uint16_t y = (uint16_t)(lt.tm_year + 1900); memcpy(d + 6, &y, 2);
}

// sceMcTblGetDir: create[8] modify[8] size attr(u16) resv1(u16) resv2 pdaAplNo name[32]
static void put_tbl(uint8_t *t, const DirEnt &e) {
    memset(t, 0, 64);
    put_date(t, e.mtime); put_date(t + 8, e.mtime);
    memcpy(t + 16, &e.size, 4);
    uint16_t attr = e.dir ? 0x8427 : 0x8497;
    memcpy(t + 20, &attr, 2);
    strncpy((char *)t + 32, e.name.c_str(), 31);
}

static int do_getdir(const char *name, int flags, int maxent, uint32_t tbl) {
    if (flags == 0) {
        g_dir_pending.clear();
        std::string cp = card_path(name);
        size_t slash = cp.rfind('/');
        std::string dir = cp.substr(0, slash), pat = cp.substr(slash + 1);
        if (dir.empty()) dir = "/";
        fs::path hd = host_path(dir);
        std::error_code ec;
        if (!fs::is_directory(hd, ec)) return RES_NOENTRY;
        if (dir != "/") {   // real cards list "." and ".." in subdirectories
            auto t = fs::last_write_time(hd, ec);
            if (wild_match(pat.c_str(), ".")) g_dir_pending.push_back({".", true, 0, t});
            if (wild_match(pat.c_str(), "..")) g_dir_pending.push_back({"..", true, 0, t});
        }
        for (auto &de : fs::directory_iterator(hd, ec)) {
            std::string n = de.path().filename().string();
            if (!wild_match(pat.c_str(), n.c_str())) continue;
            bool d = de.is_directory(ec);
            g_dir_pending.push_back({n, d, d ? 0u : (uint32_t)de.file_size(ec), de.last_write_time(ec)});
        }
    }
    if (maxent < 0) {   // no table: just count the matches (the game checks a file exists after writing it)
        int total = (int)g_dir_pending.size();
        g_dir_pending.clear();
        return total;
    }
    int n = std::min<int>(maxent, (int)g_dir_pending.size());
    for (int i = 0; i < n; i++) put_tbl(ee_ptr(tbl + i * 64), g_dir_pending[i]);
    g_dir_pending.erase(g_dir_pending.begin(), g_dir_pending.begin() + n);
    return n;
}

static int do_open(const char *name, int flags) {
    std::string cp = card_path(name);
    fs::path hp = host_path(cp);
    std::error_code ec;
    if (flags & O_DIR) {   // sceMcMkdir
        if (fs::exists(hp, ec)) return RES_NOENTRY;
        return fs::create_directories(hp, ec) ? 0 : RES_NOENTRY;
    }
    if (fs::is_directory(hp, ec)) return RES_DENIED;
    bool exists = fs::exists(hp, ec);
    if (!exists && !(flags & O_CREAT)) return RES_NOENTRY;
    if (!fs::is_directory(hp.parent_path(), ec)) return RES_NOENTRY;
    if (g_files.size() >= 32) return RES_UPLIMIT;
    const char *mode = (!exists || (flags & O_TRUNC)) ? "w+b" : ((flags & O_WR) ? "r+b" : "rb");
    FILE *f = fopen(hp.string().c_str(), mode);
    if (!f) return RES_DENIED;
    int fd = g_next_fd++;
    g_files[fd] = {f, cp};
    return fd;
}

static int do_delete(const char *name) {
    fs::path hp = host_path(card_path(name));
    std::error_code ec;
    if (!fs::exists(hp, ec)) return RES_NOENTRY;
    if (fs::is_directory(hp, ec) && !fs::is_empty(hp, ec)) return RES_NOTEMPTY;
    return fs::remove(hp, ec) ? 0 : RES_DENIED;
}

static void srv_mc(SifRpcCall &c) {
    const uint8_t *in = g_iop_ram + (c.buff & 0x1FFFFF);
    int32_t w[12]; memcpy(w, in, sizeof w);
    const char *name = (const char *)in + 20;
    int32_t res = 0;
    switch (c.fno) {
    case MC_INIT: {
        uint32_t v[3] = {0, 0x20A, 0x20E};   // result, mcserv and mcman versions (libmc needs >= 0x20A/0x20E)
        if (c.recv_addr) sif_write_ee(c.recv_addr, v, std::min<uint32_t>(c.recv_size, 12));
        return;
    }
    case MC_GETINFO: {   // desc: port w1, slot w2, want type w3 / free w4 / format w5, param w7
        if (w[1] != 0) { res = RES_NOCARD; break; }
        uint8_t *p = ee_ptr((uint32_t)w[7]);
        int32_t type = 2, free_kb = 8000, format = 1;
        memset(p, 0, 192);
        memcpy(p + 0, &type, 4); memcpy(p + 4, &free_kb, 4); memcpy(p + 144, &format, 4);
        res = g_seen_card ? 0 : -1;   // -1 the first time: "new formatted card inserted"
        g_seen_card = true;
        break;
    }
    case MC_OPEN: res = (w[0] == 0) ? do_open(name, w[2]) : RES_NOCARD; break;
    case MC_CLOSE: {
        auto it = g_files.find(w[0]);
        if (it == g_files.end()) { res = RES_NOENTRY; break; }
        fclose(it->second.f); g_files.erase(it);
        break;
    }
    case MC_SEEK: {      // fd w0, offset w4, origin w5
        auto it = g_files.find(w[0]);
        if (it == g_files.end()) { res = RES_NOENTRY; break; }
        fseek(it->second.f, w[4], w[5] == 1 ? SEEK_CUR : w[5] == 2 ? SEEK_END : SEEK_SET);
        res = (int32_t)ftell(it->second.f);
        break;
    }
    case MC_READ: {      // fd w0, size w3, buffer w6, param w7
        auto it = g_files.find(w[0]);
        memset(ee_ptr((uint32_t)w[7]), 0, 16);   // no unaligned head/tail for libmc to fix up
        if (it == g_files.end()) { res = RES_NOENTRY; break; }
        res = (int32_t)fread(ee_ptr((uint32_t)w[6]), 1, (size_t)std::max(0, w[3]), it->second.f);
        break;
    }
    case MC_WRITE: {     // fd w0; origin w5 bytes from data[] (+32), then size w3 bytes from buffer w6
        auto it = g_files.find(w[0]);
        if (it == g_files.end()) { res = RES_NOENTRY; break; }
        FILE *f = it->second.f;
        size_t n = fwrite(in + 32, 1, (size_t)std::clamp(w[5], 0, 16), f);
        if (w[3] > 0) n += fwrite(ee_ptr((uint32_t)w[6]), 1, (size_t)w[3], f);
        fflush(f);
        res = (int32_t)n;
        break;
    }
    case MC_FLUSH: {
        auto it = g_files.find(w[0]);
        if (it != g_files.end()) fflush(it->second.f);
        break;
    }
    case MC_CHDIR: {     // name = new dir ("" keeps it); old dir -> EE buffer at w4
        if (w[4]) { char *o = (char *)ee_ptr((uint32_t)w[4]); strncpy(o, g_cwd.c_str(), 1023); o[1023] = 0; }
        if (name[0]) {
            std::string cp = card_path(name);
            std::error_code ec;
            if (!fs::is_directory(host_path(cp), ec)) { res = RES_NOENTRY; break; }
            g_cwd = cp;
        }
        break;
    }
    case MC_GETDIR: res = (w[0] == 0) ? do_getdir(name, w[2], w[3], (uint32_t)w[4]) : RES_NOCARD; break;
    case MC_SETINFO: {   // flags w2, table w4; only rename (flag 0x10) matters on a host folder
        std::string cp = card_path(name);
        std::error_code ec;
        if (!fs::exists(host_path(cp), ec)) { res = RES_NOENTRY; break; }
        if (w[2] & 0x10) {
            const char *nn = (const char *)ee_ptr((uint32_t)w[4]) + 32;
            fs::path to = host_path(cp).parent_path() / nn;
            fs::rename(host_path(cp), to, ec);
            if (ec) res = RES_DENIED;
        }
        break;
    }
    case MC_DELETE: res = do_delete(name); break;
    case MC_FORMAT: case MC_UNFORMAT: break;
    case MC_GETENTSPACE: res = 0x20; break;
    default:
        if (rt_once(0xF30000000ull | c.fno)) rt_log(LOG_INFO, "mc: unhandled fn %u (%08X %08X %08X %08X)", c.fno, w[0], w[1], w[2], w[3]);
    }
    if (c.fno != MC_GETINFO || rt_once(0xF31000000ull))
        rt_log(LOG_INFO, "mc: fn %u %s -> %d", c.fno,
               (c.fno == MC_OPEN || c.fno == MC_GETDIR || c.fno == MC_CHDIR || c.fno == MC_DELETE || c.fno == MC_SETINFO) ? name : "", res);
    if (c.recv_addr) sif_write_ee(c.recv_addr, &res, std::min<uint32_t>(c.recv_size, 4));
}

void iop_mc_init() {
    g_root = fs::path("memcard") / "port0";
    std::error_code ec;
    fs::create_directories(g_root, ec);
    sif_register_server(0x80000400, "mc", srv_mc);
}
