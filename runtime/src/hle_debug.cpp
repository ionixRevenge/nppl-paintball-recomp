// Native versions of the engine's logging/assert helpers, so its messages
// show up in ps2rt.log. They only print; they don't change game behaviour.
#include "rt_internal.h"
#include <string>
#include <cstdarg>

static std::string guest_str(uint32_t a) {
    std::string s;
    if (!a) return "(null)";
    for (uint32_t k = 0; k < 2048; k++) { char ch = (char)LB(a + k); if (!ch) break; s += ch; }
    return s;
}

// Minimal printf over guest varargs: integer args come from a0..t3, then the stack.
static std::string guest_format(R5900 *c, const std::string &fmt, int first_reg) {
    int reg = first_reg;
    uint32_t stack = (uint32_t)c->r[29].ud[0];
    int stack_i = 0;
    auto next = [&]() -> uint64_t {
        if (reg <= 11) return c->r[reg++].ud[0];
        return LD(stack + 8 * (stack_i++));
    };
    std::string out;
    for (size_t i = 0; i < fmt.size(); i++) {
        if (fmt[i] != '%') { out += fmt[i]; continue; }
        size_t j = i + 1;
        std::string spec = "%";
        while (j < fmt.size() && strchr("-+ #0123456789.lhz", fmt[j])) spec += fmt[j++];
        if (j >= fmt.size()) break;
        char conv = fmt[j];
        char buf[512];
        std::string sp;
        for (char ch : spec) if (ch != 'l' && ch != 'h' && ch != 'z') sp += ch;
        switch (conv) {
        case '%': out += '%'; break;
        case 's': snprintf(buf, sizeof buf, (sp + "s").c_str(), guest_str((uint32_t)next()).c_str()); out += buf; break;
        case 'c': out += (char)next(); break;
        case 'd': case 'i': snprintf(buf, sizeof buf, (sp + "lld").c_str(), (long long)(int32_t)next()); out += buf; break;
        case 'u': snprintf(buf, sizeof buf, (sp + "llu").c_str(), (unsigned long long)(uint32_t)next()); out += buf; break;
        case 'x': case 'X': case 'o': snprintf(buf, sizeof buf, (sp + "ll" + conv).c_str(), (unsigned long long)(uint32_t)next()); out += buf; break;
        case 'p': snprintf(buf, sizeof buf, "%08X", (uint32_t)next()); out += buf; break;
        case 'f': case 'g': case 'e': {
            uint64_t v = next(); double d; memcpy(&d, &v, 8);   // doubles are passed in 64-bit GPRs
            snprintf(buf, sizeof buf, (sp + conv).c_str(), d); out += buf; break;
        }
        default: out += spec; out += conv;
        }
        i = j;
    }
    while (!out.empty() && (out.back() == 10 || out.back() == 13)) out.pop_back();
    return out;
}

extern "C" {

// void Log(int channel, const char *fmt, ...)
void hle_game_log(R5900 *c) {
    int ch = (int)(int32_t)c->r[4].ud[0];
    std::string msg = guest_format(c, guest_str((uint32_t)c->r[5].ud[0]), 6);
    rt_log(LOG_INFO, "GAME[%d]: %s", ch, msg.c_str());
}

// void Assert(const char *expr, const char *file, int line, const char *function)
void hle_game_assert(R5900 *c) {
    rt_log(LOG_ERR, "GAME ASSERT: '%s' in file '%s' at line %d in function '%s'",
           guest_str((uint32_t)c->r[4].ud[0]).c_str(), guest_str((uint32_t)c->r[5].ud[0]).c_str(),
           (int)(int32_t)c->r[6].ud[0], guest_str((uint32_t)c->r[7].ud[0]).c_str());
}

} // extern "C"

// printf-style log callable from (temporarily instrumented) recompiled C.
extern "C" void rt_trace(const char *fmt, ...) {
    char b[512]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    rt_log(LOG_INFO, "%s", b);
}
