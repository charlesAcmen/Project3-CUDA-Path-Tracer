#pragma once

// ====================================================================
// Lightweight tagged logging.
//
// Usage:  Log::info("Config",  "compactMethod=%d", v);
//         Log::warn("Scene",   "Unknown type: %s", t);
//         Log::error("Mesh",   "Failed: %s", p);
//
// Output: info → stdout,  warn/error → stderr
// ====================================================================

#include <cstdio>
#include <cstdarg>

namespace Log {

// ---- Raw output (stdout, no tag or newline) ----
// For structured display that shouldn't have a [tag] prefix
// (startup summary, help text, etc.).
inline void raw(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

// ---- Info (stdout) ----
inline void info(const char* tag, const char* fmt, ...) {
    printf("[%s] ", tag);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

inline void warn(const char* tag, const char* fmt, ...) {
    fprintf(stderr, "[%s] ", tag);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

inline void error(const char* tag, const char* fmt, ...) {
    fprintf(stderr, "[%s] ERROR: ", tag);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

} // namespace Log
