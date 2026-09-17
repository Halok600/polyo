/* Peak-live-bytes malloc/free interposer for the C/C++ oracle drivers
 * (plan §6's table: "malloc/free interposer via LD_PRELOAD, peak live
 * bytes"). Built ONCE (see .github/workflows/oracle-label.yml) and
 * LD_PRELOAD-ed for every C/C++ driver run, rather than compiled into each
 * one -- this is what lets it use libdl's RTLD_NEXT to find the real
 * malloc/free instead of needing driver-specific link-time wiring.
 *
 * Linux/glibc only, by design (plan §0: this project's oracle runs on
 * ubuntu-latest in CI precisely because this dev machine has no C
 * toolchain to test this against locally).
 *
 * Two accessor functions below (not standard libc symbols) are what a
 * driver calls directly to read the tracked peak: `polyo_peak_bytes()` and
 * `polyo_reset_peak()`. They resolve at compile+link time as ordinary
 * function calls in the driver binary; LD_PRELOAD makes THIS file's
 * malloc/free/calloc/realloc take priority over libc's for that same
 * process, so the driver's own allocations (and the solution's) are the
 * ones actually being counted.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <malloc.h>
#include <stdatomic.h>
#include <stddef.h>

static void *(*real_malloc)(size_t) = NULL;
static void *(*real_calloc)(size_t, size_t) = NULL;
static void *(*real_realloc)(void *, size_t) = NULL;
static void (*real_free)(void *) = NULL;

static _Atomic long long live_bytes = 0;
static _Atomic long long peak_bytes = 0;

/* `dlsym` itself can call `calloc` before `real_calloc` is resolved (glibc
 * allocates a small buffer for dlerror state on first use) -- the classic
 * bootstrapping hazard of a calloc interposer. A small static arena serves
 * those bootstrap calls only; anything from it is never passed to
 * `real_free`, since it was never `real_malloc`-backed. */
static unsigned char boot_arena[4096];
static size_t boot_used = 0;

static int in_boot_arena(void *ptr) {
    return ptr >= (void *)boot_arena && ptr < (void *)(boot_arena + sizeof(boot_arena));
}

static void ensure_resolved(void) {
    if (real_malloc && real_calloc && real_realloc && real_free) {
        return;
    }
    if (!real_calloc) {
        /* Resolve calloc first: dlsym's own internals may need it, and
         * until it's resolved, our calloc override below serves those
         * calls from the boot arena instead of recursing into dlsym. */
        real_calloc = (void *(*)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
    }
    if (!real_malloc) {
        real_malloc = (void *(*)(size_t))dlsym(RTLD_NEXT, "malloc");
    }
    if (!real_realloc) {
        real_realloc = (void *(*)(void *, size_t))dlsym(RTLD_NEXT, "realloc");
    }
    if (!real_free) {
        real_free = (void (*)(void *))dlsym(RTLD_NEXT, "free");
    }
}

static void track_alloc(long long delta) {
    long long now = atomic_fetch_add(&live_bytes, delta) + delta;
    long long prev_peak = atomic_load(&peak_bytes);
    while (now > prev_peak && !atomic_compare_exchange_weak(&peak_bytes, &prev_peak, now)) {
        /* prev_peak refreshed by the failed CAS itself; retry. */
    }
}

void *malloc(size_t size) {
    ensure_resolved();
    void *p = real_malloc(size);
    if (p != NULL) {
        track_alloc((long long)malloc_usable_size(p));
    }
    return p;
}

void *calloc(size_t nmemb, size_t size) {
    if (!real_calloc) {
        /* Still bootstrapping (see ensure_resolved's comment) -- dlsym
         * itself is what's asking, by way of glibc's internals. */
        size_t total = nmemb * size;
        if (boot_used + total > sizeof(boot_arena)) {
            return NULL;
        }
        void *p = &boot_arena[boot_used];
        boot_used += total;
        return p;
    }
    ensure_resolved();
    void *p = real_calloc(nmemb, size);
    if (p != NULL) {
        track_alloc((long long)malloc_usable_size(p));
    }
    return p;
}

void *realloc(void *ptr, size_t size) {
    ensure_resolved();
    if (ptr != NULL && in_boot_arena(ptr)) {
        /* Never hand a boot-arena pointer to the real allocator. */
        void *p = real_malloc(size);
        if (p != NULL) {
            track_alloc((long long)malloc_usable_size(p));
        }
        return p;
    }
    size_t old_size = ptr != NULL ? malloc_usable_size(ptr) : 0;
    void *p = real_realloc(ptr, size);
    if (p != NULL) {
        track_alloc((long long)malloc_usable_size(p) - (long long)old_size);
    }
    return p;
}

void free(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    if (in_boot_arena(ptr)) {
        return;
    }
    ensure_resolved();
    track_alloc(-(long long)malloc_usable_size(ptr));
    real_free(ptr);
}

long long polyo_peak_bytes(void) {
    return atomic_load(&peak_bytes);
}

void polyo_reset_peak(void) {
    /* Resets the peak to the CURRENT live total, not zero -- baseline
     * allocations already outstanding (e.g. libc/runtime startup) stay
     * subtracted out, matching every other language driver's
     * measure-relative-to-a-baseline approach. */
    atomic_store(&peak_bytes, atomic_load(&live_bytes));
}
