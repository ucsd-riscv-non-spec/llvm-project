// Optional simulator verification driver. See ns-docs/verify-loop-metadata.py.
typedef unsigned long u64;
static int a[256];
_Alignas(8) static volatile int out[256];
extern void ir_depth_cutoff(int *, int *);
extern void ir_enabled(int *, int *);
extern void ir_alias(int *, int *);
extern void ir_forced(int *, int *);
extern void machine_enabled(int *, int *);
extern void machine_disabled(int *, int *);
extern void machine_alias(int *, int *);
extern void machine_forced(int *, int *);
extern void pipeline(int *, int *, u64);
extern void siblings(int *, u64);
extern void nested(int *, u64, u64);
extern void forced_load(int *, int *);
extern void unsafe_load(int *, int *);
extern void toggle(int *, u64);
extern u64 shift(u64);

static void reset(unsigned round) {
  for (unsigned i = 0; i < 256; ++i) {
    a[i] = i < 128 ? (int)((i + round) % 11) - 5 : 0;
    out[i] = 0;
  }
}
static void record(u64 extra) {
  u64 hash = extra;
  for (unsigned i = 0; i < 256; ++i)
    hash = hash * 31 + (unsigned)out[i] + (unsigned)a[i];
  register u64 fd asm("a0") = 1;
  register const void *data asm("a1") = &hash;
  register u64 len asm("a2") = sizeof(hash);
  register u64 nr asm("a7") = 64;
  asm volatile("ecall" : "+r"(fd) : "r"(data), "r"(len), "r"(nr) : "memory");
}
#define FIXED(fn)                                                              \
  reset(round);                                                                \
  fn(a, (int *)out);                                                           \
  record(0)
void _start(void) {
  static const u64 sizes[] = {0, 1, 3, 17, 35, 127};
  for (unsigned round = 0; round < 6; ++round) {
    FIXED(ir_depth_cutoff);
    FIXED(ir_enabled);
    FIXED(ir_alias);
    FIXED(ir_forced);
    FIXED(machine_enabled);
    FIXED(machine_disabled);
    FIXED(machine_alias);
    FIXED(machine_forced);
    reset(round);
    pipeline(a, (int *)out, sizes[round]);
    record(0);
    reset(round);
    siblings(a, sizes[round]);
    record(0);
    reset(round);
    nested(a, sizes[round], 2);
    record(0);
    FIXED(forced_load);
    FIXED(unsafe_load);
    reset(round);
    toggle((int *)out, sizes[round]);
    record(shift(0x5a5aUL + round));
  }
  register u64 code asm("a0") = 0;
  register u64 nr asm("a7") = 93;
  asm volatile("ecall" : : "r"(code), "r"(nr) : "memory");
  __builtin_unreachable();
}
