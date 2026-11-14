// RUN: %clang_cc1 -triple riscv64 -std=c++11 -O0 -verify=expected,cxx %s
// RUN: %clang_cc1 -triple riscv64 -std=c++11 -O2 -verify=expected,cxx %s
// RUN: %clang_cc1 -triple riscv64 -x c -std=c11 -verify=expected,c %s

#define DISTANCE (2 * 4)
#define ENABLE enable
enum { QueueDepth = 64 };

void valid(int *a, int n) {
#pragma clang loop riscv_ns_swp(enable)
#pragma clang loop riscv_ns_swp_distance(1)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_swp(disable)
  while (n > 0) a[--n]++;
#pragma clang loop riscv_ns_ir_fission(enable)
#pragma clang loop riscv_ns_swp_max_loop_size(1)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_ir_fission(disable)
  while (n > 0) a[--n]++;
#pragma clang loop riscv_ns_machine_fission(enable)
#pragma clang loop riscv_ns_swp_max_loop_size(1)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_machine_fission(disable)
  while (n > 0) a[--n]++;
#pragma clang loop riscv_ns_force(enable)
#pragma clang loop riscv_ns_swp_max_loop_size(1)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_force(disable)
  while (n > 0) a[--n]++;
#pragma clang loop riscv_ns_swp_speculative(enable)
#pragma clang loop riscv_ns_swp_max_loop_size(1)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_swp_speculative(disable)
  while (n > 0) a[--n]++;
#pragma clang loop riscv_ns_swp_shift_recurrences(enable)
#pragma clang loop riscv_ns_swp_max_loop_size(1)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_swp_shift_recurrences(disable)
  while (n > 0) a[--n]++;
#pragma clang loop riscv_ns_swp_toggles(enable)
#pragma clang loop riscv_ns_swp_max_loop_size(1)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_swp_toggles(disable)
  while (n > 0) a[--n]++;
#pragma clang loop riscv_ns_swp(ENABLE) riscv_ns_swp_distance(DISTANCE)
#pragma clang loop riscv_ns_ir_fission(disable) riscv_ns_machine_fission(disable)
#pragma clang loop riscv_ns_force(disable) unroll(disable) pipeline(disable)
  do { a[n]++; } while (--n > 0);

  // Numeric hints and disabled transformations are independent.
#pragma clang loop riscv_ns_swp(disable) riscv_ns_swp_distance(62)
#pragma clang loop riscv_ns_swp_speculative(disable) riscv_ns_swp_speculative_distance(62)
#pragma clang loop riscv_ns_ir_fission(disable) riscv_ns_machine_fission(disable)
#pragma clang loop riscv_ns_fission_queue_depth(QueueDepth)
#pragma clang loop riscv_ns_swp_max_slice_size(2147483647)
#pragma clang loop riscv_ns_swp_max_loop_size(2147483647)
#pragma clang loop riscv_ns_swp_max_prologue_size(2147483647)
  for (int i = 0; i < n; ++i) a[i]++;
}

void duplicates(int *a, int n) {
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp(enable)' and 'riscv_ns_swp(enable)'}}
#pragma clang loop riscv_ns_swp(enable) riscv_ns_swp(enable)
  for (int i = 0; i < n; ++i) a[i]++;
  // expected-error@+1 {{duplicate directives 'riscv_ns_ir_fission(enable)' and 'riscv_ns_ir_fission(enable)'}}
#pragma clang loop riscv_ns_ir_fission(enable) riscv_ns_ir_fission(enable)
  for (int i = 0; i < n; ++i) a[i]++;
  // expected-error@+1 {{duplicate directives 'riscv_ns_machine_fission(enable)' and 'riscv_ns_machine_fission(enable)'}}
#pragma clang loop riscv_ns_machine_fission(enable) riscv_ns_machine_fission(enable)
  for (int i = 0; i < n; ++i) a[i]++;
  // expected-error@+1 {{duplicate directives 'riscv_ns_force(enable)' and 'riscv_ns_force(enable)'}}
#pragma clang loop riscv_ns_force(enable) riscv_ns_force(enable)
  for (int i = 0; i < n; ++i) a[i]++;
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp_speculative(enable)' and 'riscv_ns_swp_speculative(enable)'}}
#pragma clang loop riscv_ns_swp_speculative(enable) riscv_ns_swp_speculative(enable)
  for (int i = 0; i < n; ++i) a[i]++;
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp_shift_recurrences(enable)' and 'riscv_ns_swp_shift_recurrences(enable)'}}
#pragma clang loop riscv_ns_swp_shift_recurrences(enable) riscv_ns_swp_shift_recurrences(enable)
  for (int i = 0; i < n; ++i) a[i]++;
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp_toggles(enable)' and 'riscv_ns_swp_toggles(enable)'}}
#pragma clang loop riscv_ns_swp_toggles(enable) riscv_ns_swp_toggles(enable)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_swp_distance(1)
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp_distance(1)' and 'riscv_ns_swp_distance(2)'}}
#pragma clang loop riscv_ns_swp_distance(2)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_swp_speculative_distance(1)
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp_speculative_distance(1)' and 'riscv_ns_swp_speculative_distance(2)'}}
#pragma clang loop riscv_ns_swp_speculative_distance(2)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_swp_max_slice_size(1)
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp_max_slice_size(1)' and 'riscv_ns_swp_max_slice_size(2)'}}
#pragma clang loop riscv_ns_swp_max_slice_size(2)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_swp_max_loop_size(1)
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp_max_loop_size(1)' and 'riscv_ns_swp_max_loop_size(2)'}}
#pragma clang loop riscv_ns_swp_max_loop_size(2)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_swp_max_prologue_size(1)
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp_max_prologue_size(1)' and 'riscv_ns_swp_max_prologue_size(2)'}}
#pragma clang loop riscv_ns_swp_max_prologue_size(2)
  for (int i = 0; i < n; ++i) a[i]++;
#pragma clang loop riscv_ns_fission_queue_depth(1)
  // expected-error@+1 {{duplicate directives 'riscv_ns_fission_queue_depth(1)' and 'riscv_ns_fission_queue_depth(2)'}}
#pragma clang loop riscv_ns_fission_queue_depth(2)
  for (int i = 0; i < n; ++i) a[i]++;
  // expected-error@+1 {{duplicate directives 'riscv_ns_swp(disable)' and 'riscv_ns_swp(enable)'}}
#pragma clang loop riscv_ns_swp(disable) riscv_ns_swp(enable)
  for (int i = 0; i < n; ++i) a[i]++;
}

void invalid(int *a, int n) { // cxx-note {{declared here}}
  // expected-error@+1 {{invalid value '0'; must be positive}}
#pragma clang loop riscv_ns_swp_distance(0)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid value 63 for 'riscv_ns_swp_distance'; must be in the range [1, 62]}}
#pragma clang loop riscv_ns_swp_distance(63)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid value '0'; must be positive}}
#pragma clang loop riscv_ns_swp_speculative_distance(0)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid value 63 for 'riscv_ns_swp_speculative_distance'; must be in the range [1, 62]}}
#pragma clang loop riscv_ns_swp_speculative_distance(63)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid value '0'; must be positive}}
#pragma clang loop riscv_ns_swp_max_slice_size(0)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{value '2147483648' is too large}}
#pragma clang loop riscv_ns_swp_max_slice_size(2147483648ULL)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid value '0'; must be positive}}
#pragma clang loop riscv_ns_swp_max_loop_size(0)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{value '2147483648' is too large}}
#pragma clang loop riscv_ns_swp_max_loop_size(2147483648ULL)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid value '0'; must be positive}}
#pragma clang loop riscv_ns_swp_max_prologue_size(0)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{value '2147483648' is too large}}
#pragma clang loop riscv_ns_swp_max_prologue_size(2147483648ULL)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid value '0'; must be positive}}
#pragma clang loop riscv_ns_fission_queue_depth(0)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid value 65 for 'riscv_ns_fission_queue_depth'; must be in the range [1, 64]}}
#pragma clang loop riscv_ns_fission_queue_depth(65)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid value '-1'; must be positive}}
#pragma clang loop riscv_ns_swp_distance(-1)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{value '18446744073709551615' is too large}}
#pragma clang loop riscv_ns_fission_queue_depth(18446744073709551615ULL)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid argument of type 'double'; expected an integer type}}
#pragma clang loop riscv_ns_swp_distance(2.5)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{expression is not an}} cxx-note@+1 {{function parameter 'n' with unknown value cannot be used in a constant expression}}
#pragma clang loop riscv_ns_swp_distance(n)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{missing argument; expected an integer value}}
#pragma clang loop riscv_ns_swp_distance()
  while (n > 0) a[--n]++;
  // expected-error@+1 {{missing argument; expected 'enable' or 'disable'}}
#pragma clang loop riscv_ns_force()
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid argument; expected 'enable' or 'disable'}}
#pragma clang loop riscv_ns_swp(assume_safety)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid argument; expected 'enable' or 'disable'}}
#pragma clang loop riscv_ns_ir_fission(full)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid argument; expected 'enable' or 'disable'}}
#pragma clang loop riscv_ns_machine_fission(true)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid argument; expected 'enable' or 'disable'}}
#pragma clang loop riscv_ns_force(1)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid argument; expected 'enable' or 'disable'}}
#pragma clang loop riscv_ns_swp_speculative(on)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid argument; expected 'enable' or 'disable'}}
#pragma clang loop riscv_ns_swp_shift_recurrences(assume_safety)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid argument; expected 'enable' or 'disable'}}
#pragma clang loop riscv_ns_swp_toggles(full)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{unexpected tokens in 'riscv_ns_swp' loop hint argument}}
#pragma clang loop riscv_ns_swp(enable, disable)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{unexpected tokens in 'riscv_ns_swp_distance' loop hint argument}}
#pragma clang loop riscv_ns_swp_distance(4, 8)
  while (n > 0) a[--n]++;
  // expected-error@+1 {{expected '('}}
#pragma clang loop riscv_ns_swp
  while (n > 0) a[--n]++;
  // expected-error@+1 {{expected ')'}}
#pragma clang loop riscv_ns_swp_distance(4
  while (n > 0) a[--n]++;
  // expected-error@+1 {{invalid option 'riscv_ns_unknown'}}
#pragma clang loop riscv_ns_unknown(enable)
  while (n > 0) a[--n]++;
#pragma clang loop riscv_ns_swp(enable)
  // expected-error@+1 {{expected a for, while, or do-while loop to follow '#pragma clang loop'}}
  a[0]++;
}

#ifdef __cplusplus
template <int Distance> void bad_distance(int *a, int n) {
  // expected-error@+1 {{invalid value 63 for 'riscv_ns_swp_distance'; must be in the range [1, 62]}}
#pragma clang loop riscv_ns_swp_distance(Distance)
  for (int i = 0; i < n; ++i) a[i]++;
}
template <int Depth> void bad_depth(int *a, int n) {
  // expected-error@+1 {{invalid value '0'; must be positive}}
#pragma clang loop riscv_ns_fission_queue_depth(Depth)
  for (int i = 0; i < n; ++i) a[i]++;
}
template <unsigned long long Size> void bad_size(int *a, int n) {
  // expected-error@+1 {{value '2147483648' is too large}}
#pragma clang loop riscv_ns_swp_max_loop_size(Size)
  for (int i = 0; i < n; ++i) a[i]++;
}
template <typename T> void bad_type(int *a, int n) {
  // expected-error@+1 {{invalid argument of type 'bool'; expected an integer type}}
#pragma clang loop riscv_ns_swp_distance(T(1))
  for (int i = 0; i < n; ++i) a[i]++;
}
void instantiate(int *a, int n) {
  bad_distance<63>(a, n); // expected-note {{in instantiation of function template specialization}}
  bad_depth<0>(a, n); // expected-note {{in instantiation of function template specialization}}
  bad_size<2147483648ULL>(a, n); // expected-note {{in instantiation of function template specialization}}
  bad_type<bool>(a, n); // expected-note {{in instantiation of function template specialization}}
}
#endif
