/* LD_PRELOAD-able interposer that adds flags to every
 * clang::CompilerInvocation produced inside cslc-backend.
 *
 * It is built with the host compiler (see CMakeLists.txt) and loaded into
 * cslc-backend inside the Cerebras SDK container, whose glibc 2.30 is older
 * than the host's. Every glibc symbol it uses must therefore exist in glibc
 * 2.30; the build checks this.
 *
 * Use:
 *   APPTAINERENV_LD_PRELOAD=/path/to/libinject_mllvm.so cslc ...
 *
 * Set CSL_INJECT_MLLVM_VERBOSE to report in which processes the hook is
 * active.
 *
 * WARNING: the mangled symbol below is tied to the libclang ABI of the
 * Clang shipped with the Cerebras SDK. If the SDK is rebuilt against a
 * different Clang, re-derive it with
 *   nm -D libclangFrontendTool.so.* | grep CreateFromArgs
 * and update it here.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Since glibc 2.34, dlsym is defined in libc under the version GLIBC_2.34.
 * Bind it to its original version, which every glibc provides. */
#if defined(__GLIBC__) && defined(__x86_64__)
__asm__(".symver dlsym,dlsym@GLIBC_2.2.5");
#endif

/* Edit this list to whatever you want appended to every Clang invocation. */
static const char *const kExtraFlags[] = {
    "-fstack-size-section",
};
enum { kExtraFlagsCount = sizeof(kExtraFlags) / sizeof(kExtraFlags[0]) };

typedef bool (*create_from_args_t)(void *res, const char *const *argv,
                                   size_t argc, void *diags, const char *argv0);

static create_from_args_t real_create_from_args = NULL;
static int hook_active = 0;

__attribute__((constructor)) static void init(void) {
  /* Mangled name of:
   *   clang::CompilerInvocation::CreateFromArgs(
   *       CompilerInvocation&, llvm::ArrayRef<const char*>,
   *       DiagnosticsEngine&, const char*)
   */
  const char *sym = "_ZN5clang18CompilerInvocation14CreateFromArgsERS0_"
                    "N4llvm8ArrayRefIPKcEERNS_17DiagnosticsEngineES5_";
  real_create_from_args = (create_from_args_t)dlsym(RTLD_NEXT, sym);
  if (real_create_from_args) {
    hook_active = 1;
    if (!getenv("CSL_INJECT_MLLVM_VERBOSE"))
      return;
    fprintf(stderr, "[inject_mllvm] hook active in pid %d; will append:",
            (int)getpid());
    for (int i = 0; i < kExtraFlagsCount; ++i)
      fprintf(stderr, " %s", kExtraFlags[i]);
    fputc('\n', stderr);
  }
  /* If the symbol does not resolve, the LD_PRELOAD lib was loaded into a
   * process that does not contain Clang (e.g. cslc-driver itself, or a
   * subprocess). In that case our exported override is never reached
   * because nothing inside that process calls it. */
}

bool _ZN5clang18CompilerInvocation14CreateFromArgsERS0_N4llvm8ArrayRefIPKcEERNS_17DiagnosticsEngineES5_(
    void *res, const char *const *argv, size_t argc, void *diags,
    const char *argv0) {
  if (!hook_active)
    return false; /* defensive; should be unreachable */

  size_t new_argc = argc + kExtraFlagsCount;
  const char **new_argv =
      (const char **)malloc(new_argc * sizeof(const char *));
  if (!new_argv)
    return real_create_from_args(res, argv, argc, diags, argv0);

  memcpy(new_argv, argv, argc * sizeof(const char *));
  for (int i = 0; i < kExtraFlagsCount; ++i)
    new_argv[argc + i] = kExtraFlags[i];

  bool ok = real_create_from_args(res, new_argv, new_argc, diags, argv0);
  free(new_argv);
  return ok;
}
