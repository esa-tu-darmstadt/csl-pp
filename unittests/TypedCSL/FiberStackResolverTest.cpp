//===- FiberStackResolverTest.cpp - Unit tests for the stack resolver -----===//
//
// Tests the stack reach the resolver computes from a captured call graph and
// measured frame sizes, in particular for bounded recursion.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberStackResolver.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

/// Bytes of the arg-area at the top of each root task's stack.
constexpr uint64_t kArgAreaBytes = kArgAreaU16Slots * 2;

/// The u32-word count of a root task stack whose frames reach `bytes`.
uint32_t rootWords(uint64_t bytes) {
  return static_cast<uint32_t>((bytes + kArgAreaBytes + 3) / 4);
}

/// Builds a captured call graph of out-of-line functions with known frame
/// sizes.
struct GraphBuilder {
  CapturedCallGraph cg;
  llvm::StringMap<uint32_t> stackSizes;
  DwarfInlineInfo inlineInfo;

  FuncCallInfo &func(llvm::StringRef name, uint32_t frameBytes) {
    stackSizes[name] = frameBytes;
    inlineInfo.survivingSubprograms.insert(name);
    return cg.funcs[name];
  }

  FuncCallInfo &root(llvm::StringRef name, uint32_t frameBytes) {
    cg.rootTasks.insert(name);
    return func(name, frameBytes);
  }

  FailureOr<ResolvedSizes> resolve() {
    return resolveFiberSizes(stackSizes, inlineInfo, cg, "test");
  }
};

TEST(FiberStackResolverTest, SelfRecursion) {
  GraphBuilder g;
  g.root("root", 10).syncCallees = {"fact"};
  FuncCallInfo &fact = g.func("fact", 20);
  fact.maxRecursionDepth = 4;
  fact.syncCallees = {"fact"};

  FailureOr<ResolvedSizes> sizes = g.resolve();
  ASSERT_TRUE(succeeded(sizes));
  // Four activations of fact below the root frame.
  EXPECT_EQ(sizes->rootStackU32Words.lookup("root"), rootWords(10 + 4 * 20));
}

TEST(FiberStackResolverTest, MutualRecursion) {
  GraphBuilder g;
  g.root("root", 10).syncCallees = {"a"};
  FuncCallInfo &a = g.func("a", 10);
  a.maxRecursionDepth = 3;
  a.syncCallees = {"b"};
  g.func("b", 6).syncCallees = {"a"};

  FailureOr<ResolvedSizes> sizes = g.resolve();
  ASSERT_TRUE(succeeded(sizes));
  // a -> b -> a -> b -> a -> b: three activations of a, each followed by b.
  EXPECT_EQ(sizes->rootStackU32Words.lookup("root"),
            rootWords(10 + 3 * (10 + 6)));
}

TEST(FiberStackResolverTest, EntryThroughUnannotatedFunction) {
  GraphBuilder g;
  g.root("root", 10).syncCallees = {"b"};
  FuncCallInfo &a = g.func("a", 10);
  a.maxRecursionDepth = 2;
  a.syncCallees = {"b"};
  g.func("b", 6).syncCallees = {"a"};

  FailureOr<ResolvedSizes> sizes = g.resolve();
  ASSERT_TRUE(succeeded(sizes));
  // b -> a -> b -> a -> b: two activations of a, one more of b.
  EXPECT_EQ(sizes->rootStackU32Words.lookup("root"),
            rootWords(10 + 3 * 6 + 2 * 10));
}

TEST(FiberStackResolverTest, BudgetSharedByAnnotatedFunctions) {
  GraphBuilder g;
  g.root("root", 10).syncCallees = {"a"};
  FuncCallInfo &a = g.func("a", 10);
  a.maxRecursionDepth = 1;
  a.syncCallees = {"b"};
  FuncCallInfo &b = g.func("b", 6);
  b.maxRecursionDepth = 1;
  b.syncCallees = {"a"};

  FailureOr<ResolvedSizes> sizes = g.resolve();
  ASSERT_TRUE(succeeded(sizes));
  EXPECT_EQ(sizes->rootStackU32Words.lookup("root"), rootWords(10 + 10 + 6));
}

TEST(FiberStackResolverTest, RecursionAroundParallelSite) {
  GraphBuilder g;
  g.root("root", 10).syncCallees = {"rec"};
  FuncCallInfo &rec = g.func("rec", 8);
  rec.maxRecursionDepth = 2;
  rec.syncCallees = {"rec"};
  rec.parallelSites = {{"leaf_a", "leaf_b"}};
  g.func("leaf_a", 4);
  g.func("leaf_b", 6);
  g.stackSizes[kCallerJoinRuntimeSym] = 2;

  FailureOr<ResolvedSizes> sizes = g.resolve();
  ASSERT_TRUE(succeeded(sizes));
  // The parallel site reaches max(caller_join, leaf_a) + arg-area + leaf_b
  // below each activation of rec; the deepest activation contains it.
  uint64_t site = std::max<uint64_t>(2, 4) + kArgAreaBytes + 6;
  EXPECT_EQ(sizes->rootStackU32Words.lookup("root"),
            rootWords(10 + 2 * 8 + site));
}

TEST(FiberStackResolverTest, RecursiveParallelCallee) {
  GraphBuilder g;
  g.root("root", 10).parallelSites = {{"leaf", "fact"}};
  g.func("leaf", 4);
  FuncCallInfo &fact = g.func("fact", 20);
  fact.maxRecursionDepth = 4;
  fact.syncCallees = {"fact"};

  FailureOr<ResolvedSizes> sizes = g.resolve();
  ASSERT_TRUE(succeeded(sizes));
  // The worker region of fact holds all four activations.
  EXPECT_EQ(sizes->calleeSizeWords.lookup("fact"), (4 * 20) / 2);
  EXPECT_EQ(sizes->rootStackU32Words.lookup("root"),
            rootWords(10 + 4 + kArgAreaBytes + 4 * 20));
}

TEST(FiberStackResolverTest, UnannotatedCycleFails) {
  GraphBuilder g;
  g.root("root", 10).syncCallees = {"a"};
  g.func("a", 10).syncCallees = {"b"};
  g.func("b", 6).syncCallees = {"a"};

  EXPECT_TRUE(failed(g.resolve()));
}

TEST(FiberStackResolverTest, StackSizeBelowRecursiveCalleeFails) {
  GraphBuilder g;
  g.root("root", 10).syncCallees = {"x"};
  FuncCallInfo &x = g.func("x", 10);
  x.stackSizeOverride = 50;
  x.syncCallees = {"fact"};
  FuncCallInfo &fact = g.func("fact", 20);
  fact.maxRecursionDepth = 4;
  fact.syncCallees = {"fact"};

  // The body of x needs 10 + 4 * 20 bytes.
  EXPECT_TRUE(failed(g.resolve()));
}

TEST(FiberStackResolverTest, StackSizeBreaksCycle) {
  GraphBuilder g;
  g.root("root", 10).syncCallees = {"a"};
  FuncCallInfo &a = g.func("a", 10);
  a.stackSizeOverride = 100;
  a.syncCallees = {"b"};
  g.func("b", 6).syncCallees = {"a"};

  FailureOr<ResolvedSizes> sizes = g.resolve();
  ASSERT_TRUE(succeeded(sizes));
  EXPECT_EQ(sizes->rootStackU32Words.lookup("root"), rootWords(10 + 100));
}

} // namespace
