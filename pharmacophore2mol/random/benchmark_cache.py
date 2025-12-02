import time
import functools
import uuid
from functools import lru_cache

# ==========================================
# STRATEGY 1: LRU CACHE
# ==========================================
class LRUNode:
    def __init__(self, parents=None, name=""):
        self.parents = parents if parents else []
        self.name = name

    # We use lru_cache. 
    # CRITICAL: We must include 'call_id' in arguments to differentiate 
    # between "fetching index 0 now" and "fetching index 0 in the next epoch".
    @lru_cache(maxsize=16)
    def get(self, index, call_id):
        # Recursion
        val = 0
        for p in self.parents:
            val += p.get(index, call_id)
        
        # Simulate tiny work
        return val + 1

# ==========================================
# STRATEGY 2: CONTEXT DICT
# ==========================================
class ContextNode:
    def __init__(self, parents=None, name=""):
        self.parents = parents if parents else []
        self.name = name

    def get(self, index, context):
        # 1. Check Cache
        key = (id(self), index)
        if key in context:
            return context[key]

        # 2. Recursion
        val = 0
        for p in self.parents:
            val += p.get(index, context)
        
        # 3. Update Cache
        res = val + 1
        context[key] = res
        return res

# ==========================================
# SETUP THE DIAMOND GRAPH
# ==========================================
# Structure: Source -> (Branch1, Branch2) -> Merge
# This forces the caching mechanism to handle the "Source" access twice.

def build_lru_graph():
    source = LRUNode(name="Source")
    b1 = LRUNode([source], name="B1")
    b2 = LRUNode([source], name="B2")
    merge = LRUNode([b1, b2], name="Merge")
    return merge

def build_ctx_graph():
    source = ContextNode(name="Source")
    b1 = ContextNode([source], name="B1")
    b2 = ContextNode([source], name="B2")
    merge = ContextNode([b1, b2], name="Merge")
    return merge

# ==========================================
# BENCHMARK
# ==========================================
ITERATIONS = 500_000

print(f"--- Benchmarking {ITERATIONS} iterations ---")

# --- TEST LRU ---
merge_lru = build_lru_graph()
start = time.perf_counter()

for i in range(ITERATIONS):
    # We must pass 'i' (or a uuid) as call_id.
    # If we don't, LRU returns the cached result from iteration 0 forever (incorrect behavior).
    # This forces LRU to handle a "Cache Miss + Eviction" every time.
    _ = merge_lru.get(0, call_id=i)

lru_time = time.perf_counter() - start
print(f"LRU Cache:      {lru_time:.4f} seconds")


# --- TEST CONTEXT ---
merge_ctx = build_ctx_graph()
start = time.perf_counter()

for i in range(ITERATIONS):
    # We create a fresh dict every call.
    # This simulates the __getitem__ entry point.
    ctx = {}
    _ = merge_ctx.get(0, context=ctx)

ctx_time = time.perf_counter() - start
print(f"Context Dict:   {ctx_time:.4f} seconds")

# --- RESULT ---
ratio = lru_time / ctx_time
print(f"\nWinner: {'Context Dict' if ratio > 1 else 'LRU Cache'}")
print(f"Speedup: {ratio:.2f}x faster")