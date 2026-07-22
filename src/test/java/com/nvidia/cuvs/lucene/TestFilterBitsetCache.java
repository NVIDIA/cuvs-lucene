/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.nvidia.cuvs.FilterBitsetHandle;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;

/**
 * Concurrency tests for {@link FilterBitsetCache}. These exercise the compute-once and
 * reference-counting behavior in isolation by injecting a fake {@link FilterBitsetHandle}, so no GPU
 * or native library is required.
 */
public class TestFilterBitsetCache {

  /**
   * Fake handle faithfully modeling the {@link FilterBitsetHandle} reference-counting contract, so
   * the cache's use of it can be observed. {@link #decRef()} throws if released more times than
   * referenced, turning any over-release by the cache into a test failure.
   */
  private static final class CountingHandle implements FilterBitsetHandle {
    final AtomicInteger refCount = new AtomicInteger(1); // initial reference, as in the real handle
    final AtomicInteger closeCalls = new AtomicInteger();
    final AtomicBoolean freed = new AtomicBoolean();
    private final AtomicBoolean initialReleased = new AtomicBoolean();

    @Override
    public boolean tryIncRef() {
      int c;
      do {
        c = refCount.get();
        if (c == 0) return false;
      } while (!refCount.compareAndSet(c, c + 1));
      return true;
    }

    @Override
    public void decRef() {
      int c = refCount.decrementAndGet();
      if (c == 0) {
        freed.set(true);
      } else if (c < 0) {
        throw new IllegalStateException("decRef() called more times than references were taken");
      }
    }

    @Override
    public void close() {
      closeCalls.incrementAndGet();
      if (initialReleased.compareAndSet(false, true)) {
        decRef();
      }
    }
  }

  private static Object segKey(String s) {
    return s;
  }

  /** Concurrent misses on the same key build the handle exactly once and each get their own ref. */
  @Test
  public void computeOnceUnderConcurrentAcquire() throws Exception {
    final int threads = 32;
    AtomicInteger buildCount = new AtomicInteger();
    CountingHandle handle = new CountingHandle();
    final String field = "computeOnce";

    ExecutorService pool = Executors.newFixedThreadPool(threads);
    CountDownLatch start = new CountDownLatch(1);
    List<Future<FilterBitsetHandle>> results = new ArrayList<>();
    try {
      for (int i = 0; i < threads; i++) {
        results.add(
            pool.submit(
                () -> {
                  start.await();
                  return FilterBitsetCache.acquire(
                      null,
                      segKey(field),
                      field,
                      () -> {
                        buildCount.incrementAndGet();
                        // Widen the race window so waiters reach the future before it completes.
                        try {
                          Thread.sleep(20);
                        } catch (InterruptedException e) {
                          Thread.currentThread().interrupt();
                        }
                        return handle;
                      });
                }));
      }
      start.countDown();
      for (Future<FilterBitsetHandle> r : results) {
        assertSame(handle, r.get(10, TimeUnit.SECONDS));
      }
    } finally {
      pool.shutdownNow();
      pool.awaitTermination(5, TimeUnit.SECONDS);
    }

    assertEquals("builder must run exactly once for a key", 1, buildCount.get());
    // One cache reference plus one caller reference per acquiring thread.
    assertEquals(1 + threads, handle.refCount.get());
    assertFalse(handle.freed.get());

    // Releasing every caller reference leaves the cache's own reference intact.
    for (int i = 0; i < threads; i++) {
      handle.decRef();
    }
    assertEquals(1, handle.refCount.get());
    assertFalse(handle.freed.get());
  }

  /** Eviction releases exactly the cache's reference, freeing evicted handles and keeping the rest. */
  @Test
  public void evictionReleasesCacheReferenceExactlyOnce() throws Exception {
    final int max = FilterBitsetCache.MAX_HOST_ENTRIES;
    final int total = 3 * max;
    final int firstRetained = total - max; // oldest `total - max` entries are evicted

    List<CountingHandle> handles = new ArrayList<>();
    for (int i = 0; i < total; i++) {
      CountingHandle h = new CountingHandle();
      handles.add(h);
      final String field = "evict-" + i;
      FilterBitsetHandle got = FilterBitsetCache.acquire(null, segKey(field), field, () -> h);
      assertSame(h, got);
      got.decRef(); // caller finished; the cache keeps its reference
    }

    for (int i = 0; i < firstRetained; i++) {
      CountingHandle h = handles.get(i);
      assertTrue("handle " + i + " should have been evicted and freed", h.freed.get());
      assertEquals("evicted handle closed exactly once", 1, h.closeCalls.get());
    }
    for (int i = firstRetained; i < total; i++) {
      CountingHandle h = handles.get(i);
      assertFalse("handle " + i + " should still be cached", h.freed.get());
      assertEquals("retained handle must not be closed", 0, h.closeCalls.get());
    }
  }

  /** A failed build is not cached, so a later acquire rebuilds. */
  @Test
  public void failedBuildIsNotCachedAndCanRetry() throws Exception {
    final String field = "failing";
    AtomicInteger buildCount = new AtomicInteger();

    try {
      FilterBitsetCache.acquire(
          null,
          segKey(field),
          field,
          () -> {
            buildCount.incrementAndGet();
            throw new IOException("boom");
          });
      fail("expected IOException to propagate");
    } catch (IOException expected) {
      assertEquals("boom", expected.getMessage());
    }
    assertEquals(1, buildCount.get());

    CountingHandle handle = new CountingHandle();
    FilterBitsetHandle got =
        FilterBitsetCache.acquire(
            null,
            segKey(field),
            field,
            () -> {
              buildCount.incrementAndGet();
              return handle;
            });
    assertSame(handle, got);
    assertEquals("failed entry must not be cached; retry rebuilds", 2, buildCount.get());
    got.decRef();
  }

  /**
   * Hammer acquire/decRef concurrently across several keys. Any unbalanced reference handling by the
   * cache trips {@link CountingHandle#decRef()}'s below-zero guard and fails the test.
   */
  @Test
  public void concurrentAcquireReleaseIsBalanced() throws Exception {
    final int keySpace = 8; // < MAX_HOST_ENTRIES, so these keys are never evicted mid-run
    final int threads = 16;
    final int iterationsPerThread = 500;

    ExecutorService pool = Executors.newFixedThreadPool(threads);
    CountDownLatch start = new CountDownLatch(1);
    List<Future<?>> tasks = new ArrayList<>();
    try {
      for (int t = 0; t < threads; t++) {
        final int seed = t;
        tasks.add(
            pool.submit(
                () -> {
                  start.await();
                  for (int i = 0; i < iterationsPerThread; i++) {
                    final String field = "stress-" + ((seed + i) % keySpace);
                    // A fresh handle per build; a cached key reuses whatever was built first.
                    FilterBitsetHandle h =
                        FilterBitsetCache.acquire(null, segKey(field), field, CountingHandle::new);
                    h.decRef();
                  }
                  return null;
                }));
      }
      start.countDown();
      for (Future<?> f : tasks) {
        f.get(30, TimeUnit.SECONDS); // surfaces any exception thrown inside a task
      }
    } finally {
      pool.shutdownNow();
      pool.awaitTermination(5, TimeUnit.SECONDS);
    }
  }
}
