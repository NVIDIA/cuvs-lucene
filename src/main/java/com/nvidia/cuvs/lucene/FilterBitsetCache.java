/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.FilterBitsetHandle;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.Query;

/**
 * Shared LRU cache mapping (filter Query, single-segment reader key, field) → {@link
 * FilterBitsetHandle}. One entry per segment, so an index update only invalidates the changed
 * segments' entries rather than the whole cache.
 *
 * <p>Host-side cache holding packed bitset arrays; the device-side upload is managed inside {@link
 * FilterBitsetHandle} itself. Entries are evicted in LRU order.
 *
 * <h2>Concurrency</h2>
 *
 * <p>Values are stored as {@link CompletableFuture}s so that concurrent misses on the same key
 * build the handle exactly once: the first thread inserts an incomplete future and builds; others
 * find the future and await it (without holding the cache lock, so builds for different keys still
 * run in parallel).
 *
 * <p>Lifetime is reference-counted (see {@link FilterBitsetHandle}). A cached handle carries one
 * cache-owned reference, released via {@link FilterBitsetHandle#close()} when the entry is evicted.
 * {@link #acquire} returns a handle with an <em>additional</em> reference already taken; the caller
 * must {@link FilterBitsetHandle#decRef()} it after use. Because the free happens only when the last
 * reference is dropped, an eviction concurrent with an in-flight search cannot free a handle still
 * in use, and a replaced entry cannot leak.
 */
final class FilterBitsetCache {

  static final int MAX_HOST_ENTRIES = 16;

  /** Builds the handle for a key on a cache miss. */
  @FunctionalInterface
  interface FilterBuilder {
    FilterBitsetHandle build() throws IOException;
  }

  private record FilterCacheKey(Query filter, Object segReaderKey, String field) {
    @Override
    public boolean equals(Object o) {
      if (!(o instanceof FilterCacheKey other)) return false;
      return Objects.equals(filter, other.filter)
          && Objects.equals(segReaderKey, other.segReaderKey)
          && Objects.equals(field, other.field);
    }

    @Override
    public int hashCode() {
      return Objects.hash(filter, segReaderKey, field);
    }
  }

  private static final LinkedHashMap<FilterCacheKey, CompletableFuture<FilterBitsetHandle>> CACHE =
      new LinkedHashMap<>(MAX_HOST_ENTRIES + 2, 0.75f, /* access-order= */ true) {
        @Override
        protected boolean removeEldestEntry(
            Map.Entry<FilterCacheKey, CompletableFuture<FilterBitsetHandle>> eldest) {
          if (size() > MAX_HOST_ENTRIES) {
            releaseCacheRef(eldest.getValue());
            return true;
          }
          return false;
        }
      };

  // Segment reader keys with a registered close listener, so each reader registers exactly once.
  private static final Set<Object> REGISTERED = new HashSet<>();

  private FilterBitsetCache() {}

  /**
   * Returns the handle for the given key, building it once if absent. The returned handle has a
   * reference taken on the caller's behalf that must be released with {@link
   * FilterBitsetHandle#decRef()} once the search completes.
   */
  static FilterBitsetHandle acquire(
      Query filter, Object segReaderKey, String field, FilterBuilder builder) throws IOException {
    FilterCacheKey key = new FilterCacheKey(filter, segReaderKey, field);
    while (true) {
      CompletableFuture<FilterBitsetHandle> future;
      boolean owner = false;
      synchronized (FilterBitsetCache.class) {
        future = CACHE.get(key);
        if (future == null) {
          future = new CompletableFuture<>();
          CACHE.put(key, future);
          owner = true;
        }
      }

      if (owner) {
        // Build outside the cache lock so misses on other keys are not serialized behind this one.
        try {
          future.complete(builder.build());
        } catch (Throwable t) {
          // Drop the failed entry so a later call rebuilds, then propagate to this thread and any
          // waiters (which observe the exception via join() and retry).
          synchronized (FilterBitsetCache.class) {
            CACHE.remove(key, future);
          }
          future.completeExceptionally(t);
          if (t instanceof IOException io) throw io;
          if (t instanceof RuntimeException re) throw re;
          if (t instanceof Error err) throw err;
          throw new RuntimeException(t);
        }
      }

      FilterBitsetHandle handle;
      try {
        handle = future.join();
      } catch (CompletionException e) {
        // The owning thread's build failed and removed the entry; retry so exactly one thread
        // rebuilds.
        continue;
      }

      if (handle.tryIncRef()) {
        return handle;
      }
      // Rare: the entry was evicted and its cache reference released between completion and our
      // acquisition. It is already gone from the map, so retry and rebuild.
    }
  }

  /**
   * Registers a one-time close listener on {@code helper} so every cache entry for this segment
   * reader is evicted (and its device allocation released) as soon as the reader's core closes —
   * e.g. when the segment is merged away or the reader is reopened with new deletes. This ties the
   * cache to the segment lifecycle (mirroring Lucene's own query cache) so obsolete entries do not
   * linger until LRU eviction.
   */
  static void ensureCloseListener(IndexReader.CacheHelper helper) {
    Object segReaderKey = helper.getKey();
    synchronized (FilterBitsetCache.class) {
      if (!REGISTERED.add(segReaderKey)) return; // a listener is already registered for this reader
    }
    helper.addClosedListener(FilterBitsetCache::invalidateReader);
  }

  /** Evicts and releases every cache entry belonging to the given segment reader key. */
  static void invalidateReader(Object segReaderKey) {
    List<CompletableFuture<FilterBitsetHandle>> toRelease = new ArrayList<>();
    synchronized (FilterBitsetCache.class) {
      REGISTERED.remove(segReaderKey);
      var it = CACHE.entrySet().iterator();
      while (it.hasNext()) {
        Map.Entry<FilterCacheKey, CompletableFuture<FilterBitsetHandle>> e = it.next();
        if (Objects.equals(e.getKey().segReaderKey(), segReaderKey)) {
          toRelease.add(e.getValue());
          it.remove();
        }
      }
    }
    for (CompletableFuture<FilterBitsetHandle> future : toRelease) {
      releaseCacheRef(future);
    }
  }

  /** Releases the cache's reference once the (possibly in-flight) build completes. */
  private static void releaseCacheRef(CompletableFuture<FilterBitsetHandle> future) {
    future.whenComplete(
        (handle, err) -> {
          if (handle != null) handle.close();
        });
  }
}
