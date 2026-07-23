/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

/**
 * Configuration for the filter-bitset cache used by multi-segment GPU searches.
 *
 * <p>This configuration is local to one vectors-format instance. It does not configure cuvs-java's
 * process-wide filter-bitset device pool.
 *
 * @param enabled whether filter bitsets are cached between searches
 * @param maxBytes maximum bytes cached by this format; {@code 0} derives a budget from the first
 *     filtered search
 */
public record FilterBitsetCacheConfig(boolean enabled, long maxBytes) {

  /** Default configuration used by SPI-created codecs and vector formats. */
  public static final FilterBitsetCacheConfig DEFAULT = new FilterBitsetCacheConfig(true, 0);

  public FilterBitsetCacheConfig {
    if (maxBytes < 0) {
      throw new IllegalArgumentException("maxBytes must be non-negative");
    }
  }
}
