/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.FilterBitsetHandle;
import com.nvidia.cuvs.MultiPartitionCagraSearch;
import com.nvidia.cuvs.MultiPartitionSearchResults;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

/**
 * Extends {@link KnnFloatVectorQuery} for GPU-only search.
 *
 * <p>When all index segments use {@link CuVS2510GPUVectorsReader}, {@link #rewrite} delegates a
 * single multi-partition search to cuVS, passing one Lucene segment per cuVS partition. cuVS
 * runs the per-partition CAGRA searches, applies distance post-processing, and performs the
 * cross-partition top-k merge internally; the returned arrays are mapped to Lucene doc IDs on
 * the host. The effective CAGRA algorithm (SINGLE_CTA or MULTI_KERNEL) is selected by cuVS
 * based on {@code searchAlgo} and {@code itopk_size}, with MULTI_KERNEL handling k beyond
 * SINGLE_CTA's per-partition cap.
 *
 * <p>If the query has an explicit {@code filter}, or if any segment carries live-document deletes,
 * the combined acceptance mask (filter ∩ liveDocs) is packed across all segments into a single
 * {@link FilterBitsetHandle}. The host-side packed arrays are cached per unique
 * (filter, reader-state, field) triple via {@link FilterBitsetCache}; the device upload is cached
 * inside the handle itself across threads.
 *
 * <p>Falls back to the standard per-segment Lucene path when the optimized path cannot be
 * applied (mixed segment types or missing CAGRA index for the field on any segment).
 *
 * @since 25.10
 */
public class GPUKnnFloatVectorQuery extends KnnFloatVectorQuery {

  private final int iTopK;
  private final int searchWidth;
  private final int threadBlockSize;
  private final int maxIterations;
  private final CagraSearchParams.SearchAlgo searchAlgo;

  /**
   * Initializes {@link GPUKnnFloatVectorQuery} with {@link CagraSearchParams.SearchAlgo#AUTO},
   * and max_iterations auto-selected (0).
   *
   * @param field       the vector field name
   * @param target      the query vector
   * @param k           the number of nearest neighbors to return
   * @param filter      optional pre-filter query
   * @param iTopK       CAGRA itopk_size parameter
   * @param searchWidth CAGRA search_width parameter
   */
  public GPUKnnFloatVectorQuery(
      String field, float[] target, int k, Query filter, int iTopK, int searchWidth) {
    this(field, target, k, filter, iTopK, searchWidth, 0, 0, CagraSearchParams.SearchAlgo.AUTO);
  }

  /**
   * Initializes {@link GPUKnnFloatVectorQuery}.
   *
   * @param field           the vector field name
   * @param target          the query vector
   * @param k               the number of nearest neighbors to return
   * @param filter          optional pre-filter query
   * @param iTopK           CAGRA itopk_size parameter
   * @param searchWidth     CAGRA search_width parameter
   * @param threadBlockSize CAGRA thread_block_size (0 = auto)
   * @param maxIterations   CAGRA max_iterations (0 = auto)
   * @param searchAlgo      CAGRA search algorithm
   */
  public GPUKnnFloatVectorQuery(
      String field,
      float[] target,
      int k,
      Query filter,
      int iTopK,
      int searchWidth,
      int threadBlockSize,
      int maxIterations,
      CagraSearchParams.SearchAlgo searchAlgo) {
    super(field, target, k, filter);
    this.iTopK = iTopK;
    this.searchWidth = searchWidth;
    this.threadBlockSize = threadBlockSize;
    this.maxIterations = maxIterations;
    this.searchAlgo = searchAlgo;
  }

  // -------------------------------------------------------------------------
  // Optimized multi-segment path
  // -------------------------------------------------------------------------

  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    IndexReader reader = indexSearcher.getIndexReader();
    List<LeafReaderContext> leaves = reader.leaves();
    if (leaves.isEmpty()) {
      return new MatchNoDocsQuery();
    }

    // Collect a CuVS2510GPUVectorsReader for every segment; fall back if any segment
    // lacks one or has no CAGRA index for this field.
    List<CuVS2510GPUVectorsReader> gpuReaders = new ArrayList<>(leaves.size());
    for (LeafReaderContext ctx : leaves) {
      CuVS2510GPUVectorsReader gpuReader = unwrapGpuReader(ctx);
      if (gpuReader == null || gpuReader.getCagraIndexForField(field) == null) {
        return super.rewrite(indexSearcher);
      }
      gpuReaders.add(gpuReader);
    }

    // Build a single filter handle encoding (filter ∩ liveDocs) across every segment whenever
    // any filtering is required — either an explicit Lucene filter, or live-document deletes in
    // at least one segment. With the single-query multi-partition API, there is no other channel
    // for per-segment liveDocs, so they must be folded into the FilterBitsetHandle.
    boolean hasExplicitFilter = (filter != null);
    boolean hasDeletes = false;
    for (LeafReaderContext ctx : leaves) {
      if (ctx.reader().getLiveDocs() != null) {
        hasDeletes = true;
        break;
      }
    }
    FilterBitsetHandle filterHandle = null;
    if (hasExplicitFilter || hasDeletes) {
      filterHandle = buildOrGetCachedFilterHandle(indexSearcher, leaves, gpuReaders);
    }

    // Build the per-segment CagraIndex list (one entry per Lucene segment / cuVS partition).
    CuVSResources resources = getCuVSResourcesInstance();
    List<CagraIndex> cagraIndices = new ArrayList<>(leaves.size());

    try {
      float[] target = getTargetCopy();
      CagraSearchParams searchParams =
          new CagraSearchParams.Builder()
              .withItopkSize(Math.max(iTopK, k))
              .withSearchWidth(searchWidth)
              .withThreadBlockSize(threadBlockSize)
              .withMaxIterations(maxIterations)
              .withAlgo(searchAlgo)
              .build();

      // Upload the query vector to device once; the same matrix view is searched against every
      // partition by the cuVS multi-partition API.
      CuVSMatrix.Builder<?> vectorBuilder =
          CuVSMatrix.deviceBuilder(resources, 1, target.length, CuVSMatrix.DataType.FLOAT);
      vectorBuilder.addVector(target);

      ScoreDoc[] scoreDocs;
      try (CuVSMatrix queryVector = vectorBuilder.build()) {
        for (int i = 0; i < leaves.size(); i++) {
          cagraIndices.add(gpuReaders.get(i).getCagraIndexForField(field));
        }

        CagraQuery cagraQuery =
            new CagraQuery.Builder(resources)
                .withTopK(k)
                .withSearchParams(searchParams)
                .withQueryVectors(queryVector)
                .build();

        MultiPartitionSearchResults results =
            MultiPartitionCagraSearch.search(resources, cagraIndices, cagraQuery, k, filterHandle);

        if (results.count() == 0) {
          return new MatchNoDocsQuery();
        }

        // Map (segmentIdx, ordinal) → global Lucene doc ID; compute normalized score.
        // Stash FloatVectorValues per segment: results.count() can be far larger than the
        // number of segments, and getFloatVectorValues() allocates on each call.
        FloatVectorValues[] segVectorValues = new FloatVectorValues[leaves.size()];
        scoreDocs = new ScoreDoc[results.count()];
        for (int j = 0; j < results.count(); j++) {
          int segIdx = results.getPartitionIndex(j);
          int ordinal = results.getOrdinal(j);
          float dist = results.getDistance(j);

          FloatVectorValues fvv = segVectorValues[segIdx];
          if (fvv == null) {
            fvv = gpuReaders.get(segIdx).getFloatVectorValues(field);
            segVectorValues[segIdx] = fvv;
          }
          LeafReaderContext ctx = leaves.get(segIdx);
          int globalDoc = ctx.docBase + fvv.ordToDoc(ordinal);
          float score = 1.0f / (1.0f + dist);
          scoreDocs[j] = new ScoreDoc(globalDoc, score);
        }

        Arrays.sort(scoreDocs, Comparator.comparingDouble((ScoreDoc sd) -> sd.score).reversed());

        return docAndScoreQuery(scoreDocs);
      }

    } catch (Throwable t) {
      if (t instanceof IOException) throw (IOException) t;
      if (t instanceof RuntimeException) throw (RuntimeException) t;
      throw new RuntimeException("Multi-segment GPU search failed", t);
    }
  }

  // -------------------------------------------------------------------------
  // Per-segment fallback path (used when k > 1024 or not all GPU segments)
  // -------------------------------------------------------------------------

  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      Bits acceptDocs,
      int visitedLimit,
      KnnCollectorManager knnCollectorManager)
      throws IOException {
    GPUPerLeafCuVSKnnCollector results =
        new GPUPerLeafCuVSKnnCollector(
            k, visitedLimit, iTopK, searchWidth, threadBlockSize, maxIterations, searchAlgo);
    context.reader().searchNearestVectors(field, getTargetCopy(), results, acceptDocs);
    return results.topDocs();
  }

  // -------------------------------------------------------------------------
  // Filter handle construction
  // -------------------------------------------------------------------------

  /**
   * Returns a {@link FilterBitsetHandle} encoding ({@link #filter} ∩ liveDocs) for every segment,
   * pulling from {@link FilterBitsetCache} when the reader state is unchanged.
   *
   * <p>Cache key uses per-reader keys (not just core keys) so that liveDocs changes — which happen
   * when a reader is reopened after deletes — automatically invalidate the cached bitset.
   */
  private FilterBitsetHandle buildOrGetCachedFilterHandle(
      IndexSearcher indexSearcher,
      List<LeafReaderContext> leaves,
      List<CuVS2510GPUVectorsReader> gpuReaders)
      throws IOException {

    List<Object> segReaderKeys = new ArrayList<>(leaves.size());
    for (LeafReaderContext ctx : leaves) {
      var helper = ctx.reader().getReaderCacheHelper();
      if (helper == null) {
        // This reader can't be cached; build without caching.
        return buildFilterHandle(indexSearcher, leaves, gpuReaders);
      }
      segReaderKeys.add(helper.getKey());
    }

    FilterBitsetHandle cached = FilterBitsetCache.get(filter, segReaderKeys, field);
    if (cached != null) return cached;

    FilterBitsetHandle handle = buildFilterHandle(indexSearcher, leaves, gpuReaders);
    FilterBitsetCache.put(filter, segReaderKeys, field, handle);
    return handle;
  }

  /**
   * Evaluates {@link #filter} per segment (when set), intersects with liveDocs, and packs the
   * result into a new {@link FilterBitsetHandle}. When {@link #filter} is {@code null}, the
   * handle encodes liveDocs alone — this path is taken when one or more segments have deletes
   * but no explicit Lucene filter was supplied.
   */
  private FilterBitsetHandle buildFilterHandle(
      IndexSearcher indexSearcher,
      List<LeafReaderContext> leaves,
      List<CuVS2510GPUVectorsReader> gpuReaders)
      throws IOException {

    Weight filterWeight = null;
    if (filter != null) {
      filterWeight =
          indexSearcher.createWeight(
              indexSearcher.rewrite(filter), ScoreMode.COMPLETE_NO_SCORES, 1.0f);
    }

    int numSegments = leaves.size();
    long[] segBitOffsets = new long[numSegments];
    long totalBits = 0;
    for (int i = 0; i < numSegments; i++) {
      segBitOffsets[i] = totalBits;
      int numOrds = gpuReaders.get(i).getFloatVectorValues(field).size();
      totalBits += ((long) (numOrds + 63) / 64) * 64;
    }
    long[] combinedLongs = new long[(int) (totalBits / 64)];

    for (int i = 0; i < numSegments; i++) {
      LeafReaderContext ctx = leaves.get(i);
      Bits liveDocs = ctx.reader().getLiveDocs();
      // When filterWeight is null, accept all live documents (acceptDocs == liveDocs, which may
      // itself be null to mean "all docs accepted" in this segment).
      Bits acceptDocs = (filterWeight != null) ? evalFilter(filterWeight, ctx, liveDocs) : liveDocs;
      FloatVectorValues fvv = gpuReaders.get(i).getFloatVectorValues(field);
      Bits acceptedOrds = fvv.getAcceptOrds(acceptDocs);
      int numOrds = fvv.size();
      int longOffset = (int) (segBitOffsets[i] / 64);
      packOrdsToLongs(acceptedOrds, numOrds, combinedLongs, longOffset);
    }

    // Only the combined bitset is transported; cuVS recomputes the per-partition bit offsets from
    // the index sizes. segBitOffsets is used above solely to pack each segment's slice (64-bit
    // word-aligned), matching that recomputation.
    return FilterBitsetHandle.create(combinedLongs);
  }

  /**
   * Evaluates {@code filterWeight} in {@code ctx} and intersects with {@code liveDocs}.
   * Returns a {@link Bits} over local doc IDs where {@code get(doc)} is true for accepted docs.
   */
  private static Bits evalFilter(Weight filterWeight, LeafReaderContext ctx, Bits liveDocs)
      throws IOException {
    ScorerSupplier scorerSupplier = filterWeight.scorerSupplier(ctx);
    if (scorerSupplier == null) {
      int maxDoc = ctx.reader().maxDoc();
      return new Bits() {
        @Override
        public boolean get(int i) {
          return false;
        }

        @Override
        public int length() {
          return maxDoc;
        }
      };
    }

    int maxDoc = ctx.reader().maxDoc();
    FixedBitSet filterBits = new FixedBitSet(maxDoc);
    Scorer scorer = scorerSupplier.get(Long.MAX_VALUE);
    DocIdSetIterator it = scorer.iterator();
    int doc;
    while ((doc = it.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
      filterBits.set(doc);
    }

    if (liveDocs == null) return filterBits;

    // Intersect: accept only docs that pass both filter and liveDocs.
    return new Bits() {
      @Override
      public boolean get(int i) {
        return filterBits.get(i) && liveDocs.get(i);
      }

      @Override
      public int length() {
        return maxDoc;
      }
    };
  }

  /**
   * Packs {@code numOrds} ordinal bits from {@code bits} into {@code dest} starting at long index
   * {@code destLongOffset}. {@code bits == null} means all ordinals accepted (Lucene convention).
   */
  private static void packOrdsToLongs(Bits bits, int numOrds, long[] dest, int destLongOffset) {
    if (bits == null) {
      // All ordinals accepted: fill with all-ones, masking the last partial word.
      int numLongs = (numOrds + 63) / 64;
      Arrays.fill(dest, destLongOffset, destLongOffset + numLongs, -1L);
      int tail = numOrds % 64;
      if (tail != 0) {
        dest[destLongOffset + numLongs - 1] = (1L << tail) - 1L;
      }
      return;
    }
    for (int i = 0; i < numOrds; i++) {
      if (bits.get(i)) {
        dest[destLongOffset + i / 64] |= (1L << (i % 64));
      }
    }
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  /**
   * Unwraps the {@link LeafReaderContext}'s reader to a {@link CuVS2510GPUVectorsReader}, or
   * returns {@code null} if the reader is not of that type.
   */
  private static CuVS2510GPUVectorsReader unwrapGpuReader(LeafReaderContext ctx) {
    var unwrapped = FilterLeafReader.unwrap(ctx.reader());
    if (!(unwrapped instanceof CodecReader)) return null;
    KnnVectorsReader vr = ((CodecReader) unwrapped).getVectorReader();
    return (vr instanceof CuVS2510GPUVectorsReader gpuReader) ? gpuReader : null;
  }

  /**
   * Builds a {@link Query} that matches exactly the given pre-scored documents.
   *
   * <p>Partitions {@code scoreDocs} by segment (using {@link ScoreDoc#shardIndex} as the segment
   * offset relative to {@link LeafReaderContext#docBase}), then returns a {@link Scorer} per
   * segment that iterates those docs in ascending doc-ID order and replays their pre-computed
   * scores.
   */
  private static Query docAndScoreQuery(ScoreDoc[] scoreDocs) {
    return new Query() {
      @Override
      public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost)
          throws IOException {
        return new Weight(this) {
          @Override
          public ScorerSupplier scorerSupplier(LeafReaderContext ctx) {
            int base = ctx.docBase;
            int maxDoc = base + ctx.reader().maxDoc();
            // Collect docs belonging to this segment; re-sort by local doc ID ascending.
            int[] localDocs = new int[scoreDocs.length];
            float[] localScores = new float[scoreDocs.length];
            int count = 0;
            for (ScoreDoc sd : scoreDocs) {
              if (sd.doc >= base && sd.doc < maxDoc) {
                localDocs[count] = sd.doc - base;
                localScores[count] = sd.score * boost;
                count++;
              }
            }
            if (count == 0) return null;
            final int n = count;
            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Arrays.sort(idx, Comparator.comparingInt(i -> localDocs[i]));
            final int[] sortedDocs = new int[n];
            final float[] sortedScores = new float[n];
            for (int i = 0; i < n; i++) {
              sortedDocs[i] = localDocs[idx[i]];
              sortedScores[i] = localScores[idx[i]];
            }

            return new ScorerSupplier() {
              @Override
              public Scorer get(long leadCost) {
                return new Scorer() {
                  private int pos = -1;

                  @Override
                  public DocIdSetIterator iterator() {
                    return new DocIdSetIterator() {
                      @Override
                      public int docID() {
                        return pos < 0 ? -1 : (pos >= n ? NO_MORE_DOCS : sortedDocs[pos]);
                      }

                      @Override
                      public int nextDoc() {
                        pos++;
                        return docID();
                      }

                      @Override
                      public int advance(int target) {
                        if (pos < 0) pos = 0;
                        while (pos < n && sortedDocs[pos] < target) pos++;
                        return docID();
                      }

                      @Override
                      public long cost() {
                        return n;
                      }
                    };
                  }

                  @Override
                  public float getMaxScore(int upTo) {
                    return Float.MAX_VALUE;
                  }

                  @Override
                  public float score() {
                    return sortedScores[pos];
                  }

                  @Override
                  public int docID() {
                    return pos < 0
                        ? -1
                        : (pos >= n ? DocIdSetIterator.NO_MORE_DOCS : sortedDocs[pos]);
                  }
                };
              }

              @Override
              public long cost() {
                return n;
              }
            };
          }

          @Override
          public boolean isCacheable(LeafReaderContext ctx) {
            return false;
          }

          @Override
          public Explanation explain(LeafReaderContext ctx, int doc) {
            for (ScoreDoc sd : scoreDocs) {
              if (sd.doc == ctx.docBase + doc) {
                return Explanation.match(sd.score, "GPU multi-segment CAGRA search");
              }
            }
            return Explanation.noMatch("not a GPU search result");
          }
        };
      }

      @Override
      public String toString(String field) {
        return "GPUDocAndScoreQuery";
      }

      @Override
      public void visit(QueryVisitor visitor) {}

      @Override
      public boolean equals(Object o) {
        return this == o;
      }

      @Override
      public int hashCode() {
        return System.identityHashCode(this);
      }
    };
  }
}
