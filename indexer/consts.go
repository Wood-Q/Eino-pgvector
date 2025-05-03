/*
 * Copyright 2024 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package indexer

const typ = "PGVector"

// VectorType represents the vector types supported by PGVector
type VectorType string

// IndexType represents the index types supported by PGVector
type IndexType string

// DistanceType 表示向量距离类型
// 支持 L2、内积、余弦等
// 可根据实际需求扩展
type DistanceType string

const (
	extraKeyPGVectorFields = "_pgvector_fields" // value: map[string]interface{}
	extraKeyPGVectorTTL    = "_pgvector_ttl"    // value: int64
)

const (
	DefaultFieldAutoID   = "id"
	DefaultFieldID       = "document_id"
	DefaultFieldVector   = "embedding"
	DefaultFieldContent  = "content"
	DefaultFieldMetadata = "metadata"
	DefaultBatchSize     = 10
)

const (
	DistanceTypeL2           DistanceType = "l2"
	DistanceTypeInnerProduct DistanceType = "inner_product"
	DistanceTypeCosine       DistanceType = "cosine"
)

const (
	// IndexTypeHNSW represents HNSW index type
	IndexTypeHNSW IndexType = "hnsw"
	// IndexTypeIVFFlat represents IVFFlat index type
	IndexTypeIVFFlat IndexType = "ivfflat"
)

// VectorTypeVector standard vector type, supports up to 16000 dimensions
const VectorTypeVector VectorType = "vector"

// VectorTypeHalfvec half-precision vector type, supports up to 16000 dimensions
const VectorTypeHalfvec VectorType = "halfvec"

// VectorTypeBit bit vector type, supports up to 64000 dimensions
const VectorTypeBit VectorType = "bit"

// VectorTypeSparsevec sparse vector type, supports up to 16000 non-zero elements
const VectorTypeSparsevec VectorType = "sparsevec"
