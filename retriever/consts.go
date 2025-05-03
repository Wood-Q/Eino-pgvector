package retriever

// VectorType represents the vector types supported by PGVector
type VectorType string

// IndexType represents the index types supported by PGVector
type IndexType string

// DistanceType 表示向量距离类型
// 支持 L2、内积、余弦等
// 可根据实际需求扩展
type DistanceType string

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

const (
	DefaultFieldAutoID   = "id"
	DefaultFieldID       = "document_id"
	DefaultFieldVector   = "embedding"
	DefaultFieldContent  = "content"
	DefaultFieldMetadata = "metadata"
	DefaultBatchSize     = 10
)
