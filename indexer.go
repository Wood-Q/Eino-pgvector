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

package pgvector

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	_ "github.com/lib/pq"
	"github.com/pgvector/pgvector-go"

	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino/schema"
)

const (
	defaultBatchSize = 10
)

// VectorType represents the vector types supported by PGVector
type VectorType string

// VectorTypeVector standard vector type, supports up to 16000 dimensions
const VectorTypeVector VectorType = "vector"

// VectorTypeHalfvec half-precision vector type, supports up to 16000 dimensions
const VectorTypeHalfvec VectorType = "halfvec"

// VectorTypeBit bit vector type, supports up to 64000 dimensions
const VectorTypeBit VectorType = "bit"

// VectorTypeSparsevec sparse vector type, supports up to 16000 non-zero elements
const VectorTypeSparsevec VectorType = "sparsevec"

// IndexType represents the index types supported by PGVector
type IndexType string

const (
	// IndexTypeHNSW represents HNSW index type
	IndexTypeHNSW IndexType = "hnsw"
	// IndexTypeIVFFlat represents IVFFlat index type
	IndexTypeIVFFlat IndexType = "ivfflat"
)

// DistanceType 表示向量距离类型
// 支持 L2、内积、余弦等
// 可根据实际需求扩展
type DistanceType string

const (
	DistanceTypeL2           DistanceType = "l2"
	DistanceTypeInnerProduct DistanceType = "inner_product"
	DistanceTypeCosine       DistanceType = "cosine"
)

// IndexerConfig configures the PGVector indexer
type IndexerConfig struct {
	// PostgreSQL connection information
	Host     string `json:"host"`
	Port     int    `json:"port"`
	User     string `json:"user"`
	Password string `json:"password"`
	DBName   string `json:"db_name"`
	SSLMode  string `json:"ssl_mode"`

	// Table name
	TableName string `json:"table_name"`

	// Vector dimension
	Dimension int `json:"dimension"`

	// Vector type
	VectorType VectorType `json:"vector_type"`

	// Index type
	IndexType IndexType `json:"index_type"`

	// Index options
	IndexOptions map[string]interface{} `json:"index_options"`

	// Batch insert size
	BatchSize int `json:"batch_size"`

	// Vectorization configuration
	Embedding embedding.Embedder `json:"embedding"`
}

// Indexer implements the PGVector indexer
type Indexer struct {
	config *IndexerConfig
	db     *sql.DB
}

// getSuitableVectorType automatically selects the appropriate vector type based on dimension
func getSuitableVectorType(dimension int) VectorType {
	switch {
	case dimension <= 16000:
		return VectorTypeVector
	case dimension <= 64000:
		return VectorTypeBit
	default:
		return VectorTypeSparsevec
	}
}

// NewIndexer creates a new PGVector indexer
func NewIndexer(ctx context.Context, config *IndexerConfig) (*Indexer, error) {
	if config.Embedding == nil {
		return nil, fmt.Errorf("[PGVectorIndexer] embedding is required")
	}

	if config.BatchSize == 0 {
		config.BatchSize = defaultBatchSize
	}

	// Set default vector type
	if config.VectorType == "" {
		config.VectorType = getSuitableVectorType(config.Dimension)
	} else {
		// If the specified vector type is not supported for the current dimension, automatically switch to a suitable type
		if err := validateVectorConfig(config.VectorType, config.Dimension); err != nil {
			newType := getSuitableVectorType(config.Dimension)
			fmt.Printf("[PGVectorIndexer] Warning: Automatically switching vector type from %s to %s to support %d dimensions\n",
				config.VectorType, newType, config.Dimension)
			config.VectorType = newType
		}
	}

	// Build connection string
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		config.Host, config.Port, config.User, config.Password, config.DBName, config.SSLMode)

	// Connect to database
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("[PGVectorIndexer] failed to connect to database: %w", err)
	}

	// Test connection
	if err = db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("[PGVectorIndexer] failed to ping database: %w", err)
	}

	// Create indexer instance
	i := &Indexer{
		config: config,
		db:     db,
	}

	// Ensure table exists
	if err = i.ensureTable(ctx); err != nil {
		return nil, err
	}

	return i, nil
}

// validateVectorConfig checks if the vector configuration is valid
func validateVectorConfig(vectorType VectorType, dimension int) error {
	switch vectorType {
	case VectorTypeVector:
		if dimension > 16000 {
			return fmt.Errorf("[PGVectorIndexer] vector type 'vector' supports up to 16000 dimensions, got %d", dimension)
		}
	case VectorTypeHalfvec:
		if dimension > 16000 {
			return fmt.Errorf("[PGVectorIndexer] vector type 'halfvec' supports up to 16000 dimensions, got %d", dimension)
		}
	case VectorTypeBit:
		if dimension > 64000 {
			return fmt.Errorf("[PGVectorIndexer] vector type 'bit' supports up to 64000 dimensions, got %d", dimension)
		}
	case VectorTypeSparsevec:
		if dimension > 16000 {
			return fmt.Errorf("[PGVectorIndexer] vector type 'sparsevec' supports up to 16000 non-zero elements, got %d", dimension)
		}
	default:
		return fmt.Errorf("[PGVectorIndexer] unsupported vector type: %s", vectorType)
	}
	return nil
}

// ensureTable ensures that the vector table exists
func (i *Indexer) ensureTable(ctx context.Context) error {
	// Check if pgvector extension is installed
	_, err := i.db.ExecContext(ctx, "CREATE EXTENSION IF NOT EXISTS vector;")
	if err != nil {
		return fmt.Errorf("[PGVectorIndexer] failed to create vector extension: %w", err)
	}

	// Create table
	createTableSQL := fmt.Sprintf(`
	CREATE TABLE IF NOT EXISTS %s (
		%s TEXT PRIMARY KEY,
		%s TEXT NOT NULL,
		%s JSONB,
		%s %s(%d) NOT NULL
	);
	`,
		i.config.TableName,
		DefaultFieldID,
		DefaultFieldContent,
		DefaultFieldMetadata,
		DefaultFieldVector,
		i.config.VectorType,
		i.config.Dimension,
	)

	_, err = i.db.ExecContext(ctx, createTableSQL)
	if err != nil {
		return fmt.Errorf("[PGVectorIndexer] failed to create table: %w", err)
	}

	// Create index if specified
	if i.config.IndexType != "" {
		if err := i.createIndex(ctx); err != nil {
			return err
		}
	}

	return nil
}

// createIndex creates the specified index type
func (i *Indexer) createIndex(ctx context.Context) error {
	var indexSQL string
	var indexOps string

	// Determine index operations based on vector type
	switch i.config.VectorType {
	case VectorTypeVector:
		indexOps = "vector_l2_ops"
	case VectorTypeHalfvec:
		indexOps = "halfvec_l2_ops"
	case VectorTypeBit:
		indexOps = "bit_hamming_ops"
	case VectorTypeSparsevec:
		indexOps = "sparsevec_l2_ops"
	default:
		return fmt.Errorf("[PGVectorIndexer] unsupported vector type for indexing: %s", i.config.VectorType)
	}

	// Build index creation SQL based on index type
	switch i.config.IndexType {
	case IndexTypeHNSW:
		// Get HNSW parameters
		m := 16              // default max connections per layer
		efConstruction := 64 // default size of dynamic candidate list
		if opts, ok := i.config.IndexOptions["m"].(int); ok {
			m = opts
		}
		if opts, ok := i.config.IndexOptions["ef_construction"].(int); ok {
			efConstruction = opts
		}

		indexSQL = fmt.Sprintf(
			"CREATE INDEX IF NOT EXISTS %s_%s_idx ON %s USING hnsw (%s %s) WITH (m = %d, ef_construction = %d);",
			i.config.TableName,
			DefaultFieldVector,
			i.config.TableName,
			DefaultFieldVector,
			indexOps,
			m,
			efConstruction,
		)

	case IndexTypeIVFFlat:
		// Get IVFFlat parameters
		lists := 100 // default number of lists
		if opts, ok := i.config.IndexOptions["lists"].(int); ok {
			lists = opts
		}

		indexSQL = fmt.Sprintf(
			"CREATE INDEX IF NOT EXISTS %s_%s_idx ON %s USING ivfflat (%s %s) WITH (lists = %d);",
			i.config.TableName,
			DefaultFieldVector,
			i.config.TableName,
			DefaultFieldVector,
			indexOps,
			lists,
		)

	default:
		return fmt.Errorf("[PGVectorIndexer] unsupported index type: %s", i.config.IndexType)
	}

	// Execute index creation
	_, err := i.db.ExecContext(ctx, indexSQL)
	if err != nil {
		return fmt.Errorf("[PGVectorIndexer] failed to create index: %w", err)
	}

	return nil
}

// Store stores documents to PGVector
func (i *Indexer) Store(ctx context.Context, docs []*schema.Document, opts ...indexer.Option) (ids []string, err error) {
	options := indexer.GetCommonOptions(&indexer.Options{
		Embedding: i.config.Embedding,
	}, opts...)

	ctx = callbacks.EnsureRunInfo(ctx, i.GetType(), components.ComponentOfIndexer)
	ctx = callbacks.OnStart(ctx, &indexer.CallbackInput{Docs: docs})
	defer func() {
		if err != nil {
			callbacks.OnError(ctx, err)
		}
	}()

	// Get embedding vectors
	emb := options.Embedding
	if emb == nil {
		return nil, fmt.Errorf("[PGVectorIndexer] embedding not provided")
	}

	// Extract document content
	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.Content)
	}

	// Generate vector embeddings
	vectors, err := emb.EmbedStrings(i.makeEmbeddingCtx(ctx, emb), texts)
	if err != nil {
		return nil, fmt.Errorf("[PGVectorIndexer] embedding failed: %w", err)
	}

	if len(vectors) != len(docs) {
		return nil, fmt.Errorf("[PGVectorIndexer] embedding result length mismatch: need %d, got %d", len(docs), len(vectors))
	}

	// Batch insert documents
	ids = make([]string, 0, len(docs))
	for j := 0; j < len(docs); j += i.config.BatchSize {
		end := j + i.config.BatchSize
		if end > len(docs) {
			end = len(docs)
		}

		batchDocs := docs[j:end]
		batchVectors := vectors[j:end]

		batchIDs, err := i.batchInsert(ctx, batchDocs, batchVectors)
		if err != nil {
			return nil, err
		}

		ids = append(ids, batchIDs...)
	}

	callbacks.OnEnd(ctx, &indexer.CallbackOutput{IDs: ids})

	return ids, nil
}

// batchInsert batch inserts documents to PGVector
func (i *Indexer) batchInsert(ctx context.Context, docs []*schema.Document, vectors [][]float64) ([]string, error) {
	if len(docs) == 0 {
		return []string{}, nil
	}

	// Begin transaction
	tx, err := i.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("[PGVectorIndexer] failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			tx.Rollback()
		}
	}()

	// Build insert statement
	valueStrings := make([]string, 0, len(docs))
	valueArgs := make([]interface{}, 0, len(docs)*4)
	ids := make([]string, 0, len(docs))

	for idx, doc := range docs {
		ids = append(ids, doc.ID)

		// Process metadata
		metadata, err := json.Marshal(doc.MetaData)
		if err != nil {
			return nil, fmt.Errorf("[PGVectorIndexer] failed to marshal metadata: %w", err)
		}

		// Process vector
		vectorStr := i.formatVector(vectors[idx])

		placeholderOffset := idx * 4
		valueStrings = append(valueStrings, fmt.Sprintf("($%d, $%d, $%d, $%d)",
			placeholderOffset+1, placeholderOffset+2, placeholderOffset+3, placeholderOffset+4))

		valueArgs = append(valueArgs, doc.ID, doc.Content, string(metadata), vectorStr)
	}

	// Build complete SQL
	sql := fmt.Sprintf(
		"INSERT INTO %s (%s, %s, %s, %s) VALUES %s ON CONFLICT (%s) DO UPDATE SET %s = EXCLUDED.%s, %s = EXCLUDED.%s, %s = EXCLUDED.%s",
		i.config.TableName,
		DefaultFieldID, DefaultFieldContent, DefaultFieldMetadata, DefaultFieldVector,
		strings.Join(valueStrings, ", "),
		DefaultFieldID,
		DefaultFieldContent, DefaultFieldContent,
		DefaultFieldMetadata, DefaultFieldMetadata,
		DefaultFieldVector, DefaultFieldVector,
	)

	// Execute insert
	_, err = tx.ExecContext(ctx, sql, valueArgs...)
	if err != nil {
		return nil, fmt.Errorf("[PGVectorIndexer] failed to insert documents: %w", err)
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return nil, fmt.Errorf("[PGVectorIndexer] failed to commit transaction: %w", err)
	}

	return ids, nil
}

// formatVector formats a float array as PGVector format
func (i *Indexer) formatVector(vector []float64) string {
	strValues := make([]string, len(vector))
	for i, v := range vector {
		strValues[i] = strconv.FormatFloat(v, 'f', -1, 64)
	}
	return fmt.Sprintf("[%s]", strings.Join(strValues, ","))
}

// makeEmbeddingCtx creates embedding context
func (i *Indexer) makeEmbeddingCtx(ctx context.Context, emb embedding.Embedder) context.Context {
	runInfo := &callbacks.RunInfo{
		Component: components.ComponentOfEmbedding,
	}

	if embType, ok := components.GetType(emb); ok {
		runInfo.Type = embType
	}

	runInfo.Name = runInfo.Type + string(runInfo.Component)

	return callbacks.ReuseHandlers(ctx, runInfo)
}

// GetType returns the indexer type
func (i *Indexer) GetType() string {
	return typ
}

// IsCallbacksEnabled returns whether callbacks are enabled
func (i *Indexer) IsCallbacksEnabled() bool {
	return true
}

// Close closes the database connection
func (i *Indexer) Close() error {
	if i.db != nil {
		return i.db.Close()
	}
	return nil
}

// SearchOptions configures the vector search
type SearchOptions struct {
	// Distance type to use for search
	DistanceType DistanceType
	// Maximum number of results to return
	Limit int
	// Maximum distance threshold (optional)
	MaxDistance *float64
	// Additional filter conditions in SQL WHERE clause format (without the WHERE keyword)
	Filter string
	// Filter parameters for prepared statements
	FilterParams []interface{}
	// HNSW specific options
	HNSWOptions *HNSWSearchOptions
	// IVFFlat specific options
	IVFFlatOptions *IVFFlatSearchOptions
}

// HNSWSearchOptions configures HNSW specific search parameters
type HNSWSearchOptions struct {
	// Size of the dynamic candidate list for search (default: 40)
	EFSearch int
	// Whether to use iterative scan
	IterativeScan string // "strict_order" or "relaxed_order"
	// Max number of tuples to visit (default: 20000)
	MaxScanTuples int
	// Memory multiplier for scan (default: 1)
	ScanMemMultiplier int
}

// IVFFlatSearchOptions configures IVFFlat specific search parameters
type IVFFlatSearchOptions struct {
	// Number of probes (default: 1)
	Probes int
	// Whether to use iterative scan
	IterativeScan string // "strict_order" or "relaxed_order"
	// Max number of probes
	MaxProbes int
}

// CalculateDistance calculates the distance between two vectors
func (i *Indexer) CalculateDistance(ctx context.Context, vector []float64, distType DistanceType) (float64, error) {
	// Format the vector
	formattedVector := i.formatVector(vector)

	// Build the SQL query based on distance type
	var query string
	switch distType {
	case DistanceTypeInnerProduct:
		// For inner product, multiply by -1 since <#> returns negative inner product
		query = fmt.Sprintf(
			"SELECT (%s <#> '%s') * -1 AS distance FROM %s",
			DefaultFieldVector,
			formattedVector,
			i.config.TableName,
		)
	case DistanceTypeCosine:
		// For cosine similarity, use 1 - cosine distance
		query = fmt.Sprintf(
			"SELECT 1 - (%s <=> '%s') AS distance FROM %s",
			DefaultFieldVector,
			formattedVector,
			i.config.TableName,
		)
	default:
		// For other distance types (L2, L1, etc.)
		query = fmt.Sprintf(
			"SELECT %s %s '%s' AS distance FROM %s",
			DefaultFieldVector,
			getDistanceOperator(distType),
			formattedVector,
			i.config.TableName,
		)
	}

	// Execute the query
	var distance float64
	err := i.db.QueryRowContext(ctx, query).Scan(&distance)
	if err != nil {
		return 0, fmt.Errorf("[PGVectorIndexer] distance calculation failed: %w", err)
	}

	return distance, nil
}

// Search performs a vector similarity search
func (i *Indexer) Search(ctx context.Context, queryVector []float64, opts *SearchOptions) ([]*SearchResult, error) {
	if opts == nil {
		opts = &SearchOptions{
			DistanceType: DistanceTypeL2,
			Limit:        10,
		}
	}

	if opts.Limit <= 0 {
		opts.Limit = 10
	}

	// Set search parameters based on index type
	if i.config.IndexType == IndexTypeHNSW && opts.HNSWOptions != nil {
		if opts.HNSWOptions.EFSearch > 0 {
			_, err := i.db.ExecContext(ctx, fmt.Sprintf("SET LOCAL hnsw.ef_search = %d", opts.HNSWOptions.EFSearch))
			if err != nil {
				return nil, fmt.Errorf("[PGVectorIndexer] failed to set hnsw.ef_search: %w", err)
			}
		}
		if opts.HNSWOptions.IterativeScan != "" {
			_, err := i.db.ExecContext(ctx, fmt.Sprintf("SET LOCAL hnsw.iterative_scan = %s", opts.HNSWOptions.IterativeScan))
			if err != nil {
				return nil, fmt.Errorf("[PGVectorIndexer] failed to set hnsw.iterative_scan: %w", err)
			}
		}
		if opts.HNSWOptions.MaxScanTuples > 0 {
			_, err := i.db.ExecContext(ctx, fmt.Sprintf("SET LOCAL hnsw.max_scan_tuples = %d", opts.HNSWOptions.MaxScanTuples))
			if err != nil {
				return nil, fmt.Errorf("[PGVectorIndexer] failed to set hnsw.max_scan_tuples: %w", err)
			}
		}
		if opts.HNSWOptions.ScanMemMultiplier > 0 {
			_, err := i.db.ExecContext(ctx, fmt.Sprintf("SET LOCAL hnsw.scan_mem_multiplier = %d", opts.HNSWOptions.ScanMemMultiplier))
			if err != nil {
				return nil, fmt.Errorf("[PGVectorIndexer] failed to set hnsw.scan_mem_multiplier: %w", err)
			}
		}
	} else if i.config.IndexType == IndexTypeIVFFlat && opts.IVFFlatOptions != nil {
		if opts.IVFFlatOptions.Probes > 0 {
			_, err := i.db.ExecContext(ctx, fmt.Sprintf("SET LOCAL ivfflat.probes = %d", opts.IVFFlatOptions.Probes))
			if err != nil {
				return nil, fmt.Errorf("[PGVectorIndexer] failed to set ivfflat.probes: %w", err)
			}
		}
		if opts.IVFFlatOptions.IterativeScan != "" {
			_, err := i.db.ExecContext(ctx, fmt.Sprintf("SET LOCAL ivfflat.iterative_scan = %s", opts.IVFFlatOptions.IterativeScan))
			if err != nil {
				return nil, fmt.Errorf("[PGVectorIndexer] failed to set ivfflat.iterative_scan: %w", err)
			}
		}
		if opts.IVFFlatOptions.MaxProbes > 0 {
			_, err := i.db.ExecContext(ctx, fmt.Sprintf("SET LOCAL ivfflat.max_probes = %d", opts.IVFFlatOptions.MaxProbes))
			if err != nil {
				return nil, fmt.Errorf("[PGVectorIndexer] failed to set ivfflat.max_probes: %w", err)
			}
		}
	}

	// Format the query vector
	formattedVector := i.formatVector(queryVector)

	// Build the SQL query based on distance type
	var query string
	switch opts.DistanceType {
	case DistanceTypeInnerProduct:
		// For inner product, multiply by -1 since <#> returns negative inner product
		query = fmt.Sprintf(
			"SELECT %s, %s, %s, (%s <#> '%s') * -1 AS distance FROM %s",
			DefaultFieldID,
			DefaultFieldContent,
			DefaultFieldMetadata,
			DefaultFieldVector,
			formattedVector,
			i.config.TableName,
		)
	case DistanceTypeCosine:
		// For cosine similarity, use 1 - cosine distance
		query = fmt.Sprintf(
			"SELECT %s, %s, %s, 1 - (%s <=> '%s') AS distance FROM %s",
			DefaultFieldID,
			DefaultFieldContent,
			DefaultFieldMetadata,
			DefaultFieldVector,
			formattedVector,
			i.config.TableName,
		)
	default:
		// For other distance types (L2, L1, etc.)
		query = fmt.Sprintf(
			"SELECT %s, %s, %s, %s %s '%s' AS distance FROM %s",
			DefaultFieldID,
			DefaultFieldContent,
			DefaultFieldMetadata,
			DefaultFieldVector,
			getDistanceOperator(opts.DistanceType),
			formattedVector,
			i.config.TableName,
		)
	}

	// Add filter condition if provided
	args := make([]interface{}, 0)
	if opts.Filter != "" {
		query += " WHERE " + opts.Filter
		args = append(args, opts.FilterParams...)
	}

	// Add distance threshold if provided
	if opts.MaxDistance != nil {
		if opts.Filter == "" {
			query += " WHERE "
		} else {
			query += " AND "
		}
		query += fmt.Sprintf("%s %s '%s' < $%d",
			DefaultFieldVector,
			getDistanceOperator(opts.DistanceType),
			formattedVector,
			len(args)+1,
		)
		args = append(args, *opts.MaxDistance)
	}

	// Add order by and limit
	query += fmt.Sprintf(" ORDER BY distance ASC LIMIT $%d", len(args)+1)
	args = append(args, opts.Limit)

	// Execute the query
	rows, err := i.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("[PGVectorIndexer] search query failed: %w", err)
	}
	defer rows.Close()

	// Process results
	results := make([]*SearchResult, 0, opts.Limit)
	for rows.Next() {
		var (
			id          string
			content     string
			metadataRaw string
			distance    float64
		)

		if err := rows.Scan(&id, &content, &metadataRaw, &distance); err != nil {
			return nil, fmt.Errorf("[PGVectorIndexer] failed to scan search result: %w", err)
		}

		// Parse metadata
		var metadata map[string]interface{}
		if err := json.Unmarshal([]byte(metadataRaw), &metadata); err != nil {
			return nil, fmt.Errorf("[PGVectorIndexer] failed to unmarshal metadata: %w", err)
		}

		results = append(results, &SearchResult{
			ID:       id,
			Content:  content,
			Metadata: metadata,
			Distance: distance,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("[PGVectorIndexer] error iterating search results: %w", err)
	}

	return results, nil
}

// getDistanceOperator 根据距离类型返回对应的 PGVector 操作符
func getDistanceOperator(distType DistanceType) string {
	switch distType {
	case DistanceTypeL2:
		return "<->" // L2 距离
	case DistanceTypeInnerProduct:
		return "<#>" // 内积
	case DistanceTypeCosine:
		return "<=>" // 余弦距离
	default:
		return "<->" // 默认 L2
	}
}

// SearchResult 表示向量检索的结果
type SearchResult struct {
	ID       string                 `json:"id"`
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata"`
	Distance float64                `json:"distance"`
}

func (i *Indexer) SearchByDocID(ctx context.Context, docID string, opts *SearchOptions) ([]*SearchResult, error) {
	// Get the vector for the document ID
	query := fmt.Sprintf(
		"SELECT %s FROM %s WHERE %s = $1",
		DefaultFieldVector,
		i.config.TableName,
		DefaultFieldID,
	)

	var vector pgvector.Vector
	err := i.db.QueryRowContext(ctx, query, docID).Scan(&vector)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("[PGVectorIndexer] document with ID %s not found", docID)
		}
		return nil, fmt.Errorf("[PGVectorIndexer] failed to get vector for document ID: %w", err)
	}

	// Add filter to exclude the query document itself
	filter := fmt.Sprintf("%s != $1::text", DefaultFieldID) // 明确指定类型为text
	filterParams := []interface{}{docID}

	// Merge with existing filter if any
	if opts != nil && opts.Filter != "" {
		// 调整现有过滤条件中的参数占位符编号
		adjustedFilter := opts.Filter
		for i := 1; i <= len(opts.FilterParams); i++ {
			// 使用正则表达式确保只替换独立的参数占位符
			oldParam := fmt.Sprintf("\\$%d([^0-9]|$)", i)
			newParam := fmt.Sprintf("$%d$1", i+1)
			re := regexp.MustCompile(oldParam)
			adjustedFilter = re.ReplaceAllString(adjustedFilter, newParam)
		}
		filter = filter + " AND (" + adjustedFilter + ")"
		filterParams = append(filterParams, opts.FilterParams...)
	}

	// Create new options with the filter
	newOpts := &SearchOptions{
		Limit:        10,
		MaxDistance:  nil,
		Filter:       filter,
		FilterParams: filterParams,
	}

	// Copy other options if provided
	if opts != nil {
		newOpts.Limit = opts.Limit
		newOpts.MaxDistance = opts.MaxDistance
		newOpts.HNSWOptions = opts.HNSWOptions
		newOpts.IVFFlatOptions = opts.IVFFlatOptions
	}

	// Convert float32 vector to float64 vector
	float64Vector := make([]float64, len(vector.Slice()))
	for i, v := range vector.Slice() {
		float64Vector[i] = float64(v)
	}

	// Perform the search
	return i.Search(ctx, float64Vector, newOpts)
}
