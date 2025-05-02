package pgvector

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/schema"
)

// mockEmbedder implements a simple mock embedder for testing
type mockEmbedder struct{}

// Generate a simple vector for each text
func (m *mockEmbedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	vectors := make([][]float64, len(texts))
	for i := range texts {
		// 为每个文本生成一个简单的向量
		vectors[i] = make([]float64, 3)
		for j := range vectors[i] {
			vectors[i][j] = float64(i+j) / 10.0
		}
	}
	return vectors, nil
}

func (m *mockEmbedder) GetDimension() int {
	return 3
}

// testConfig database configuration for testing
var testConfig = &IndexerConfig{
	Host:      "localhost",
	Port:      5432,
	User:      "postgres",
	Password:  "postgres",
	DBName:    "vectorDB",
	SSLMode:   "disable",
	TableName: "test_vectors",
	Dimension: 3,
	Embedding: &mockEmbedder{},
}

func TestNewIndexer(t *testing.T) {
	tests := []struct {
		name    string
		config  *IndexerConfig
		wantErr bool
	}{
		{
			name:    "valid config",
			config:  testConfig,
			wantErr: false,
		},
		{
			name: "missing embedding",
			config: &IndexerConfig{
				Host:      testConfig.Host,
				Port:      testConfig.Port,
				User:      testConfig.User,
				Password:  testConfig.Password,
				DBName:    testConfig.DBName,
				SSLMode:   testConfig.SSLMode,
				TableName: testConfig.TableName,
				Dimension: testConfig.Dimension,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			indexer, err := NewIndexer(ctx, tt.config)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, indexer)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, indexer)
				if indexer != nil {
					defer indexer.Close()
				}
			}
		})
	}
}

func TestIndexer_Store(t *testing.T) {
	// Create indexer
	ctx := context.Background()
	indexer, err := NewIndexer(ctx, testConfig)
	require.NoError(t, err)
	defer indexer.Close()

	// Prepare test documents
	docs := []*schema.Document{
		{
			ID:      "doc1",
			Content: "test document 1",
			MetaData: map[string]interface{}{
				"source": "test",
			},
		},
		{
			ID:      "doc2",
			Content: "test document 2",
			MetaData: map[string]interface{}{
				"source": "test",
			},
		},
	}

	// Test storing documents
	ids, err := indexer.Store(ctx, docs)
	assert.NoError(t, err)
	assert.Len(t, ids, 2)
	assert.Equal(t, "doc1", ids[0])
	assert.Equal(t, "doc2", ids[1])
}

func TestVectorTypeValidation(t *testing.T) {
	tests := []struct {
		name      string
		vecType   VectorType
		dimension int
		wantErr   bool
	}{
		{
			name:      "valid vector type",
			vecType:   VectorTypeVector,
			dimension: 1000,
			wantErr:   false,
		},
		{
			name:      "invalid dimension for vector",
			vecType:   VectorTypeVector,
			dimension: 17000,
			wantErr:   true,
		},
		{
			name:      "valid halfvec type",
			vecType:   VectorTypeHalfvec,
			dimension: 1000,
			wantErr:   false,
		},
		{
			name:      "invalid dimension for halfvec",
			vecType:   VectorTypeHalfvec,
			dimension: 17000,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateVectorConfig(tt.vecType, tt.dimension)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestGetSuitableVectorType(t *testing.T) {
	tests := []struct {
		name      string
		dimension int
		want      VectorType
	}{
		{
			name:      "vector type for small dimension",
			dimension: 1000,
			want:      VectorTypeVector,
		},
		{
			name:      "vector type for medium dimension",
			dimension: 3000,
			want:      VectorTypeVector,
		},
		{
			name:      "bit type for large dimension",
			dimension: 50000,
			want:      VectorTypeBit,
		},
		{
			name:      "sparsevec type for very large dimension",
			dimension: 100000,
			want:      VectorTypeSparsevec,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getSuitableVectorType(tt.dimension)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestFormatVector(t *testing.T) {
	indexer := &Indexer{}
	vector := []float64{1.0, 2.0, 3.0}
	expected := "[1,2,3]"

	got := indexer.formatVector(vector)
	assert.Equal(t, expected, got)
}

func TestIndexer_CalculateDistance(t *testing.T) {
	// 创建索引器
	ctx := context.Background()
	indexer, err := NewIndexer(ctx, testConfig)
	require.NoError(t, err)
	defer indexer.Close()

	// 准备测试文档
	docs := []*schema.Document{
		{
			ID:      "dist_doc1",
			Content: "distance test document 1",
			MetaData: map[string]interface{}{
				"category": "test",
			},
		},
	}

	// 存储文档
	_, err = indexer.Store(ctx, docs)
	require.NoError(t, err)

	// 测试向量
	queryVector := []float64{1.0, 2.0, 3.0}

	// 测试L2距离
	distance, err := indexer.CalculateDistance(ctx, queryVector, DistanceTypeL2)
	assert.NoError(t, err)
	assert.Greater(t, distance, 0.0)

	// 测试内积
	distance, err = indexer.CalculateDistance(ctx, queryVector, DistanceTypeInnerProduct)
	assert.NoError(t, err)
	// 内积应该是正值
	assert.Greater(t, distance, 0.0)

	// 测试余弦相似度
	distance, err = indexer.CalculateDistance(ctx, queryVector, DistanceTypeCosine)
	assert.NoError(t, err)
	// 余弦相似度应该在0到1之间
	assert.GreaterOrEqual(t, distance, 0.0)
	assert.LessOrEqual(t, distance, 1.0)
}

func TestIndexer_SearchWithDifferentDistanceTypes(t *testing.T) {
	// 创建索引器
	ctx := context.Background()
	indexer, err := NewIndexer(ctx, testConfig)
	require.NoError(t, err)
	defer indexer.Close()

	// 准备测试文档
	docs := []*schema.Document{
		{
			ID:      "search_doc1",
			Content: "search test document 1",
			MetaData: map[string]interface{}{
				"category": "test",
				"priority": 1,
			},
		},
		{
			ID:      "search_doc2",
			Content: "search test document 2",
			MetaData: map[string]interface{}{
				"category": "test",
				"priority": 2,
			},
		},
	}

	// 存储文档
	_, err = indexer.Store(ctx, docs)
	require.NoError(t, err)

	// 测试向量
	queryVector := []float64{0.1, 0.2, 0.3}

	// 测试L2距离搜索
	results, err := indexer.Search(ctx, queryVector, &SearchOptions{
		DistanceType: DistanceTypeL2,
		Limit:        10,
	})
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
	// 验证结果按距离排序（从小到大）
	for i := 1; i < len(results); i++ {
		assert.LessOrEqual(t, results[i-1].Distance, results[i].Distance)
	}

	// 测试内积搜索
	results, err = indexer.Search(ctx, queryVector, &SearchOptions{
		DistanceType: DistanceTypeInnerProduct,
		Limit:        10,
	})
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
	// 验证内积结果按相似度排序（从大到小）
	for i := 1; i < len(results); i++ {
		assert.LessOrEqual(t, results[i].Distance, results[i-1].Distance)
	}

	// 测试余弦相似度搜索
	results, err = indexer.Search(ctx, queryVector, &SearchOptions{
		DistanceType: DistanceTypeCosine,
		Limit:        10,
	})
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
	// 验证余弦相似度结果按相似度排序（从大到小）
	for i := 1; i < len(results); i++ {
		assert.LessOrEqual(t, results[i].Distance, results[i-1].Distance)
	}
}

func TestIndexer_SearchWithHNSWOptions(t *testing.T) {
	// 创建索引器配置
	config := *testConfig
	config.IndexType = IndexTypeHNSW
	config.IndexOptions = map[string]interface{}{
		"m":               16,
		"ef_construction": 64,
	}

	// 创建索引器
	ctx := context.Background()
	indexer, err := NewIndexer(ctx, &config)
	require.NoError(t, err)
	defer indexer.Close()

	// 准备测试文档
	docs := []*schema.Document{
		{
			ID:      "hnsw_doc1",
			Content: "HNSW test document 1",
			MetaData: map[string]interface{}{
				"category": "test",
			},
		},
	}

	// 存储文档
	_, err = indexer.Store(ctx, docs)
	require.NoError(t, err)

	// 测试向量
	queryVector := []float64{0.1, 0.2, 0.3}

	// 测试带HNSW选项的搜索
	results, err := indexer.Search(ctx, queryVector, &SearchOptions{
		DistanceType: DistanceTypeL2,
		Limit:        10,
		HNSWOptions: &HNSWSearchOptions{
			EFSearch:          100,
			IterativeScan:     "relaxed_order",
			MaxScanTuples:     20000,
			ScanMemMultiplier: 1,
		},
	})
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
}

func TestIndexer_SearchWithIVFFlatOptions(t *testing.T) {
	// 创建索引器配置
	config := *testConfig
	config.IndexType = IndexTypeIVFFlat
	config.IndexOptions = map[string]interface{}{
		"lists": 100,
	}

	// 创建索引器
	ctx := context.Background()
	indexer, err := NewIndexer(ctx, &config)
	require.NoError(t, err)
	defer indexer.Close()

	// 准备测试文档
	docs := []*schema.Document{
		{
			ID:      "ivfflat_doc1",
			Content: "IVFFlat test document 1",
			MetaData: map[string]interface{}{
				"category": "test",
			},
		},
	}

	// 存储文档
	_, err = indexer.Store(ctx, docs)
	require.NoError(t, err)

	// 测试向量
	queryVector := []float64{0.1, 0.2, 0.3}

	// 测试带IVFFlat选项的搜索
	results, err := indexer.Search(ctx, queryVector, &SearchOptions{
		DistanceType: DistanceTypeL2,
		Limit:        10,
		IVFFlatOptions: &IVFFlatSearchOptions{
			Probes:        10,
			IterativeScan: "relaxed_order",
			MaxProbes:     100,
		},
	})
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
}

func TestIndexer_SearchByDocID(t *testing.T) {
	// 创建索引器
	ctx := context.Background()
	indexer, err := NewIndexer(ctx, testConfig)
	require.NoError(t, err)
	defer indexer.Close()

	// 准备测试文档
	docs := []*schema.Document{
		{
			ID:      "ref_doc1",
			Content: "reference document 1",
			MetaData: map[string]interface{}{
				"category": "reference",
			},
		},
		{
			ID:      "ref_doc2",
			Content: "reference document 2",
			MetaData: map[string]interface{}{
				"category": "reference",
			},
		},
		{
			ID:      "ref_doc3",
			Content: "reference document 3",
			MetaData: map[string]interface{}{
				"category": "other",
			},
		},
	}

	// 存储文档
	_, err = indexer.Store(ctx, docs)
	require.NoError(t, err)

	// 测试基于文档ID的搜索
	results, err := indexer.SearchByDocID(ctx, "ref_doc1", &SearchOptions{
		DistanceType: DistanceTypeL2,
		Limit:        10,
		Filter:       fmt.Sprintf("%s != $1", DefaultFieldID),
		FilterParams: []interface{}{"ref_doc1"},
	})

	assert.NoError(t, err)
	assert.NotEmpty(t, results)
	// 验证结果不包含查询文档本身
	for _, result := range results {
		assert.NotEqual(t, "ref_doc1", result.ID)
	}

	// 测试带过滤条件的搜索
	results, err = indexer.SearchByDocID(ctx, "ref_doc1", &SearchOptions{
		DistanceType: DistanceTypeL2,
		Limit:        10,
		Filter:       fmt.Sprintf("%s != $1 AND metadata->>'category' = $2", DefaultFieldID),
		FilterParams: []interface{}{"ref_doc1", "reference"},
	})

	assert.NoError(t, err)
	assert.NotEmpty(t, results)
	// 验证所有结果都是reference类别
	for _, result := range results {
		assert.Equal(t, "reference", result.Metadata["category"])
	}

	// 测试不存在的文档ID
	_, err = indexer.SearchByDocID(ctx, "non_existent_doc", nil)
	assert.Error(t, err)
}
