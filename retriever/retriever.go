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

package retriever

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/components/embedding"
)

// RetrieverConfig 配置检索器的参数
type RetrieverConfig struct {
	// PostgreSQL 连接信息
	Host     string `json:"host"`
	Port     int    `json:"port"`
	User     string `json:"user"`
	Password string `json:"password"`
	DBName   string `json:"db_name"`
	SSLMode  string `json:"ssl_mode"`

	// 表名
	TableName string `json:"table_name"`

	// 向量维度
	Dimension int `json:"dimension"`

	// 向量化配置
	Embedding embedding.Embedder `json:"embedding"`
}

// SearchResult 表示检索结果
type SearchResult struct {
	ID       string                 `json:"id"`
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata"`
	Distance float64                `json:"distance"`
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

// SearchOptions 配置检索选项
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

// Retriever 实现向量检索功能
type Retriever struct {
	config *RetrieverConfig
	db     *sql.DB
}

// NewRetriever 创建新的检索器实例
func NewRetriever(ctx context.Context, config *RetrieverConfig) (*Retriever, error) {
	// 验证配置
	if config.Embedding == nil {
		return nil, fmt.Errorf("[PGVectorRetriever] embedding is required")
	}

	// 构建连接字符串
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		config.Host, config.Port, config.User, config.Password, config.DBName, config.SSLMode)

	// 连接数据库
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("[PGVectorRetriever] failed to connect to database: %w", err)
	}

	// 测试连接
	if err = db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("[PGVectorRetriever] failed to ping database: %w", err)
	}

	return &Retriever{
		config: config,
		db:     db,
	}, nil
}

// Retrieve 执行向量相似性搜索
func (r *Retriever) Retrieve(ctx context.Context, query string, opts *SearchOptions) ([]*SearchResult, error) {
	// 使用嵌入模型将查询转换为向量
	queryVectors, err := r.config.Embedding.EmbedStrings(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("[PGVectorRetriever] failed to embed query: %w", err)
	}
	queryVector := queryVectors[0]

	// 设置默认选项
	if opts == nil {
		opts = &SearchOptions{
			DistanceType: DistanceTypeL2,
			Limit:        10,
		}
	}

	if opts.Limit <= 0 {
		opts.Limit = 10
	}

	// 格式化查询向量
	formattedVector := formatVector(queryVector)
	switch opts.DistanceType {
	case DistanceTypeInnerProduct:
		// 对于内积，乘以-1因为<#>返回负内积
		query = fmt.Sprintf(
			"SELECT %s, %s, %s, (%s <#> '%s') * -1 AS distance FROM %s",
			DefaultFieldID,
			DefaultFieldContent,
			DefaultFieldMetadata,
			DefaultFieldVector,
			formattedVector,
			r.config.TableName,
		)
	case DistanceTypeCosine:
		// 对于余弦相似度，使用1减去余弦距离
		query = fmt.Sprintf(
			"SELECT %s, %s, %s, 1 - (%s <=> '%s') AS distance FROM %s",
			DefaultFieldID,
			DefaultFieldContent,
			DefaultFieldMetadata,
			DefaultFieldVector,
			formattedVector,
			r.config.TableName,
		)
	default:
		// 对于其他距离类型（L2, L1等）
		query = fmt.Sprintf(
			"SELECT %s, %s, %s, %s %s '%s' AS distance FROM %s",
			DefaultFieldID,
			DefaultFieldContent,
			DefaultFieldMetadata,
			DefaultFieldVector,
			getDistanceOperator(opts.DistanceType),
			formattedVector,
			r.config.TableName,
		)
	}

	// 添加过滤条件
	args := make([]interface{}, 0)
	if opts.Filter != "" {
		query += " WHERE " + opts.Filter
		args = append(args, opts.FilterParams...)
	}

	// 添加距离阈值
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

	// 添加排序和限制
	query += fmt.Sprintf(" ORDER BY distance ASC LIMIT $%d", len(args)+1)
	args = append(args, opts.Limit)

	// 执行查询
	rows, err := r.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("[PGVectorRetriever] search query failed: %w", err)
	}
	defer rows.Close()

	// 处理结果
	results := make([]*SearchResult, 0, opts.Limit)
	for rows.Next() {
		var (
			id          string
			content     string
			metadataRaw string
			distance    float64
		)

		if err := rows.Scan(&id, &content, &metadataRaw, &distance); err != nil {
			return nil, fmt.Errorf("[PGVectorRetriever] failed to scan search result: %w", err)
		}

		// 解析元数据
		var metadata map[string]interface{}
		if err := json.Unmarshal([]byte(metadataRaw), &metadata); err != nil {
			return nil, fmt.Errorf("[PGVectorRetriever] failed to unmarshal metadata: %w", err)
		}

		results = append(results, &SearchResult{
			ID:       id,
			Content:  content,
			Metadata: metadata,
			Distance: distance,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("[PGVectorRetriever] error iterating search results: %w", err)
	}

	return results, nil
}

// Close 关闭数据库连接
func (r *Retriever) Close() error {
	if r.db != nil {
		return r.db.Close()
	}
	return nil
}

// formatVector 将向量格式化为PostgreSQL向量格式
func formatVector(vector []float64) string {
	strVals := make([]string, len(vector))
	for i, v := range vector {
		strVals[i] = fmt.Sprintf("%f", v)
	}
	return "[" + strings.Join(strVals, ",") + "]"
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
