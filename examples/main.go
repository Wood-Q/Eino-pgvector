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

package main

import (
	"context"
	"fmt"
	"log"

	pgvector "github.com/Wood-Q/Eino-pgvector"
	"github.com/cloudwego/eino-ext/components/embedding/tencentcloud"
	"github.com/cloudwego/eino/schema"
	"github.com/google/uuid"
	_ "github.com/lib/pq"
)

func main() {
	ctx := context.Background()

	// 创建 embedder 配置
	cfg := &tencentcloud.EmbeddingConfig{
		SecretID:  "your-api-ID",
		SecretKey: "your-api-Key",
		Region:    "ap-guangzhou",
	}

	// 创建 embedder
	embedder, err := tencentcloud.NewEmbedder(ctx, cfg)
	if err != nil {
		panic(err)
	}

	// 创建 pgvector indexer
	indexer, err := pgvector.NewIndexer(ctx, &pgvector.IndexerConfig{
		Host:      "localhost",
		Port:      5433,
		User:      "postgres",
		Password:  "123456",
		DBName:    "vectorDB",
		SSLMode:   "disable",
		TableName: "documents",
		Dimension: 1024,
		IndexType: "hnsw",
		IndexOptions: map[string]interface{}{
			"m":               16,
			"ef_construction": 64,
		},
		Embedding: embedder,
	})
	if err != nil {
		log.Fatalf("创建indexer失败: %v", err)
	}
	defer indexer.Close()

	// 创建文档
	docs := []*schema.Document{
		{
			ID:      uuid.New().String(),
			Content: "123",
			MetaData: map[string]interface{}{
				"source":   "database_intro",
				"author":   "postgres_team",
				"category": "database",
			},
		},
		{
			ID:      uuid.New().String(),
			Content: "223",
			MetaData: map[string]interface{}{
				"source":   "pgvector_intro",
				"author":   "pgvector_team",
				"category": "extension",
			},
		},
		{
			ID:      uuid.New().String(),
			Content: "345",
			MetaData: map[string]interface{}{
				"source":   "vector_db_intro",
				"author":   "db_expert",
				"category": "database",
			},
		},
	}

	// 存储文档
	ids, err := indexer.Store(ctx, docs)
	if err != nil {
		log.Fatalf("存储文档失败: %v", err)
	}

	fmt.Printf("成功存储 %d 个文档，ID: %v\n\n", len(ids), ids)

	// 示例1: 基本向量搜索
	fmt.Println("=== 示例1: 基本向量搜索 ===")
	// 假设我们已经有了查询向量
	queryText := "PostgreSQL数据库的向量搜索功能"
	queryVectors, err := embedder.EmbedStrings(ctx, []string{queryText})
	if err != nil {
		log.Fatalf("生成查询向量失败: %v", err)
	}
	queryVector := queryVectors[0]

	// 执行向量搜索
	results, err := indexer.Search(ctx, queryVector, &pgvector.SearchOptions{
		Limit: 5, // 返回前5个结果
	})
	if err != nil {
		log.Fatalf("搜索失败: %v", err)
	}

	// 显示搜索结果
	fmt.Printf("查询: %s\n", queryText)
	fmt.Println("搜索结果:")
	for i, result := range results {
		fmt.Printf("  %d. ID: %s\n     内容: %s\n     距离: %.4f\n     元数据: %v\n\n",
			i+1, result.ID, result.Content, result.Distance, result.Metadata)
	}

	// 示例2: 带过滤条件的搜索
	fmt.Println("=== 示例2: 带过滤条件的搜索 ===")
	results, err = indexer.Search(ctx, queryVector, &pgvector.SearchOptions{
		Limit:        5,
		Filter:       "metadata->>'category' = $1", // 只搜索特定类别
		FilterParams: []interface{}{"database"},
	})
	if err != nil {
		log.Fatalf("带过滤条件的搜索失败: %v", err)
	}

	fmt.Printf("查询: %s (仅限 'database' 类别)\n", queryText)
	fmt.Println("搜索结果:")
	for i, result := range results {
		fmt.Printf("  %d. ID: %s\n     内容: %s\n     距离: %.4f\n     元数据: %v\n\n",
			i+1, result.ID, result.Content, result.Distance, result.Metadata)
	}

	// 示例3: 基于文档ID的搜索
	fmt.Println("=== 示例3: 基于文档ID的搜索 ===")
	// 使用第一个文档作为查询基准
	referenceID := ids[0]
	results, err = indexer.SearchByDocID(ctx, referenceID, &pgvector.SearchOptions{
		Limit: 5,
	})
	if err != nil {
		log.Fatalf("基于文档ID的搜索失败: %v", err)
	}

	fmt.Printf("查询文档ID: %s\n", referenceID)
	fmt.Println("搜索结果:")
	for i, result := range results {
		fmt.Printf("  %d. ID: %s\n     内容: %s\n     距离: %.4f\n     元数据: %v\n\n",
			i+1, result.ID, result.Content, result.Distance, result.Metadata)
	}

	// 示例4: 使用 HNSW 索引选项
	fmt.Println("=== 示例4: 使用 HNSW 索引选项 ===")
	results, err = indexer.Search(ctx, queryVector, &pgvector.SearchOptions{
		Limit: 5,
		HNSWOptions: &pgvector.HNSWSearchOptions{
			EFSearch: 40,
		},
	})
	if err != nil {
		log.Fatalf("使用 HNSW 选项的搜索失败: %v", err)
	}

	fmt.Printf("查询: %s (使用 HNSW 选项)\n", queryText)
	fmt.Println("搜索结果:")
	for i, result := range results {
		fmt.Printf("  %d. ID: %s\n     内容: %s\n     距离: %.4f\n     元数据: %v\n\n",
			i+1, result.ID, result.Content, result.Distance, result.Metadata)
	}
}
