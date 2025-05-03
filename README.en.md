[English](README.en.md) | [中文](README.md)

# Eino-pgvector

Eino-pgvector is a PostgreSQL vector database extension based on the CloudWeGo Eino framework, integrating the pgvector plugin. It supports efficient vector retrieval, storage, and management, suitable for AI, recommendation systems, semantic search, and more.

This project is developed by myself. Please give it a star! 🥰

(Official integration PR is in progress...)

## Features

- Supports multiple vector index types (e.g., HNSW)
- Flexible metadata storage and filtering
- Integrated mainstream embedding services (e.g., Tencent Cloud)
- Supports Retriever semantic search
- Easy to extend and integrate

## Installation & Dependencies

### Requirements

- Go 1.18 or above
- PostgreSQL 13+ with [pgvector](https://github.com/pgvector/pgvector) plugin installed

### Dependency Installation

```bash
git clone https://github.com/Wood-Q/Eino-pgvector.git
cd Eino-pgvector
go mod tidy
```

### Database Preparation

1. If you already have PostgreSQL, install the pgvector plugin:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
2. If not, you can use `docker-compose.yaml` for one-click deployment:
   ```bash
   docker-compose up -d
   ```

## Quick Start

Refer to `examples/main.go` for usage. Main usage is as follows:

```go
import (
    "context"
    "github.com/Wood-Q/Eino-pgvector/indexer"
    "github.com/Wood-Q/Eino-pgvector/retriever"
    "github.com/cloudwego/eino-ext/components/embedding/tencentcloud"
    "github.com/cloudwego/eino/schema"
    "github.com/google/uuid"
)

func main() {
    ctx := context.Background()
    // Create embedder
    // Example uses Tencent Cloud
    // For configuration, refer to Eino official documentation
    cfg := &tencentcloud.EmbeddingConfig{SecretID: "<YourSecretID>", SecretKey: "<YourSecretKey>"}
    embedder, _ := tencentcloud.NewEmbedder(ctx, cfg)
    // Create indexer
    myindexer, _ := indexer.NewIndexer(ctx, &indexer.IndexerConfig{
        Host: "localhost", Port: 5432, User: "postgres", Password: "postgres", DBName: "vectorDB",
        SSLMode: "disable", TableName: "documents", Dimension: 1024, IndexType: "hnsw",
        IndexOptions: map[string]interface{}{ "m": 16, "ef_construction": 64 }, Embedding: embedder,
    })
    defer myindexer.Close()
    // Store documents
    docs := []*schema.Document{{ID: uuid.New().String(), Content: "123", MetaData: map[string]interface{}{ "source": "database_intro" }}}
    ids, _ := myindexer.Store(ctx, docs)
    // Vector search
    queryVectors, _ := embedder.EmbedStrings(ctx, []string{"91011"})
    results, _ := myindexer.Search(ctx, queryVectors[0], &indexer.SearchOptions{Limit: 5})
    // Create retriever
    myretriever, _ := retriever.NewRetriever(ctx, &retriever.RetrieverConfig{...})
    // Semantic search
    resultsment, _ := myretriever.Retrieve(ctx, "91011", &retriever.SearchOptions{Limit: 5})
}
```

## Main Directory Structure

```
Eino-pgvector/
├── indexer/      # Vector indexing and storage implementation
├── retriever/    # Retriever implementation
├── examples/     # Usage examples
├── init.sql      # Database initialization script
├── docker-compose.yaml # One-click deployment script
```

## Main Function Parameter Configurations

```go
    // RetrieverConfig parameters for retriever
    // IndexerConfig parameters for indexer
    // Both share the same parameters
    type RetrieverConfig struct {
        // PostgreSQL connection info
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

        // Embedding configuration
        Embedding embedding.Embedder `json:"embedding"`
    }

    // SearchResult represents search results
    type SearchResult struct {
        ID       string                 `json:"id"`
        Content  string                 `json:"content"`
        Metadata map[string]interface{} `json:"metadata"`
        Distance float64                `json:"distance"`
    }

    // HNSWSearchOptions for HNSW index search parameters
    type HNSWSearchOptions struct {
        // Size of dynamic candidate list during search (default: 40)
        EFSearch int
        // Whether to use iterative scan
        IterativeScan string // "strict_order" or "relaxed_order"
        // Max tuples to scan (default: 20000)
        MaxScanTuples int
        // Scan memory multiplier (default: 1)
        ScanMemMultiplier int
    }

    // IVFFlatSearchOptions for IVFFlat-specific search parameters
    type IVFFlatSearchOptions struct {
        // Number of probes (default: 1)
        Probes int
        // Whether to use iterative scan
        IterativeScan string // "strict_order" or "relaxed_order"
        // Max number of probes
        MaxProbes int
    }

    // SearchOptions for search options
    type SearchOptions struct {
        // Distance type used in search
        DistanceType DistanceType
        // Max number of results returned
        Limit int
        // Max distance threshold (optional)
        MaxDistance *float64
        // Additional filter conditions, SQL WHERE clause format (without WHERE keyword)
        Filter string
        // Filter parameters for prepared statements
        FilterParams []interface{}
        // HNSW-specific options
        HNSWOptions *HNSWSearchOptions
        // IVFFlat-specific options
        IVFFlatOptions *IVFFlatSearchOptions
    }
```

## License

This project is released under the MIT License.

## Contributing

Contributions via PR or Issue are welcome.
