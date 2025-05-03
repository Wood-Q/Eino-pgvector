[English](README.en.md) | [中文](README.md)

# Eino-pgvector

Eino-pgvector 是基于 CloudWeGo Eino 框架的 PostgreSQL 向量数据库扩展，集成了 pgvector 插件，支持高效的向量检索、存储与管理，适用于 AI、推荐系统、语义搜索等场景。

本项目由我个人开发，求求 star 啦！🥰

（官方集成功能 PR 提出中...）

## 特性

- 支持多种向量索引类型（如 HNSW）
- 灵活的元数据存储与过滤
- 集成主流 Embedding 服务（如腾讯云）
- 支持 Retriever 语义检索
- 易于扩展与集成

## 安装与依赖

### 环境要求

- Go 1.18 及以上
- PostgreSQL 13+，需安装 [pgvector](https://github.com/pgvector/pgvector) 插件

### 依赖安装

```bash
git clone https://github.com/Wood-Q/Eino-pgvector.git
cd Eino-pgvector
go mod tidy
```

### 数据库准备

1. 如果已经有 postgreSQL，安装 pgvector 插件：
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
2. 如果还没有，可以使用 `docker-compose.yaml` 一键部署：
   ```bash
   docker-compose up -d
   ```

## 快速开始

参考 `examples/main.go`，以下为主要用法：

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
    // 创建 embedder
    // 示例使用腾讯云
    // 相关配置参考Eino官方文档
    cfg := &tencentcloud.EmbeddingConfig{SecretID: "<YourSecretID>", SecretKey: "<YourSecretKey>"}
    embedder, _ := tencentcloud.NewEmbedder(ctx, cfg)
    // 创建 indexer
    myindexer, _ := indexer.NewIndexer(ctx, &indexer.IndexerConfig{
        Host: "localhost", Port: 5432, User: "postgres", Password: "postgres", DBName: "vectorDB",
        SSLMode: "disable", TableName: "documents", Dimension: 1024, IndexType: "hnsw",
        IndexOptions: map[string]interface{}{ "m": 16, "ef_construction": 64 }, Embedding: embedder,
    })
    defer myindexer.Close()
    // 存储文档
    docs := []*schema.Document{{ID: uuid.New().String(), Content: "123", MetaData: map[string]interface{}{ "source": "database_intro" }}}
    ids, _ := myindexer.Store(ctx, docs)
    // 通过向量搜索
    queryVectors, _ := embedder.EmbedStrings(ctx, []string{"91011"})
    results, _ := myindexer.Search(ctx, queryVectors[0], &indexer.SearchOptions{Limit: 5})
    // 创建 retriever
    myretriever, _ := retriever.NewRetriever(ctx, &retriever.RetrieverConfig{...})
    // 通过语义搜索
    resultsment, _ := myretriever.Retrieve(ctx, "91011", &retriever.SearchOptions{Limit: 5})
}
```

## 主要目录结构

```
Eino-pgvector/
├── indexer/      # 向量索引与存储实现
├── retriever/    # 检索器实现
├── examples/     # 使用示例
├── init.sql      # 数据库初始化脚本
├── docker-compose.yaml # 一键部署脚本
```

## 相关函数参数配置

```go
    // RetrieverConfig 配置检索器的参数
    // IndexerConfig 配置索引器的参数
    // 两个的参数是一样的
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

    // HNSWSearchOptions 配置 HNSW 索引的搜索参数
    type HNSWSearchOptions struct {
        // 搜索时动态候选列表的大小（默认值：40）
        EFSearch int
        // 是否使用迭代扫描
        IterativeScan string // "strict_order" 或 "relaxed_order"
        // 允许访问的最大元组数（默认值：20000）
        MaxScanTuples int
        // 扫描时的内存倍数（默认值：1）
        ScanMemMultiplier int
    }

    // IVFFlatSearchOptions 配置 IVFFlat 特定搜索参数
    type IVFFlatSearchOptions struct {
        // 探针数量（默认值：1）
        Probes int
        // 是否使用迭代扫描
        IterativeScan string // "strict_order" 或 "relaxed_order"
        // 最大探针数
        MaxProbes int
    }

    // SearchOptions 配置检索选项
    type SearchOptions struct {
        // 搜索时使用的距离类型
        DistanceType DistanceType
        // 返回结果的最大数量
        Limit int
        // 最大距离阈值（可选）
        MaxDistance *float64
        // 额外的过滤条件，SQL WHERE 子句格式（不含 WHERE 关键字）
        Filter string
        // 预处理语句的过滤参数
        FilterParams []interface{}
        // HNSW 特定选项
        HNSWOptions *HNSWSearchOptions
        // IVFFlat 特定选项
        IVFFlatOptions *IVFFlatSearchOptions
    }
```

## 许可证

本项目基于 MITLicense 发布。

## 欢迎贡献

欢迎通过 PR 或 Issue 来贡献代码或提出问题。
