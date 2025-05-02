# Eino-pgvector

Eino-pgvector 是一个基于 PostgreSQL 和 pgvector 扩展的向量数据库集成包，为 Go 应用程序提供高效的向量存储和相似度搜索功能。它是 CloudWeGo Eino 项目的一部分，专门用于处理向量数据的存储和检索需求。

## 特性

- 简单易用的 API 接口
- 支持向量数据的存储和检索
- 集成 HNSW 索引，提供高效的近似最近邻搜索
- 支持元数据过滤
- 支持基于文档 ID 的相似度搜索
- 完全兼容 CloudWeGo Eino 生态系统

## 前置要求

- Go 1.24 或更高版本
- PostgreSQL 数据库
- pgvector 扩展

## 安装

```bash
go get github.com/Wood-Q/Eino-pgvector
```

## 快速开始

### example/main.go是使用的示例文件
### PostgreSQL的索引仅支持2000维度大小，所以推荐使用腾讯的混元模型生成，火山引擎的维度会超，希望注意一下

### 1. 配置数据库

可以根据仓库里提供的docker-compose和init.sql实现初始化，配置带有pgvector插件的数据库

### 2. 初始化 Indexer

```go
indexer, err := pgvector.NewIndexer(ctx, &pgvector.IndexerConfig{
    Host:      "localhost",
    Port:      5432,
    User:      "postgres",
    Password:  "your-password",
    DBName:    "your-database",
    SSLMode:   "disable",
    TableName: "documents",
    Dimension: 1024,//按你使用的向量转化模型实现
    IndexType: "hnsw",
    IndexOptions: map[string]interface{}{
        "m":               16,
        "ef_construction": 64,
    },
    Embedding: embedder, // 您需要提供一个实现了 Embedder 接口的嵌入模型
})
```

### 3. 存储文档

```go
docs := []*schema.Document{
    {
        ID:      uuid.New().String(),
        Content: "示例文档内容",
        MetaData: map[string]interface{}{
            "source":   "example_source",
            "category": "example_category",
        },
    },
}

ids, err := indexer.Store(ctx, docs)
```

### 4. 搜索文档

#### 基本搜索

```go
results, err := indexer.Search(ctx, queryVector, &pgvector.SearchOptions{
    Limit: 5, // 返回前5个结果
})
```

#### 带过滤条件的搜索

```go
results, err := indexer.Search(ctx, queryVector, &pgvector.SearchOptions{
    Limit:        5,
    Filter:       "metadata->>'category' = $1",
    FilterParams: []interface{}{"example_category"},
})
```

#### 基于文档 ID 的相似度搜索

```go
results, err := indexer.SearchByDocID(ctx, referenceID, &pgvector.SearchOptions{
    Limit: 5,
})
```

#### 使用 HNSW 索引选项

```go
results, err := indexer.Search(ctx, queryVector, &pgvector.SearchOptions{
    Limit: 5,
    HNSWOptions: &pgvector.HNSWSearchOptions{
        EFSearch: 40,
    },
})
```

## 高级配置

### HNSW 索引参数

- `m`: 每个节点的最大连接数（默认：16）
- `ef_construction`: 构建索引时的搜索宽度（默认：64）
- `ef_search`: 搜索时的搜索宽度，较大的值会提高搜索精度但会降低速度

## 许可证

本项目采用 Apache-2.0 许可证。详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！
