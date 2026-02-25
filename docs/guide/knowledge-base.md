# 知识库集成指南

## 概述

SwiftAgent 的知识库模块提供统一的文档存储和检索抽象层。内置 `MemoryKnowledgeBase`（纯 stdlib，适用于测试和小规模数据），也支持自定义实现对接 Weaviate、Elasticsearch 等。

## 快速开始

```python
from swiftagentx import Agent, DummyModelClient, MemoryKnowledgeBase, Document

# 1. 创建知识库并添加文档
kb = MemoryKnowledgeBase()
await kb.add_documents([
    Document(doc_id="1", content="退货政策：7天无理由退换货"),
    Document(doc_id="2", content="会员积分可在商城兑换礼品"),
])

# 2. 关联到 Agent
agent = Agent(model=my_model)
agent.set_knowledge_base(kb)  # 自动注册 KnowledgeBaseTool
```

## 数据模型

### Document

```python
class Document(BaseModel):
    doc_id: str                     # 唯一标识
    content: str                    # 正文内容
    metadata: Dict[str, Any] = {}   # 自定义元数据
```

### SearchResult

```python
class SearchResult(BaseModel):
    document: Document
    score: float        # 相似度 0~1
    match_type: str     # "exact" | "semantic" | "keyword"
```

## KnowledgeBase 抽象基类

```python
class KnowledgeBase(ABC):
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]: ...
    async def add_documents(self, documents: List[Document]) -> int: ...
    async def delete_document(self, doc_id: str) -> bool: ...
    async def get_document(self, doc_id: str) -> Optional[Document]: ...
    async def count(self) -> int: ...
    async def close(self) -> None: ...
```

## MemoryKnowledgeBase

纯内存实现，零外部依赖：

- **精准匹配**: 内容完全相同 → `score=1.0, match_type="exact"`
- **关键词匹配**: TF-IDF + 余弦相似度 → `score=0~1, match_type="keyword"`
- **中文支持**: 单字切分（无需分词库）
- **适用场景**: 测试、小规模数据 (<10000 条)

```python
from swiftagentx import MemoryKnowledgeBase, Document

kb = MemoryKnowledgeBase()
await kb.add_documents([
    Document(doc_id="faq-1", content="如何退货？联系客服申请即可。"),
])

results = await kb.search("退货")
for r in results:
    print(f"[{r.match_type} score={r.score}] {r.document.content}")
```

## KnowledgeBaseTool

`agent.set_knowledge_base(kb)` 会自动注册一个 `KnowledgeBaseTool`：

- 名称: `"knowledge_base"`
- 高分匹配 (score >= threshold) → `DIRECT_OUTPUT`（直接返回）
- 低分匹配 → `LLM_PROCESSED`（LLM 加工后回复）

可通过 `SwiftAgentConfig.kb_exact_match_threshold` 配置阈值（默认 0.95）。

## KnowledgeBaseStage

将 KB 查询插入到请求管道，实现精准匹配短路：

```python
from swiftagentx import KnowledgeBaseStage

stage = KnowledgeBaseStage(kb=my_kb, threshold=0.95)
agent.pipeline.add_stage(stage)
```

执行逻辑：
- 精准匹配 → `SHORT_CIRCUIT`，直接返回文档内容
- 模糊匹配 → 存入 `context["kb_results"]`，继续后续阶段

## 自定义实现

### Weaviate 示例

```python
import aiohttp
from swiftagentx.knowledge_base import KnowledgeBase, Document, SearchResult

class WeaviateKnowledgeBase(KnowledgeBase):
    def __init__(self, url: str, api_key: str, collection: str):
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.collection = collection

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        payload = {
            "query": query,
            "limit": top_k,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.url}/v1/search",
                json=payload,
                headers=self.headers,
            ) as resp:
                data = await resp.json(content_type=None)

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                document=Document(
                    doc_id=item["id"],
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                ),
                score=item.get("score", 0.0),
                match_type="semantic",
            ))
        return results

    async def add_documents(self, documents: list[Document]) -> int:
        # ... 实现文档上传 ...
        pass

    async def delete_document(self, doc_id: str) -> bool:
        # ... 实现文档删除 ...
        pass
```

### Elasticsearch 示例

```python
class ElasticsearchKnowledgeBase(KnowledgeBase):
    def __init__(self, es_url: str, index: str):
        self.es_url = es_url
        self.index = index

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        payload = {
            "query": {"match": {"content": query}},
            "size": top_k,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.es_url}/{self.index}/_search",
                json=payload,
            ) as resp:
                data = await resp.json(content_type=None)

        results = []
        for hit in data.get("hits", {}).get("hits", []):
            results.append(SearchResult(
                document=Document(
                    doc_id=hit["_id"],
                    content=hit["_source"]["content"],
                ),
                score=hit.get("_score", 0.0) / 10.0,  # 归一化
                match_type="keyword",
            ))
        return results

    # ... add_documents, delete_document ...
```
