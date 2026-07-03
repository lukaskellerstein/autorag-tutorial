---
globs: ["tutorial/**"]
---

# Reference Sources

Always consult these sources when building lessons. Do NOT guess at APIs — read the docs and source code first.

## AutoRAG
- **GitHub**: https://github.com/Marker-Inc-Korea/AutoRAG
- **Documentation**: https://marker-inc-korea.github.io/AutoRAG/
- **Research Paper**: https://arxiv.org/html/2410.20878v1
- **Local Source Code**: /Users/lkellers/Projects/github/marker-inc-korea/AutoRAG
- Check for the latest API signatures before writing code

## Ollama
- **Website**: https://ollama.ai
- **Model library**: https://ollama.ai/library
- Default model: `gemma4:e2b`
- Used via LiteLLM in AutoRAG config as `ollama/gemma4:e2b`

## Embedding Models
- **sentence-transformers**: https://www.sbert.net/
- **MTEB Leaderboard**: https://huggingface.co/spaces/mteb/leaderboard
- Common models: `nomic-embed-text`, `bge-small-en`, `all-MiniLM-L6-v2`

## Vector Databases (AutoRAG-supported)
- **FAISS**: https://github.com/facebookresearch/faiss
- **Chroma**: https://docs.trychroma.com/
- **Qdrant**: https://qdrant.tech/documentation/

## How to Use References

1. **Before implementing a lesson**: read the AutoRAG docs to understand the relevant API or configuration
2. **Before using an API**: check the AutoRAG source code to verify it exists and get the correct signature
3. **Local source code**: browse `/Users/lkellers/Projects/github/marker-inc-korea/AutoRAG` for implementation details
4. **Check for deprecations**: AutoRAG is actively developed — verify APIs against the latest version
5. **Config YAML**: refer to AutoRAG docs for available node types, module types, and parameter options
