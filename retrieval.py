"""CPU semantic retrieval over numbered rules and glossary passages."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CACHE = Path(".cache/huggingface/hub")


def load_reranker(revision=None, local_files_only=False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        RERANKER, revision=revision, cache_dir=CACHE, local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER, revision=revision, cache_dir=CACHE, use_safetensors=True,
        local_files_only=local_files_only,
    ).to("cpu").eval()
    return tokenizer, model


class Encoder:
    def __init__(self, revision=None, local_files_only=False):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=revision, cache_dir=CACHE, local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            MODEL, revision=revision, cache_dir=CACHE, use_safetensors=True,
            local_files_only=local_files_only,
        ).to("cpu").eval()
        self.revision = self.model.config._commit_hash

    def encode(self, texts):
        import torch
        import torch.nn.functional as functional

        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return functional.normalize(pooled, p=2, dim=1).numpy()


def build_index(index, destination=Path("data/rule_vectors.npz")):
    import numpy as np

    texts = []
    identifiers = []
    for passage in index.passages:
        words = passage.text.split()
        for offset in range(0, len(words), 100):
            texts.append(f"{passage.heading}\n{' '.join(words[offset:offset + 120])}")
            identifiers.append(passage.id)
    encoder = Encoder()
    batches = []
    for offset in range(0, len(texts), 32):
        batches.append(encoder.encode(texts[offset:offset + 32]))
        if offset % 320 == 0:
            print(f"Encoded {min(offset + 32, len(texts))}/{len(texts)} rule chunks", flush=True)
    metadata = {
        "model": MODEL, "revision": encoder.revision,
        "source_sha256": index.source_hash, "chunks": len(texts),
    }
    _, reranker = load_reranker()
    metadata["reranker"] = RERANKER
    metadata["reranker_revision"] = reranker.config._commit_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as output:
        temporary = Path(output.name)
        try:
            np.savez_compressed(output, vectors=np.concatenate(batches), ids=np.array(identifiers),
                                metadata=json.dumps(metadata))
        except Exception:
            output.close()
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    print(f"Saved {destination.resolve()}\n{json.dumps(metadata, indent=2)}")


class SemanticIndex:
    def __init__(self, path, source_hash):
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.vectors = data["vectors"]
            self.ids = data["ids"].tolist()
        if self.metadata["source_sha256"] != source_hash:
            raise ValueError("The semantic index uses different rules. Run rules.py index to rebuild it.")
        self.encoder = None
        self.reranker = None

    def rerank(self, query, passages):
        import torch
        if self.reranker is None:
            tokenizer, model = load_reranker(self.metadata.get("reranker_revision"), local_files_only=True)
            self.reranker = (tokenizer, model)
            self.metadata["reranker"] = RERANKER
            self.metadata["reranker_revision"] = model.config._commit_hash
        tokenizer, model = self.reranker
        query = tokenizer.decode(tokenizer.encode(
            query, add_special_tokens=False, truncation=True, max_length=192,
        ))
        scores = []
        for offset in range(0, len(passages), 16):
            group = passages[offset:offset + 16]
            texts = [f"{p.heading}\n{p.text}" for p in group]
            batch = tokenizer([query] * len(group), texts, padding=True,
                              truncation="only_second", max_length=512, return_tensors="pt")
            with torch.inference_mode():
                scores.extend(model(**batch).logits[:, 0].tolist())
        return [passages[position].id for position in sorted(
            range(len(passages)), key=lambda position: (-scores[position], passages[position].id),
        )]

    def search(self, query, limit):
        import numpy as np

        if self.encoder is None:
            self.encoder = Encoder(self.metadata["revision"], local_files_only=True)
        vector = self.encoder.encode([query])[0]
        scores = self.vectors @ vector
        ranked = np.argsort(-scores, kind="stable")
        result = []
        seen = set()
        for position in ranked:
            identifier = self.ids[position]
            if identifier not in seen:
                result.append(identifier)
                seen.add(identifier)
                if len(result) >= limit:
                    break
        return result
