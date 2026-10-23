"""
The RAG pipeline, split out of app.py so it can be tested and evaluated.

Four interchangeable readers turn retrieved chunks into an answer:
  * extractive (default): DistilBERT copies a span from the context
  * generative: Flan-T5, an instruction-tuned LLM, writes the answer and is
    told to reply "unanswerable" when the context lacks it
  * azure: a hosted GPT model on Azure OpenAI with structured JSON output,
    a calculator function call and guardrails (ragbot/azure_reader.py)
  * llm: the same reader against any OpenAI-compatible provider (Gemini,
    Groq, Mistral, OpenRouter, or a local Ollama server)

Default settings reproduce the original app exactly:
  fixed 900-character chunks, no overlap, top-6 retrieval, and an answer
  is always returned. `overlap` and `min_score` exist so the evaluation
  harness can measure whether changing them helps.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import faiss
import numpy as np

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
GEN_MODEL_NAME = "google/flan-t5-base"
NO_ANSWER = "Not found in the document."

# The generative model is told to reply with this word when the context
# doesn't contain the answer; is_refusal() also catches common variants.
REFUSAL_TOKEN = "unanswerable"
_REFUSALS = {
    "unanswerable", "not in context", "not in the context", "not mentioned",
    "not stated", "no answer", "i dont know", "unknown", "none", "not found",
    "not specified", "not provided", "cannot answer",
}

GEN_PROMPT = (
    "Answer the question using only the context below. Copy the exact amount, "
    "rate or phrase from the context. If the context does not contain the answer, "
    "reply with exactly: unanswerable\n\n"
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)


def build_prompt(question: str, context: str) -> str:
    return GEN_PROMPT.format(context=context, question=question)


def is_refusal(text: str) -> bool:
    t = "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").strip()
    return t in _REFUSALS or t.startswith("unanswerable")

# An embedder maps a list of strings to a 2-D float array.
Embedder = Callable[[List[str]], np.ndarray]
# A QA model takes (question, context) and returns {"answer": str, "score": float}.
QAModel = Callable[[str, str], dict]


def read_pdf(file) -> str:
    """Extract text from every page of a PDF (path or file-like object)."""
    from pypdf import PdfReader

    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def chunk_text(text: str, size: int = 900, overlap: int = 0) -> List[str]:
    """
    Split text into fixed-size character chunks.

    overlap=0 is the original behaviour. A positive overlap repeats the last
    `overlap` characters at the start of the next chunk, so a sentence cut at
    a chunk boundary still appears whole in one chunk.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
    return chunks


@dataclass
class Answer:
    text: str
    score: float
    abstained: bool
    chunk_ids: List[int] = field(default_factory=list)
    context: str = ""
    raw: str = ""            # model output before abstention was applied
    truncated: bool = False  # generative reader: context cut to fit 512 tokens
    trace: dict = field(default_factory=dict)  # Azure reader: tool calls, quote, guardrail decisions


def load_embedder() -> Embedder:
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(EMBED_MODEL_NAME)
    return lambda texts: st_model.encode(texts)


def extract_answer(tokenizer, model, question: str, context: str, max_length: int = 384,
                   stride: int = 128, max_answer_tokens: int = 30) -> dict:
    """
    Extractive QA without transformers.pipeline().

    transformers 5.x removed the "question-answering" pipeline the original app
    used (KeyError on first question), so this does the same job directly:
    split long contexts into overlapping windows, score every valid
    (start, end) span inside the context, and return the best one. The score is
    p(start) * p(end), the same confidence the old pipeline reported.
    Works on transformers 4.x and 5.x.
    """
    import torch

    enc = tokenizer(question, context, truncation="only_second", max_length=max_length,
                    stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True,
                    padding="max_length", return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    inputs = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    if "token_type_ids" in inputs and "token_type_ids" not in model.forward.__code__.co_varnames:
        inputs.pop("token_type_ids")  # DistilBERT has no segment embeddings
    with torch.no_grad():
        out = model(**inputs)

    best = {"answer": "", "score": 0.0, "start": 0, "end": 0}
    for w in range(out.start_logits.shape[0]):
        seq_ids = enc.sequence_ids(w)
        in_ctx = torch.tensor([sid == 1 for sid in seq_ids])
        if not in_ctx.any():
            continue
        start = out.start_logits[w].masked_fill(~in_ctx, float("-inf")).softmax(-1)
        end = out.end_logits[w].masked_fill(~in_ctx, float("-inf")).softmax(-1)
        # score[i, j] = p(start=i) * p(end=j), only for i <= j < i + max_answer_tokens
        scores = start[:, None] * end[None, :]
        scores = torch.triu(scores) - torch.triu(scores, diagonal=max_answer_tokens)
        flat = int(scores.argmax())
        i, j = divmod(flat, scores.shape[1])
        score = float(scores[i, j])
        if score > best["score"]:
            c0, c1 = int(offsets[w][i][0]), int(offsets[w][j][1])
            best = {"answer": context[c0:c1].strip(), "score": score, "start": c0, "end": c1}
    return best


def load_extractive_reader(model_name: str = QA_MODEL_NAME) -> QAModel:
    """DistilBERT fine-tuned on SQuAD: copies a span out of the context."""
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return lambda question, context: extract_answer(tok, model, question, context)


def load_generative_reader(model_name: str = GEN_MODEL_NAME, max_new_tokens: int = 32) -> QAModel:
    """
    Flan-T5, an instruction-tuned generative language model: writes the answer.

    Confidence = geometric mean of the generated tokens' probabilities, so the
    same min_score threshold works for both readers. Inputs longer than the
    model's 512-token window are truncated (the end of the context is cut),
    which is why smaller chunks / lower k suit this reader.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    def answer(question: str, context: str) -> dict:
        inputs = tok(build_prompt(question, context), return_tensors="pt",
                     truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1,
                                 do_sample=False, output_scores=True, return_dict_in_generate=True)
        text = tok.decode(out.sequences[0], skip_special_tokens=True).strip()
        logprobs = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
        score = float(torch.exp(logprobs.mean())) if len(logprobs) else 0.0
        truncated = inputs["input_ids"].shape[1] >= 512
        return {"answer": text, "score": score, "refused": is_refusal(text), "truncated": truncated}

    return answer


READERS = ("extractive", "generative", "azure", "llm")


def load_reader(reader: str) -> QAModel:
    if reader == "extractive":
        return load_extractive_reader()
    if reader == "generative":
        return load_generative_reader()
    if reader == "azure":
        from ragbot.azure_reader import load_azure_reader
        return load_azure_reader()
    if reader == "llm":
        from ragbot.azure_reader import load_compatible_reader
        return load_compatible_reader()
    raise ValueError(f"Unknown reader {reader!r}; use one of {', '.join(READERS)}")


def load_models(reader: str = "extractive"):
    """Load the embedder and a reader (local models download from Hugging Face on first use)."""
    if reader not in READERS:
        raise ValueError(f"Unknown reader {reader!r}; use one of {', '.join(READERS)}")
    return load_embedder(), load_reader(reader)


class RagPipeline:
    def __init__(
        self,
        embed: Embedder,
        qa: QAModel,
        chunk_size: int = 900,
        overlap: int = 0,
        k: int = 6,
        min_score: Optional[float] = None,
    ):
        self.embed = embed
        self.qa = qa
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.k = k
        self.min_score = min_score  # None = always answer (original behaviour)
        self.chunks: List[str] = []
        self.index = None

    def index_text(self, text: str) -> "RagPipeline":
        self.chunks = chunk_text(text, self.chunk_size, self.overlap)
        if not self.chunks:
            raise ValueError("No text to index (is the PDF scanned images?)")
        embeddings = np.asarray(self.embed(self.chunks), dtype="float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        return self

    def retrieve(self, question: str) -> List[int]:
        if self.index is None:
            raise RuntimeError("Call index_text() first")
        q = np.asarray(self.embed([question]), dtype="float32")
        _, ids = self.index.search(q, min(self.k, len(self.chunks)))
        return [int(i) for i in ids[0] if i >= 0]

    def ask(self, question: str) -> Answer:
        ids = self.retrieve(question)
        context = " ".join(self.chunks[i] for i in ids)
        result = self.qa(question, context)
        raw = str(result["answer"]).strip()
        score = float(result.get("score", 0.0))
        truncated = bool(result.get("truncated", False))
        refused = bool(result.get("refused", False))
        trace = result.get("trace", {})
        if refused or (self.min_score is not None and score < self.min_score):
            return Answer(NO_ANSWER, score, True, ids, context, raw, truncated, trace)
        return Answer(raw, score, False, ids, context, raw, truncated, trace)

    def ask_many(self, questions: Sequence[str]) -> List[Answer]:
        return [self.ask(q) for q in questions]
