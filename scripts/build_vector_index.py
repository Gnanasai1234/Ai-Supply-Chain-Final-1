from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.rag.hybrid_retriever import build_chroma_index


if __name__ == "__main__":
    print(build_chroma_index())
