"""CLI thử nhanh 1 câu hỏi: python -m rag_model.retrieve "Câu hỏi pháp luật ví dụ?"

Tách riêng __main__.py vì `python -m rag_model.retrieve` thực thi __main__.py của
package, không phải __init__.py.
"""
import sys

from rag_model.retrieve import LegalRetriever

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Điều kiện hưởng bảo hiểm y tế là gì?"
    retriever = LegalRetriever()
    for r in retriever.retrieve(question):
        print(f"{r['document_id']}  ({r['score']:.4f})  {r['text'][:120]}...")
