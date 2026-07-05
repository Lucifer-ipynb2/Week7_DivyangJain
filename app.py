import argparse
import os

import config
from rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Document Question Answering (RAG) CLI")
    parser.add_argument("--docs", default=config.DOCS_FOLDER,
                         help="Folder containing PDFs/TXT files to index")
    parser.add_argument("--build", action="store_true",
                         help="(Re)build the vector index before starting")
    args = parser.parse_args()

    pipeline = RAGPipeline()

    index_exists = os.path.exists(os.path.join(config.INDEX_DIR, "index.faiss"))

    if args.build or not index_exists:
        pipeline.build_index(args.docs)
    else:
        pipeline.load_index()

    print("=" * 60)
    print("RAG Question Answering System — type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            result = pipeline.answer(question)
        except Exception as e:
            print(f"[error] {e}")
            continue

        print(f"\nAnswer:\n{result['answer']}")
        if result["sources"]:
            print("\nSources used:")
            for s in result["sources"]:
                print(f"  - {s['source']} (chunk {s['chunk_id']}, score {s['score']})")


if __name__ == "__main__":
    main()
