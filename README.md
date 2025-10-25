# Local RAG Application with LLaMA

Eine einfache Implementierung einer Retrieval-Augmented Generation (RAG) Anwendung unter Verwendung von lokalen LLMs mit Ollama.

## Übersicht

Diese Anwendung ermöglicht es, PDF-Dokumente zu laden und Fragen darüber zu stellen. Sie nutzt:
- LLaMA 3.1 (8B Parameter) als Sprachmodell
- Qwen Embeddings für die Vektorisierung
- SKLearn für den Vektorspeicher
- LangChain als Framework

## Installation

1. Installieren Sie die erforderlichen Pakete:

```bash
pip install -r [requirements.txt](http://_vscodecontentref_/1)
```

2. Stellen Sie sicher, dass Ollama installiert ist und die folgenden Modelle verfügbar sind:

* llama3.1:8b
* qwen3-embedding:8b

## Verwendung

1. Legen Sie Ihre PDF-Dokumente im Ordner `library/` ab.
2. Öffnen Sie `rag_demo.ipynb` und führen Sie die Zellen aus
3. Nutzen Sie die `ask()`-Methode um Fragen zu stellen:

```python
rag_app = RAGApplication()
antwort = rag_app.ask("Ihre Frage hier")
```

## Funktionsweise

* Die Anwendung lädt PDF-Dokumente und teilt sie in kleinere Textabschnitte
* Für jede Frage werden die relevantesten Textabschnitte mittels Vektorähnlichkeit gefunden
* Das LLM generiert eine Antwort basierend auf den gefundenen Textabschnitten

## Technische Details

* **Chunk-Größe:** 250 Token
* **Retrieval-Methode:** Maximum Marginal Relevance (MMR)
* **Top-k:** 5 relevanteste Dokumente