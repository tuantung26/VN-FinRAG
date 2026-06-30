# VN-FinRAG

VN-FinRAG is a chart-aware Retrieval-Augmented Generation (RAG) framework designed for question answering over Vietnamese financial reports.

Unlike conventional OCR-based RAG systems, VN-FinRAG preserves the semantic information contained in financial charts by converting charts into structured textual representations before indexing. This enables the retriever to effectively retrieve both textual and visual financial evidence while maintaining a standard text-based RAG pipeline.

## Features

- OCR-based document parsing for Vietnamese financial reports
- Automatic chart detection using YOLOv8
- Chart semantic extraction using DePlot
- Dense vector retrieval with Chroma
- Gemini-based answer generation
- Benchmark and evaluation pipeline for financial question answering

## Repository Structure

```
VN-FinRAG/
│
├── method0/          # Conventional OCR-based RAG
├── method1/          # Multimodal RAG with chart images
├── method2/          # Proposed chart-aware semantic RAG
│
├── benchmark/        # Evaluation benchmark
├── evaluation/       # Evaluation scripts
├── figures/          # Paper figures
├── paper/            # Manuscript
└── README.md
```

## Methods

### Method 0
Traditional OCR-based Retrieval-Augmented Generation.

```
PDF
 ↓
OCR
 ↓
Embedding
 ↓
Vector Database
 ↓
LLM
```

### Method 1
Charts are preserved as images and retrieved together with OCR text using multimodal reasoning.

```
PDF
 ├── OCR Text
 └── Chart Images
        ↓
 Multimodal Retrieval
        ↓
LLM
```

### Method 2 (Proposed)
Charts are transformed into structured semantic descriptions before indexing, allowing chart knowledge to participate directly in dense retrieval.

```
PDF
 ├── OCR Text
 └── Chart
      ↓
 Chart Understanding
      ↓
 Structured Text
      ↓
 Unified Index
      ↓
 Vector Retrieval
      ↓
 LLM
```

## Dataset

The benchmark consists of approximately **200 expert-designed financial question-answer pairs** collected from Vietnamese financial reports published by OCBS and other publicly available financial analysis sources.

The benchmark evaluates two complementary dimensions:

- **Fact Fidelity**
- **Insight Relevance**

## Citation

If you find this repository useful, please cite our paper.

```bibtex
@article{VNFinRAG2026,
  title={VN-FinRAG: Chart-aware Retrieval-Augmented Generation for Vietnamese Financial Reports},
  author={Anonymous},
  year={2026}
}
```

## License

This project is released under the MIT License.
