# Schema-based Retrieval Engine

This project is a modular pipeline that transforms unstructured documents—like PDFs, emails, and CSVs—into structured data strictly following a given JSON schema. It combines schema parsing, document segmentation, hybrid retrieval (BM25 + FAISS), and LLM-based field extraction to produce accurate, review-ready outputs for complex, nested schemas across diverse domains.


## Setup

```bash
pip install -r requirements.txt
```

Make sure `Swig` is installed (required for `faiss`):

* macOS: `brew install swig`
* Ubuntu: `sudo apt-get install swig`

Set your OpenAI API key in .env:

```env
OPENAI_API_KEY=your-key-here
```

## Run

```bash
python main.py \
  --data_dir path/to/docs \
  --schema_file path/to/schema.json \
  --output_file output/final_output.json
```

## Notes

* Supports PDF, DOCX, CSV, JSON, HTML, BibTeX, and plaintext.
* Uses hybrid BM25 + FAISS retrieval and OpenAI LLM inference.
* Default embedding model: `all-MiniLM-L6-v2`.

