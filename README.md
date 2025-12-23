# AI-recipe-generator
A course project for a Generative AI course at ITMO University
## Stack
- Visual Language Model (VLM): microsoft/Florence-2-base @ Hugging Face
- Language Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 @ Hugging Face
- Large Language Model (LLM): Gemini 2.5 Flash via Google Gemini API
- Vector Database: ChromaDB
- Language model framework: LangChain

## Explanation:
- We use a VLM model(can be changed with other models like salesforce/BLIP2) for object recognition (and getting captions). Florence 2 works great on a CPU and is considered to be a SOTA model in VLM area so that's why we chose it.
- Language model is used to make embeddings which later will be stored in a vector DB.
- LLM is used to process our detected objects and then it generates a detailed instructions and recipe for our dish.
