# vstore-faiss
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A vector store library using OpenAI embeddings.
Roll-my-own, no Langchain.

## FAISS

FAISS version, locally stored files.

## AstraDB

AstraDB requires free (at the moment) account and some setup required.

## Install

Rename `_example_vstore_config.py` to `vstore_config.py`
Enter all your AstraDB details

## To Do

- [-] refactor common code into vstore_common.py
- [ ] read_files re-factor/normalize across astra and faiss
- [ ] re-factor chunking and empty chunks across astra and faiss
- [ ] token counting is not used in astra
- [ ] re-fector process_and_add_files across astra and faiss
- [ ] re-factor/normalize similarity search across astra and faiss

- [ ] other DB options
- [ ] other embedding engine options

- [ ] error handling