---
name: pdf-tools
description: Use this skill whenever the user wants to extract text from, merge, or split PDF files.
allowed-tools: [read_file, write_file]
license: Apache-2.0
---

1. Determine whether the task is extraction, merging, or splitting.
2. Extraction: run `scripts/extract.py <input.pdf>` and read the output text.
3. Merging: collect the input paths in order, then run `scripts/merge.py <out.pdf> <in1.pdf> <in2.pdf> ...`.
4. Splitting: see `references/split_notes.md` for page-range syntax.
5. Report the resulting file path(s) to the user.
