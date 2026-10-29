# Buildable matrix-instruction lists

One file per GEMM type. The loader returns nine integers per tile, in order: M, N, K, B, MIBlockM, then waveTileM, waveTileN, waveM, waveN.

A line is one of these:

- `#` starts a comment. The loader ignores the rest of that line, including a comment after numbers.
- `M N K B MIBlockM:` starts a group. Those five integers are stored once. The colon is required.
- `waveTileM waveTileN waveM waveN` is one tile. The loader appends the current group.
- `same_as <name>` is the whole file. `<name>` is a sibling list in this directory, without `.txt`. The loader reads that file. A chain is allowed. A cycle is an error. This line cannot share the file with any other directive.

Rejected tiles are not stored here.
