# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Candidate MatrixInstructions for MI Designer.

Two sources, both independent of the problem size:

* ``enumerate_candidate_mis``: every MI geko's tiling rules allow for an ARCH and
  data type (MFMA allowlist x MIBlockM x wave layout x wave tile, within the
  macro-tile bounds). Whether TensileLite can build a kernel for an MI is not
  checked here.
* Buildable-MI lists (``mi_lists/<arch>/<GEMM name>.txt``): the subset of those
  candidates for which TensileLite code generation builds at least one kernel
  within the generic (GA) search space's fork-parameter ranges. Produced by
  ``scripts/generate_buildable_mis.py``; MI Designer reads them instead of
  enumerating when one exists for the active GEMM type (except for the subtile
  search space, whose kernels the lists do not cover). A five-integer line
  ending in a colon is M, N, K, B, and MIBlockM. Each following four-integer
  line is appended to that prefix when the list is loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geko.config_generator.constants import (
    HARDWARE_MAP,
    LIST_OF_WAVEs_TO_INCLUDE,
    LSUTHRESHOLD,
    MAX_MT0,
    MAX_MT1,
    MIN_MT0,
    MIN_MT1,
    computeDataTypeSize,
    get_list_of_mt_max_size,
    library_logic_architecture,
    validMFMA,
)

MI = Tuple[int, int, int, int, int, int, int, int, int]

MI_LISTS_DIR = Path(__file__).parent / "mi_lists"


def allowable_mfmas(
    arch: str, data_type: str, mfma_override: Optional[Sequence[int]] = None
) -> List[List[int]]:
    """MFMA instructions ([M, N, K, B]) MI Designer may use: the ARCH allowlist, or the config's override."""
    if mfma_override is not None:
        return [list(mfma_override)]
    return HARDWARE_MAP[arch]["ONLY_INCLUDE_MIs"][data_type]


def smallest_mfma_dims(allowable: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """(smallest M, smallest N) over the allowed MFMA instructions (512 each when none is smaller)."""
    return min([512] + [m[0] for m in allowable]), min([512] + [m[1] for m in allowable])


def enumerate_candidate_mis(
    arch: str,
    data_type: str,
    search_space: Optional[str] = None,
    mfma_override: Optional[Sequence[int]] = None,
) -> List[MI]:
    """Every MatrixInstruction geko's tiling rules allow, as 9-tuples
    (M, N, K, B, MIBlockM, waveTileM, waveTileN, waveM, waveN).

    For each allowed MFMA and wave layout, wave tiles grow until the macro tile
    leaves [MIN_MT, MAX_MT]; a tile is dropped when its local-split-U LDS use
    exceeds LSUTHRESHOLD or its area exceeds the search space's MT cap.
    """
    allowable = allowable_mfmas(arch, data_type, mfma_override)
    mt_max_size = get_list_of_mt_max_size(search_space)

    # MIBlockM > 1 has not produced winners; raise bm_max to log2(MI[3]) to include it.
    bm_max = 0

    mis: List[MI] = []
    for mfma in reversed(validMFMA[data_type]):
        if mfma not in allowable:
            continue
        for bm in range(bm_max + 1):
            mi_block_m = 2**bm
            for wave in LIST_OF_WAVEs_TO_INCLUDE:
                wave_tile_m = 0
                while True:
                    wave_tile_m += 1
                    wave_tile_n = 0
                    mt0 = mfma[0] * mi_block_m * wave_tile_m * wave[0]
                    if mt0 < MIN_MT0:
                        continue
                    if mt0 > MAX_MT0:
                        break
                    while True:
                        wave_tile_n += 1
                        mt1 = int(mfma[1] / mi_block_m * mfma[3] * wave_tile_n * wave[1])
                        if mt1 < MIN_MT1:
                            continue
                        if mt1 > MAX_MT1:
                            break
                        lsu = max(1, 4 // wave[0] // wave[1])
                        if (
                            lsu > 1
                            and mt0 * mt1 * computeDataTypeSize[data_type] * lsu > LSUTHRESHOLD
                        ):
                            continue
                        if mt0 * mt1 > mt_max_size[data_type]:
                            continue
                        mis.append(
                            (
                                mfma[0],
                                mfma[1],
                                mfma[2],
                                mfma[3],
                                mi_block_m,
                                wave_tile_m,
                                wave_tile_n,
                                wave[0],
                                wave[1],
                            )
                        )
    return mis


def buildable_list_path(arch: str, gemm_name: str) -> Path:
    """Where the buildable-MI list for an ARCH and GEMM type (``GemmConfig.name``) lives.

    Lists are per compiler target (``gfx950_128cu`` shares ``gfx950``'s): code generation
    does not depend on the CU count.
    """
    return MI_LISTS_DIR / library_logic_architecture(arch) / f"{gemm_name}.txt"


def format_buildable_instructions(instructions: Sequence[Sequence[int]]) -> str:
    """Render matrix instructions with each shared prefix stored once.

    A five-integer line ending in a colon is M, N, K, B, and MIBlockM. Each
    following four-integer line is waveTileM, waveTileN, waveM, and waveN.
    The loader appends that tail to the most recent prefix.

    Args:
        instructions: Matrix instructions as nine integers each, in the order
            the designer should propose them.

    Returns:
        The list body. Rejected tiles are not included.

    Raises:
        ValueError: If an instruction does not contain nine integers.
    """
    lines: List[str] = []
    current: Optional[Tuple[int, ...]] = None
    for instruction in instructions:
        if len(instruction) != 9 or any(not isinstance(value, int) for value in instruction):
            raise ValueError(f"matrix instruction must be nine integers, got {instruction!r}")
        prefix = tuple(instruction[:5])
        if prefix != current:
            lines.append(" ".join(str(value) for value in prefix) + ":")
            current = prefix
        lines.append(" ".join(str(value) for value in instruction[5:]))
    return ("\n".join(lines) + "\n") if lines else ""


def _same_as_target(path: Path, line: str) -> Path:
    """Resolve a ``same_as`` directive to a sibling list file.

    Args:
        path: The file that contains the directive.
        line: The directive, including the ``same_as`` keyword.

    Returns:
        The sibling ``.txt`` path named by the directive.

    Raises:
        ValueError: If the name is missing or leaves this directory.
    """
    parts = line.split()
    if len(parts) != 2 or parts[0] != "same_as" or "/" in parts[1] or parts[1] in (".", ".."):
        raise ValueError(f"{path}: same_as needs one list name in this directory")
    return path.parent / f"{parts[1]}.txt"


def _read_instruction_rows(path: Path, stack: set[Path]) -> List[List[int]]:
    """Load nine-integer rows, following one ``same_as`` hop chain.

    Args:
        path: List file to read.
        stack: Files already entered, used to reject a cycle.

    Returns:
        Matrix instructions with each prefix appended to its tails.

    Raises:
        ValueError: If the file is missing, malformed, or part of a cycle.
    """
    resolved = path.resolve()
    if resolved in stack:
        raise ValueError(f"{path}: same_as cycle")
    if not path.is_file():
        raise ValueError(f"{path}: not a buildable-MI list")
    body = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            body.append(line)
    if len(body) == 1 and body[0].startswith("same_as"):
        stack.add(resolved)
        try:
            return _read_instruction_rows(_same_as_target(path, body[0]), stack)
        finally:
            stack.remove(resolved)
    prefix: Optional[List[int]] = None
    rows: List[List[int]] = []
    for line in body:
        if line.startswith("same_as"):
            raise ValueError(f"{path}: same_as must be the only directive")
        colon = line.endswith(":")
        text = line[:-1].strip() if colon else line
        try:
            numbers = [int(part) for part in text.split()]
        except ValueError as exc:
            raise ValueError(f"{path}: expected integers, got {line!r}") from exc
        if colon:
            if len(numbers) != 5:
                raise ValueError(f"{path}: a colon line needs 5 integers, got {line!r}")
            prefix = numbers
            continue
        if len(numbers) == 5:
            raise ValueError(f"{path}: a prefix line ends with a colon, got {line!r}")
        if len(numbers) == 4:
            if prefix is None:
                raise ValueError(f"{path}: a four-integer row needs a prefix first")
            rows.append(prefix + numbers)
            continue
        raise ValueError(f"{path}: expected a prefix or 4 integers, got {len(numbers)}")
    return rows


def load_buildable_list(path: Path) -> Dict[str, Any]:
    """Read a buildable-MI list and append each stored prefix to its tails.

    Args:
        path: A factored ``.txt`` list, or a file whose only directive is
            ``same_as <name>``.

    Returns:
        A dict whose ``MatrixInstructions`` value is a list of nine-integer lists.

    Raises:
        ValueError: If the file is missing, malformed, or ``same_as`` cycles.
    """
    return {"MatrixInstructions": _read_instruction_rows(path, set())}


def collapse_duplicate_lists(directory: Path) -> None:
    """Point identical lists in one directory at a single factored file.

    The alphabetically first file in each identical group keeps its body.
    The others become a ``same_as`` directive. A missing directory is left
    unchanged.

    Args:
        directory: An architecture directory of ``.txt`` lists.

    Returns:
        None

    Raises:
        ValueError: If a list in the directory cannot be read.
    """
    if not directory.is_dir():
        return
    groups: Dict[Tuple[Tuple[int, ...], ...], List[Path]] = {}
    for path in sorted(directory.glob("*.txt")):
        rows = tuple(tuple(row) for row in load_buildable_list(path)["MatrixInstructions"])
        groups.setdefault(rows, []).append(path)
    for members in groups.values():
        if len(members) < 2:
            continue
        keep = members[0]
        first_line = ""
        for raw in keep.read_text().splitlines():
            stripped = raw.split("#", 1)[0].strip()
            if stripped:
                first_line = stripped
                break
        if first_line.startswith("same_as"):
            rows = load_buildable_list(keep)["MatrixInstructions"]
            keep.write_text(format_buildable_instructions(rows))
        for other in members[1:]:
            other.write_text(f"same_as {keep.stem}\n")
