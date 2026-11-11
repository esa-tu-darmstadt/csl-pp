#!/usr/bin/env python3
"""Compare two LLVM IR (.ll) files produced by cslc, with symbol normalization.

This tool is designed for comparing LLVM IR output from vanilla cslc vs
csl-pp + cslc. It normalizes systematic naming differences (dot vs
double-underscore module separators, $$csl_base_address$$ prefixes) and
produces a structured diff report.

Usage:
    python3 tools/llvm-ir-diff.py <vanilla.ll> <cslpp.ll>
"""

import argparse
import difflib
import re
import sys
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Symbol name normalization
# ---------------------------------------------------------------------------

# Matches a quoted LLVM symbol name: @"..." or %"..."
_QUOTED_SYMBOL_RE = re.compile(r'([%@])"((?:[^"\\]|\\.)*)"')

# Matches an unquoted LLVM symbol name: @name or %name
# LLVM identifiers: [a-zA-Z$._][a-zA-Z$._0-9]*
_UNQUOTED_SYMBOL_RE = re.compile(r"([%@])([a-zA-Z$._][a-zA-Z$._0-9]*)")

# Matches the $$csl_base_address$$N$$ prefix in a symbol name.
_CSL_BASE_ADDR_RE = re.compile(r"\$\$csl_base_address\$\$\d+\$\$")


def _normalize_symbol_body(name: str) -> str:
    """Normalize the body of a single symbol name (without the @/% prefix).

    1. Strip $$csl_base_address$$N$$ prefix.
    2. Replace dots with double-underscores in CSL-style module paths.
       We only replace dots that appear between identifier characters
       (i.e., not after digits that look like float components, not in
       metadata references, etc.).
    """
    # Step 1: Strip csl_base_address prefix.
    name = _CSL_BASE_ADDR_RE.sub("", name)

    # Step 2: Replace dots used as module separators.
    # LLVM intrinsics like "llvm.sdr.blk.r" should be left alone since
    # both files will have identical LLVM intrinsic names.
    if name.startswith("llvm."):
        return name

    # Replace dots that function as CSL module separators.
    # A dot is a separator when preceded by [a-zA-Z_0-9] and followed by
    # [a-zA-Z_]. The follow-char must NOT be a digit alone, to avoid
    # matching inside float literals like "0.000000e+00".
    name = re.sub(r"(?<=[a-zA-Z_0-9])\.(?=[a-zA-Z_])", "__", name)

    return name


def _normalize_quoted_symbol(match: re.Match) -> str:
    """Normalize a quoted symbol @"name" or %"name"."""
    prefix = match.group(1)  # @ or %
    body = match.group(2)
    normalized = _normalize_symbol_body(body)
    # If the normalized name is a valid unquoted identifier, remove quotes.
    if re.fullmatch(r"[a-zA-Z$._][a-zA-Z$._0-9]*", normalized):
        return f"{prefix}{normalized}"
    return f'{prefix}"{normalized}"'


def _normalize_unquoted_symbol(match: re.Match) -> str:
    """Normalize an unquoted symbol @name or %name."""
    prefix = match.group(1)  # @ or %
    body = match.group(2)
    normalized = _normalize_symbol_body(body)
    if normalized == body:
        return match.group(0)
    # If normalized name needs quoting, add quotes.
    if re.fullmatch(r"[a-zA-Z$._][a-zA-Z$._0-9]*", normalized):
        return f"{prefix}{normalized}"
    return f'{prefix}"{normalized}"'


def normalize_line(line: str) -> str:
    """Normalize all symbol names in a single line of LLVM IR.

    Processes quoted symbols first (they may contain dots that are part of
    the quoted name), then unquoted symbols.
    """
    # First pass: normalize quoted symbols.
    line = _QUOTED_SYMBOL_RE.sub(_normalize_quoted_symbol, line)
    # Second pass: normalize unquoted symbols.
    line = _UNQUOTED_SYMBOL_RE.sub(_normalize_unquoted_symbol, line)
    return line


def normalize_file(content: str) -> str:
    """Normalize all symbol names in an entire .ll file."""
    lines = content.split("\n")
    normalized = []
    for line in lines:
        normalized.append(normalize_line(line))
    return "\n".join(normalized)


# ---------------------------------------------------------------------------
# Parsing globals and functions from LLVM IR
# ---------------------------------------------------------------------------

# Matches a global variable definition.
# Examples:
#   @name = internal global i16 0, align 4
#   @"quoted.name" = dso_local constant [1 x i16] zeroinitializer, align 4
_GLOBAL_DEF_RE = re.compile(
    r'^(@(?:"[^"]*"|[a-zA-Z$._][a-zA-Z$._0-9]*))\s*=' r"\s*(.*)", re.MULTILINE
)

# Matches a function definition and captures the function name.
# Examples:
#   define internal void @func_name(args) !dbg !7 {
#   define void @func_name(args) #0 !dbg !7 {
_FUNC_DEF_RE = re.compile(
    r"^(define\s+.*?"  # "define" + linkage/attrs + return type
    r'(@(?:"[^"]*"|[a-zA-Z$._][a-zA-Z$._0-9]*))'  # function name
    r"\(.*?\))"  # arguments
    r"(.*?)\{",  # attributes before opening brace
    re.MULTILINE,
)

# Matches a function declaration.
_FUNC_DECL_RE = re.compile(
    r'^declare\s+.*?(@(?:"[^"]*"|[a-zA-Z$._][a-zA-Z$._0-9]*))\s*\(', re.MULTILINE
)


def _extract_symbol_name(raw: str) -> str:
    """Extract the bare symbol name from @name or @"name"."""
    if raw.startswith('@"') and raw.endswith('"'):
        return raw[2:-1]
    if raw.startswith("@"):
        return raw[1:]
    return raw


def parse_globals(content: str) -> OrderedDict:
    """Parse all global variable definitions from normalized LLVM IR.

    Returns an OrderedDict mapping normalized name -> full definition line.
    """
    globals_dict = OrderedDict()
    for line in content.split("\n"):
        line = line.strip()
        m = _GLOBAL_DEF_RE.match(line)
        if m:
            raw_name = m.group(1)
            name = _extract_symbol_name(raw_name)
            globals_dict[name] = line
    return globals_dict


def parse_functions(content: str) -> OrderedDict:
    """Parse all function definitions from normalized LLVM IR.

    Returns an OrderedDict mapping normalized name -> list of body lines
    (including the 'define' line and closing '}'). Declarations are excluded.
    """
    funcs = OrderedDict()
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check for function definition.
        if line.startswith("define "):
            # Extract function name.
            m = _FUNC_DEF_RE.match(line)
            if m:
                raw_name = m.group(2)
                name = _extract_symbol_name(raw_name)
                # Collect all lines until the closing '}'.
                func_lines = [line]
                i += 1
                brace_depth = 1
                while i < len(lines) and brace_depth > 0:
                    func_lines.append(lines[i])
                    # Count braces (simple heuristic -- works for well-formed IR).
                    for ch in lines[i]:
                        if ch == "{":
                            brace_depth += 1
                        elif ch == "}":
                            brace_depth -= 1
                    i += 1
                funcs[name] = func_lines
                continue
        i += 1
    return funcs


def parse_declarations(content: str) -> OrderedDict:
    """Parse all function declarations from normalized LLVM IR.

    Returns an OrderedDict mapping normalized name -> full declaration line.
    """
    decls = OrderedDict()
    for line in content.split("\n"):
        line = line.strip()
        m = _FUNC_DECL_RE.match(line)
        if m:
            raw_name = m.group(1)
            name = _extract_symbol_name(raw_name)
            decls[name] = line
    return decls


# ---------------------------------------------------------------------------
# Stripping metadata/debug info for cleaner comparison
# ---------------------------------------------------------------------------

_DBG_SUFFIX_RE = re.compile(r",?\s*!dbg\s+!\d+")
_TBAA_SUFFIX_RE = re.compile(r",?\s*!tbaa\s+!\d+")


def strip_debug_metadata(line: str) -> str:
    """Remove debug metadata references from a line for comparison purposes."""
    line = _DBG_SUFFIX_RE.sub("", line)
    line = _TBAA_SUFFIX_RE.sub("", line)
    return line


def strip_func_attrs(line: str) -> str:
    """Remove function attribute group references (#N) from a define line."""
    if line.strip().startswith("define "):
        # Remove #N that appears after the closing paren of args,
        # but before !dbg or {.
        line = re.sub(r"\)\s*#\d+", ")", line)
    return line


def normalize_func_lines_for_diff(
    func_lines: list, do_strip_debug: bool = True
) -> list:
    """Prepare function lines for comparison by stripping volatile metadata."""
    result = []
    for line in func_lines:
        if do_strip_debug:
            line = strip_debug_metadata(line)
        line = strip_func_attrs(line)
        # Normalize whitespace.
        line = line.rstrip()
        if line:
            result.append(line)
    return result


# ---------------------------------------------------------------------------
# DSR global content-based matching
# ---------------------------------------------------------------------------

# DSR globals are named __dsr_lowmeminit_  __dsr_lowmeminit__0  __dsr_lowmeminit__1
# The sequential number may differ between the two files when allocations
# happen in different order.
_DSR_GLOBAL_RE = re.compile(r"^__dsr_lowmeminit_")


def _dsr_global_value(definition: str) -> str:
    """Extract the value part of a global definition (everything after ' = ')."""
    idx = definition.find(" = ")
    return definition[idx + 3 :] if idx >= 0 else definition


def match_dsr_globals_by_content(
    globals_a: OrderedDict, globals_b: OrderedDict, do_strip_debug: bool = True
) -> tuple:
    """Pool all DSR globals and match them by content instead of by name.

    DSR globals are allocated sequentially and the numbering may differ
    between the two files.  This function removes them from the regular
    name-based comparison and instead finds the best content-based pairing:
    identical-value pairs are matched first (greedy), then remaining entries
    are paired by sorted position (best effort) and shown with diffs.

    Returns:
        (dsr_equal, dsr_diff, dsr_only_a, dsr_only_b,
         filtered_globals_a, filtered_globals_b)
        - dsr_equal:          list of (name_a, name_b) – identical value
        - dsr_diff:           list of (name_a, name_b, def_a, def_b)
        - dsr_only_a:         list of (name, definition) unmatched in A
        - dsr_only_b:         list of (name, definition) unmatched in B
        - filtered_globals_a: globals_a with DSR entries removed
        - filtered_globals_b: globals_b with DSR entries removed
    """
    dsr_a = OrderedDict((k, v) for k, v in globals_a.items() if _DSR_GLOBAL_RE.match(k))
    dsr_b = OrderedDict((k, v) for k, v in globals_b.items() if _DSR_GLOBAL_RE.match(k))
    filtered_a = OrderedDict(
        (k, v) for k, v in globals_a.items() if not _DSR_GLOBAL_RE.match(k)
    )
    filtered_b = OrderedDict(
        (k, v) for k, v in globals_b.items() if not _DSR_GLOBAL_RE.match(k)
    )

    if not dsr_a and not dsr_b:
        return [], [], [], [], filtered_a, filtered_b

    def _val(defn: str) -> str:
        v = _dsr_global_value(defn)
        return strip_debug_metadata(v) if do_strip_debug else v

    a_names = sorted(dsr_a)
    b_names = sorted(dsr_b)
    a_vals = {n: _val(dsr_a[n]) for n in a_names}
    b_vals = {n: _val(dsr_b[n]) for n in b_names}

    dsr_equal: list = []
    matched_a: set = set()
    matched_b: set = set()

    # Pass 1: greedily pair entries with identical values.
    for a_name in a_names:
        for b_name in b_names:
            if b_name in matched_b:
                continue
            if a_vals[a_name] == b_vals[b_name]:
                dsr_equal.append((a_name, b_name))
                matched_a.add(a_name)
                matched_b.add(b_name)
                break

    remaining_a = [n for n in a_names if n not in matched_a]
    remaining_b = [n for n in b_names if n not in matched_b]

    # Pass 2: pair remaining by sorted position, record diffs.
    dsr_diff: list = []
    for a_name, b_name in zip(remaining_a, remaining_b):
        dsr_diff.append((a_name, b_name, dsr_a[a_name], dsr_b[b_name]))

    # Any extras where counts differ remain truly unmatched.
    dsr_only_a = [(n, dsr_a[n]) for n in remaining_a[len(remaining_b) :]]
    dsr_only_b = [(n, dsr_b[n]) for n in remaining_b[len(remaining_a) :]]

    return dsr_equal, dsr_diff, dsr_only_a, dsr_only_b, filtered_a, filtered_b


# ---------------------------------------------------------------------------
# Generic function content-based matching
# ---------------------------------------------------------------------------

# A-side naming: __generic<N><base>   (e.g. __generic10foo__bar)
_GENERIC_A_RE = re.compile(r"^__generic\d+(.+)$")
# B-side naming: <base>_inst<N>       (e.g. foo__bar_inst9561)
_GENERIC_B_RE = re.compile(r"^(.+)_inst\d+$")


def _canonical_a(name: str):
    """Return the canonical base name if this is an A-side generic, else None."""
    m = _GENERIC_A_RE.match(name)
    return m.group(1) if m else None


def _canonical_b(name: str) -> str:
    """Return the canonical base name for a B-side name (strip _instN if present)."""
    m = _GENERIC_B_RE.match(name)
    return m.group(1) if m else name


def _canonicalize_body(func_lines: list, name_to_canonical: dict) -> list:
    """Replace all known generic name references with stable canonical placeholders.

    This allows comparing bodies of functions that differ only in their
    generated generic/instantiation suffixes or prefixes.
    """
    if not name_to_canonical:
        return func_lines

    # Sort by descending length to avoid partial-name substitutions.
    sorted_names = sorted(name_to_canonical, key=len, reverse=True)
    # Symbol refs are @name (unquoted) – match name as a whole token.
    pattern = re.compile(
        r"@(" + "|".join(re.escape(n) for n in sorted_names) + r")"
        r"(?=[^a-zA-Z$._0-9]|$)"
    )

    def _repl(m: re.Match) -> str:
        return f"@<generic:{name_to_canonical[m.group(1)]}>"

    return [pattern.sub(_repl, line) for line in func_lines]


def match_generics_by_content(
    only_a: list,
    only_b: list,
    funcs_a: OrderedDict,
    funcs_b: OrderedDict,
    do_strip_debug: bool = True,
) -> tuple:
    """Content-based matching for unmatched generic functions.

    A-side functions with ``__generic<N>`` prefix are paired with B-side
    functions whose canonical name (after stripping ``_inst<N>`` suffix) is
    the same.  Bodies are compared after normalising all known generic name
    references to stable placeholders, so numbering differences don't
    produce false negatives.

    Returns:
        (generic_equal, generic_diff, still_only_a, still_only_b)
        - generic_equal:  list of (name_a, name_b) – same body after normalisation
        - generic_diff:   list of (name_a, name_b, diff_lines) – same canonical,
                          different body
        - still_only_a:   names from only_a not matched by this step
        - still_only_b:   names from only_b not matched by this step
    """
    # Partition A unmatched: only __genericN functions are candidates.
    groups_a: dict[str, list] = {}  # canonical -> [name, ...]
    non_generic_a: list = []
    for name in only_a:
        c = _canonical_a(name)
        if c is not None:
            groups_a.setdefault(c, []).append(name)
        else:
            non_generic_a.append(name)

    # All B unmatched are candidates (canonical = strip _instN if present).
    groups_b: dict[str, list] = {}
    for name in only_b:
        c = _canonical_b(name)
        groups_b.setdefault(c, []).append(name)

    # Build a global name-to-canonical map used to normalise function bodies.
    # Include both sides so cross-references between generics are also normalised.
    name_to_canonical: dict[str, str] = {}
    for c, names in groups_a.items():
        for n in names:
            name_to_canonical[n] = c
    for c, names in groups_b.items():
        for n in names:
            name_to_canonical[n] = c

    generic_equal: list = []
    generic_diff: list = []
    still_only_a: list = list(non_generic_a)
    still_only_b: list = []

    all_canonicals = sorted(set(groups_a) | set(groups_b))
    for canonical in all_canonicals:
        a_names = sorted(groups_a.get(canonical, []))
        b_names = sorted(groups_b.get(canonical, []))

        if not a_names:
            # No A-side generic with this canonical – B names stay unmatched.
            still_only_b.extend(b_names)
            continue
        if not b_names:
            still_only_a.extend(a_names)
            continue

        # Pre-compute normalised + canonicalised bodies for all candidates.
        def _body(name: str, funcs: OrderedDict) -> list:
            lines = normalize_func_lines_for_diff(funcs[name], do_strip_debug)
            return _canonicalize_body(lines, name_to_canonical)

        a_bodies = {n: _body(n, funcs_a) for n in a_names}
        b_bodies = {n: _body(n, funcs_b) for n in b_names}

        matched_a: set = set()
        matched_b: set = set()

        # Pass 1: greedily pair identical bodies.
        for a_name in a_names:
            for b_name in b_names:
                if b_name in matched_b:
                    continue
                if a_bodies[a_name] == b_bodies[b_name]:
                    generic_equal.append((a_name, b_name))
                    matched_a.add(a_name)
                    matched_b.add(b_name)
                    break

        remaining_a = [n for n in a_names if n not in matched_a]
        remaining_b = [n for n in b_names if n not in matched_b]

        # Pass 2: pair remaining by position (best-effort), record diffs.
        for a_name, b_name in zip(remaining_a, remaining_b):
            diff = list(
                difflib.unified_diff(
                    a_bodies[a_name],
                    b_bodies[b_name],
                    fromfile=f"A::{a_name}",
                    tofile=f"B::{b_name}",
                    lineterm="",
                )
            )
            generic_diff.append((a_name, b_name, diff))

        # Any extras where counts differ remain truly unmatched.
        still_only_a.extend(remaining_a[len(remaining_b) :])
        still_only_b.extend(remaining_b[len(remaining_a) :])

    return generic_equal, generic_diff, still_only_a, still_only_b


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_globals(
    globals_a: OrderedDict,
    globals_b: OrderedDict,
    label_a: str,
    label_b: str,
    do_strip_debug: bool = True,
) -> tuple:
    """Compare two sets of globals.

    Returns (only_a, only_b, differing, matched_names) where:
      - only_a: list of (name, definition) only in A
      - only_b: list of (name, definition) only in B
      - differing: list of (name, def_a, def_b) that differ
      - matched_names: list of names that match exactly
    """
    names_a = set(globals_a.keys())
    names_b = set(globals_b.keys())

    only_a = [(n, globals_a[n]) for n in sorted(names_a - names_b)]
    only_b = [(n, globals_b[n]) for n in sorted(names_b - names_a)]

    common = sorted(names_a & names_b)
    differing = []
    matched_names = []

    for name in common:
        def_a = globals_a[name]
        def_b = globals_b[name]
        if do_strip_debug:
            def_a = strip_debug_metadata(def_a)
            def_b = strip_debug_metadata(def_b)
        if def_a != def_b:
            differing.append((name, globals_a[name], globals_b[name]))
        else:
            matched_names.append(name)

    return only_a, only_b, differing, matched_names


def compare_functions(
    funcs_a: OrderedDict,
    funcs_b: OrderedDict,
    label_a: str,
    label_b: str,
    do_strip_debug: bool = True,
) -> tuple:
    """Compare two sets of functions.

    Returns (only_a, only_b, differing, matched_names) where:
      - only_a: list of names only in A
      - only_b: list of names only in B
      - differing: list of (name, unified_diff_lines)
      - matched_names: list of names that match
    """
    names_a = set(funcs_a.keys())
    names_b = set(funcs_b.keys())

    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    common = sorted(names_a & names_b)
    differing = []
    matched_names = []

    for name in common:
        lines_a = normalize_func_lines_for_diff(funcs_a[name], do_strip_debug)
        lines_b = normalize_func_lines_for_diff(funcs_b[name], do_strip_debug)

        if lines_a == lines_b:
            matched_names.append(name)
        else:
            diff = list(
                difflib.unified_diff(
                    lines_a,
                    lines_b,
                    fromfile=f"{label_a}::{name}",
                    tofile=f"{label_b}::{name}",
                    lineterm="",
                )
            )
            differing.append((name, diff))

    return only_a, only_b, differing, matched_names


def compare_declarations(decls_a: OrderedDict, decls_b: OrderedDict) -> tuple:
    """Compare function declarations.

    Returns (only_a, only_b, matched_names).
    """
    names_a = set(decls_a.keys())
    names_b = set(decls_b.keys())

    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    matched_names = sorted(names_a & names_b)

    return only_a, only_b, matched_names


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_separator(char: str = "=", width: int = 72) -> None:
    print(char * width)


def print_header(title: str) -> None:
    print()
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def report(
    globals_result,
    funcs_result,
    decls_result,
    generics_result,
    dsr_result,
    label_a: str,
    label_b: str,
    show_matching: bool = False,
) -> bool:
    """Print the full comparison report.

    Returns True if there are any differences.
    """
    g_only_a, g_only_b, g_diff, g_matched = globals_result
    f_only_a, f_only_b, f_diff, f_matched = funcs_result
    d_only_a, d_only_b, d_matched = decls_result
    gen_equal, gen_diff, gen_only_a, gen_only_b = generics_result
    dsr_equal, dsr_diff, dsr_only_a, dsr_only_b = dsr_result

    has_diff = bool(
        g_only_a
        or g_only_b
        or g_diff
        or f_only_a
        or f_only_b
        or f_diff
        or gen_only_a
        or gen_only_b
        or gen_diff
        or dsr_diff
        or dsr_only_a
        or dsr_only_b
        or d_only_a
        or d_only_b
    )

    # --- Summary ---
    print_header("SUMMARY")

    total_globals_a = len(g_matched) + len(g_diff) + len(g_only_a)
    total_globals_b = len(g_matched) + len(g_diff) + len(g_only_b)
    total_dsr_a = len(dsr_equal) + len(dsr_diff) + len(dsr_only_a)
    total_dsr_b = len(dsr_equal) + len(dsr_diff) + len(dsr_only_b)

    # total functions: name-matched + generics-matched + remaining only-in-X
    total_funcs_a = (
        len(f_matched)
        + len(f_diff)
        + len(f_only_a)
        + len(gen_equal)
        + len(gen_diff)
        + len(gen_only_a)
    )
    total_funcs_b = (
        len(f_matched)
        + len(f_diff)
        + len(f_only_b)
        + len(gen_equal)
        + len(gen_diff)
        + len(gen_only_b)
    )

    total_decls_a = len(d_matched) + len(d_only_a)
    total_decls_b = len(d_matched) + len(d_only_b)

    print(f"File A: {label_a}")
    print(f"File B: {label_b}")
    print()

    print(f"Globals:      {total_globals_a} in A, {total_globals_b} in B")
    print(f"  Matched:    {len(g_matched)}")
    print(f"  Differing:  {len(g_diff)}")
    print(f"  Only in A:  {len(g_only_a)}")
    print(f"  Only in B:  {len(g_only_b)}")
    print()

    print(f"DSR Globals:  {total_dsr_a} in A, {total_dsr_b} in B")
    print(f"  Content-matched (identical):  {len(dsr_equal)}")
    print(f"  Content-matched (differing):  {len(dsr_diff)}")
    print(f"  Only in A (unmatched):        {len(dsr_only_a)}")
    print(f"  Only in B (unmatched):        {len(dsr_only_b)}")
    if dsr_equal or dsr_diff:
        print("  DSR global pairings (A -> B):")
        for a_name, b_name in dsr_equal:
            print(f"    [=] @{a_name}  ->  @{b_name}")
        for a_name, b_name, _, __ in dsr_diff:
            print(f"    [!] @{a_name}  ->  @{b_name}")
    print()

    print(f"Functions:    {total_funcs_a} in A, {total_funcs_b} in B")
    print(f"  Name-matched (identical):          {len(f_matched)}")
    print(f"  Name-matched (differing):          {len(f_diff)}")
    print(f"  Generic-matched (identical):       {len(gen_equal)}")
    print(f"  Generic-matched (differing body):  {len(gen_diff)}")
    print(f"  Only in A (unmatched):             {len(f_only_a) + len(gen_only_a)}")
    print(f"  Only in B (unmatched):             {len(f_only_b) + len(gen_only_b)}")
    print()

    if gen_equal or gen_diff:
        print("  Generic function pairings (A -> B):")
        for a_name, b_name in gen_equal:
            print(f"    [=] @{a_name}")
            print(f"        @{b_name}")
        for a_name, b_name, _ in gen_diff:
            print(f"    [!] @{a_name}")
            print(f"        @{b_name}")
        print()

    print(f"Declarations: {total_decls_a} in A, {total_decls_b} in B")
    print(f"  Matched:    {len(d_matched)}")
    print(f"  Only in A:  {len(d_only_a)}")
    print(f"  Only in B:  {len(d_only_b)}")

    if not has_diff:
        print()
        print("No differences found (after normalization).")
        if show_matching:
            _print_matching_sections(
                g_matched, f_matched, d_matched, gen_equal, dsr_equal
            )
        return False

    # --- Matching (if requested) ---
    if show_matching:
        _print_matching_sections(g_matched, f_matched, d_matched, gen_equal, dsr_equal)

    # --- Globals only in A ---
    if g_only_a:
        print_header(f"GLOBALS ONLY IN A ({label_a})")
        for name, defn in g_only_a:
            print(f"  {defn}")

    # --- Globals only in B ---
    if g_only_b:
        print_header(f"GLOBALS ONLY IN B ({label_b})")
        for name, defn in g_only_b:
            print(f"  {defn}")

    # --- Differing globals ---
    if g_diff:
        print_header("DIFFERING GLOBALS")
        for name, def_a, def_b in g_diff:
            print(f"  @{name}")
            print(f"    A: {def_a}")
            print(f"    B: {def_b}")
            print()

    # --- DSR globals only in A ---
    if dsr_only_a:
        print_header(f"DSR GLOBALS ONLY IN A ({label_a})")
        for name, defn in dsr_only_a:
            print(f"  {defn}")

    # --- DSR globals only in B ---
    if dsr_only_b:
        print_header(f"DSR GLOBALS ONLY IN B ({label_b})")
        for name, defn in dsr_only_b:
            print(f"  {defn}")

    # --- Differing DSR globals (content-matched but different values) ---
    if dsr_diff:
        print_header("DIFFERING DSR GLOBALS (CONTENT-MATCHED)")
        for a_name, b_name, def_a, def_b in dsr_diff:
            print(f"  A: @{a_name}")
            print(f"  B: @{b_name}")
            print(f"    A: {def_a}")
            print(f"    B: {def_b}")
            print()

    # --- Declarations only in A ---
    if d_only_a:
        print_header(f"DECLARATIONS ONLY IN A ({label_a})")
        for name in d_only_a:
            print(f"  @{name}")

    # --- Declarations only in B ---
    if d_only_b:
        print_header(f"DECLARATIONS ONLY IN B ({label_b})")
        for name in d_only_b:
            print(f"  @{name}")

    # --- Functions only in A (name-unmatched and generic-unmatched) ---
    all_only_a = f_only_a + gen_only_a
    if all_only_a:
        print_header(f"FUNCTIONS ONLY IN A ({label_a})")
        for name in sorted(all_only_a):
            print(f"  @{name}")

    # --- Functions only in B ---
    all_only_b = f_only_b + gen_only_b
    if all_only_b:
        print_header(f"FUNCTIONS ONLY IN B ({label_b})")
        for name in sorted(all_only_b):
            print(f"  @{name}")

    # --- Differing functions (name-matched) ---
    if f_diff:
        print_header("DIFFERING FUNCTIONS (NAME-MATCHED)")
        for name, diff_lines in f_diff:
            print(f"--- @{name} ---")
            for dl in diff_lines:
                print(dl)
            print()

    # --- Generic-matched functions with differing bodies ---
    if gen_diff:
        print_header("DIFFERING GENERIC FUNCTIONS (CONTENT-MATCHED)")
        for a_name, b_name, diff_lines in gen_diff:
            print(f"--- A: @{a_name}")
            print(f"    B: @{b_name} ---")
            for dl in diff_lines:
                print(dl)
            print()

    return True


def _print_matching_sections(g_matched, f_matched, d_matched, gen_equal, dsr_equal):
    """Print the matching globals, functions, and declarations."""
    if g_matched:
        print_header("MATCHING GLOBALS")
        for name in g_matched:
            print(f"  @{name}")

    if dsr_equal:
        print_header("MATCHING DSR GLOBALS (BY CONTENT)")
        for a_name, b_name in dsr_equal:
            print(f"  A: @{a_name}  ->  B: @{b_name}")

    if f_matched:
        print_header("MATCHING FUNCTIONS (BY NAME)")
        for name in f_matched:
            print(f"  @{name}")

    if gen_equal:
        print_header("MATCHING GENERIC FUNCTIONS (BY CONTENT)")
        for a_name, b_name in gen_equal:
            print(f"  A: @{a_name}")
            print(f"  B: @{b_name}")
            print()

    if d_matched:
        print_header("MATCHING DECLARATIONS")
        for name in d_matched:
            print(f"  @{name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two LLVM IR (.ll) files with CSL symbol normalization."
    )
    parser.add_argument("file_a", help="First .ll file (e.g., vanilla cslc output)")
    parser.add_argument("file_b", help="Second .ll file (e.g., csl-pp + cslc output)")
    parser.add_argument(
        "--no-strip-debug",
        action="store_true",
        help="Do not strip debug metadata (!dbg, !tbaa) before comparing",
    )
    parser.add_argument(
        "--show-matching",
        action="store_true",
        help="Also list globals/functions that match",
    )
    args = parser.parse_args()

    do_strip_debug = not args.no_strip_debug

    # Read and normalize both files.
    with open(args.file_a, "r") as f:
        content_a = normalize_file(f.read())
    with open(args.file_b, "r") as f:
        content_b = normalize_file(f.read())

    # Parse structure.
    globals_a = parse_globals(content_a)
    globals_b = parse_globals(content_b)
    funcs_a = parse_functions(content_a)
    funcs_b = parse_functions(content_b)
    decls_a = parse_declarations(content_a)
    decls_b = parse_declarations(content_b)

    label_a = args.file_a
    label_b = args.file_b

    # Content-based matching for DSR globals that may have different numbering.
    # Must happen before the regular name-based comparison so DSR entries are
    # removed from globals_a/globals_b.
    (dsr_equal, dsr_diff, dsr_only_a, dsr_only_b, globals_a, globals_b) = (
        match_dsr_globals_by_content(globals_a, globals_b, do_strip_debug)
    )
    dsr_result = (dsr_equal, dsr_diff, dsr_only_a, dsr_only_b)

    # Name-based comparison for remaining (non-DSR) globals.
    globals_result = compare_globals(
        globals_a, globals_b, label_a, label_b, do_strip_debug
    )
    funcs_result = compare_functions(funcs_a, funcs_b, label_a, label_b, do_strip_debug)
    decls_result = compare_declarations(decls_a, decls_b)

    # Content-based matching for generics that differ only in naming convention.
    f_only_a, f_only_b = funcs_result[0], funcs_result[1]
    generics_result = match_generics_by_content(
        f_only_a, f_only_b, funcs_a, funcs_b, do_strip_debug
    )
    # Replace only_a/only_b in funcs_result with the post-generic-matching remainders.
    _, _, f_diff, f_matched = funcs_result
    gen_equal, gen_diff, gen_only_a, gen_only_b = generics_result
    funcs_result = ([], [], f_diff, f_matched)
    generics_result = (gen_equal, gen_diff, gen_only_a, gen_only_b)

    # Report.
    has_diff = report(
        globals_result,
        funcs_result,
        decls_result,
        generics_result,
        dsr_result,
        label_a,
        label_b,
        args.show_matching,
    )

    return 1 if has_diff else 0


if __name__ == "__main__":
    sys.exit(main())
