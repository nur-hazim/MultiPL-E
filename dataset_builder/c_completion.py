"""Conservative assembly of C completions; never repair algorithmic code."""
import re


def normalize_c_source(prefix, source):
    """Restore omitted frozen declarations, preserving the raw response elsewhere.

    Complete typedef-bearing programs are not given duplicate declarations.
    Nested helpers, extensions, prose and truncated bodies are left for the C11
    compiler to reject. Tests and function bodies are never rewritten.
    """
    changes = []
    fenced = re.fullmatch(r'\s*```(?:c)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*', source)
    if fenced:
        source = fenced.group(1)
        changes.append('removed_complete_markdown_fence')
    signature = prefix.rstrip().splitlines()[-1]
    match = re.search(r'\b(he_\w+)\s*\(', signature)
    if not match or not signature.rstrip().endswith('{'):
        raise ValueError('Expected a frozen HumanEval C signature at the end of the prompt')
    target = match.group(1)
    # Only restore declarations for a function-level answer. Do not guess how
    # to splice an arbitrary body or a partial set of typedefs into a program.
    if (not re.search(r'(?m)^\s*typedef\b', source)
            and re.search(r'\b' + re.escape(target) + r'\s*\(', source)):
        source = prefix[:prefix.rfind(signature)] + source
        changes.append('restored_frozen_preamble')
    if (re.search(r'\b(?:printf|fprintf|sprintf|snprintf|scanf|sscanf|puts|FILE)\b', source)
            and not re.search(r'#\s*include\s*[<"]stdio\.h[>"]', source)):
        source = '#include <stdio.h>\n' + source
        changes.append('added_stdio_header')
    return source, changes
