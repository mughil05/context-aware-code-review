"""Deterministic Python structural/lexical retrieval from a frozen HEAD archive.

No execution of repository code, embeddings, network, or vector database.
"""
import ast
import hashlib
import io
import json
import keyword
from pathlib import Path, PurePosixPath
import re
import tarfile

from reviewer.config import RetrievalConfig
from reviewer.diff import parse_diff
from reviewer.schemas import ContextChunk, RepositoryContext, repository_path

VERSION = 'structural_lexical_v1'
EXCLUDED = {'vendor', 'vendored', 'third_party', 'node_modules', 'site-packages', 'dist',
            'build', 'target', '__pycache__', 'venv', 'env', 'golden', 'goldens', 'reference',
            'results', 'reviewbench', 'review-bench', 'credentials', 'secrets', 'generated',
            'scoring', 'scores', 'judgments', 'benchmark_labels', 'benchmark', 'benchmarks',
            'benchmark_data', 'predictions', 'model_responses', 'responses', 'evaluation',
            'evaluations', 'artifacts', 'outputs', 'output', 'logs', 'cache'}
SECRET_NAME = re.compile(r'(^|[_-])(secrets?|credentials?|settings|config|tokens?|passwords?)([_-]|\.|$)', re.I)
SECRET_CONTENT = re.compile(r'-----BEGIN .*PRIVATE KEY-----|[\"\']?\b\w*(?:api_key|password|secret|access_token)[\"\']?\s*[:=]\s*[\"\'][^\"\']+[\"\']|\b(?:sk-|ghp_)[A-Za-z0-9_-]{16,}', re.I)
STOP = set(keyword.kwlist) | {'self', 'cls', 'None', 'True', 'False', 'str', 'int', 'bool', 'list', 'dict', 'return', 'py', 'src'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tokens(text):
    return set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', text)) - STOP


def eligible(path):
    parts = PurePosixPath(path).parts
    return (path.endswith('.py') and not any(p.startswith('.') or p.lower() in EXCLUDED for p in parts)
            and PurePosixPath(path).stem.lower() not in EXCLUDED
            and not SECRET_NAME.search(parts[-1]) and not parts[-1].endswith(('_pb2.py', '_pb2_grpc.py')))


def snapshot_sources(snapshot, expected_head):
    """Read verified archive bytes, never a mutable checkout or benchmark directory."""
    if snapshot is None or snapshot.commit != expected_head:
        raise ValueError('retrieval snapshot is not the frozen PR head')
    path = Path(snapshot.archive_path)
    if any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('unsafe retrieval snapshot archive')
    raw = path.read_bytes()
    if digest(raw) != snapshot.archive_sha256:
        raise ValueError('retrieval snapshot archive hash mismatch')
    sources, seen, root, total = {}, set(), None, 0
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r|gz') as archive:
            for index, member in enumerate(archive):
                name = member.name.rstrip('/') if member.isdir() else member.name
                repository_path(name)
                parts = name.split('/')
                root = parts[0] if root is None else root
                total += member.size
                if (index >= 100000 or total > 256 * 1024 * 1024 or member.size < 0
                        or name in seen or parts[0] != root or member.issparse()
                        or not (member.isdir() or member.isfile())):
                    raise ValueError('unsafe retrieval snapshot member')
                seen.add(name)
                if len(parts) == 1:
                    if not member.isdir():
                        raise ValueError('snapshot wrapper must be a directory')
                    continue
                relative = '/'.join(parts[1:])
                if not member.isfile() or not eligible(relative) or member.size > 256 * 1024:
                    continue
                content = archive.extractfile(member).read()
                if len(content) != member.size or b'\0' in content:
                    continue
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                if (SECRET_CONTENT.search(text) or re.search(r'auto[- ]generated|do not edit', text[:2000], re.I)):
                    continue
                sources[relative] = text
    except (tarfile.TarError, EOFError, OSError) as error:
        raise ValueError('unreadable retrieval snapshot') from error
    return dict(sorted(sources.items()))


def analyze(text, target_lines, cache_dir=None):
    """Cache AST boundaries/import metadata by source bytes + algorithm + chunk size.

    Context text is always sliced from the verified snapshot, never from cache.
    Invalid cache data is ignored; it cannot supply source text to the model.
    """
    lines = text.splitlines()
    key = digest((VERSION + ':' + str(target_lines) + ':' + text).encode())
    path = None
    if cache_dir is not None:
        root = Path(cache_dir)
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError('unsafe retrieval cache path')
        root.mkdir(parents=True, exist_ok=True)
        path = root / (key + '.json')
        if path.is_symlink():
            raise ValueError('unsafe retrieval cache entry')
        if path.is_file():
            try:
                saved = json.loads(path.read_bytes())
                data = saved['data']
                if (saved['key'] == key and saved['sha256'] == digest(json.dumps(data, sort_keys=True).encode())
                        and all(type(a) is int and type(b) is int and 1 <= a <= b <= len(lines)
                                for a, b in data['ranges'])
                        and all(isinstance(m, str) and type(l) is int and l >= 0 for m, l in data['imports'])
                        and all(isinstance(n, str) for n in data['import_names'])):
                    return data
            except (ValueError, KeyError, TypeError, OSError):
                pass
    imports, names, units = [], [], []
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name, 0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append([node.module or '', node.level])
                names.extend(alias.name for alias in node.names if alias.name != '*')
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                units.append((start, node.end_lineno))
    except (SyntaxError, ValueError, RecursionError):
        pass  # Incomplete/new syntax still receives bounded lexical chunks.
    spans, cursor = [], 1
    for start, end in units:
        if cursor < start:
            spans.append((cursor, start - 1))
        spans.append((start, end))
        cursor = end + 1
    if cursor <= len(lines):
        spans.append((cursor, len(lines)))
    ranges = [[n, min(end, n + target_lines - 1)] for start, end in spans
              for n in range(start, end + 1, target_lines)]
    data = {'ranges': ranges, 'imports': sorted(imports), 'import_names': sorted(set(names))}
    if path is not None:
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump({'key': key, 'data': data,
                           'sha256': digest(json.dumps(data, sort_keys=True).encode())}, stream, sort_keys=True)
        except FileExistsError:
            pass
    return data


def module_name(path):
    parts = list(PurePosixPath(path).with_suffix('').parts)
    if parts and parts[0] == 'src':
        parts.pop(0)
    if parts and parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def context_cost(items):
    # Include the second JSON-encoding layer used by the Responses payload.
    return len(json.dumps(json.dumps({'repository_context': items}, ensure_ascii=False),
                          ensure_ascii=False).encode('utf-8')) if items else 0


def retrieve(review_input, snapshot, config: RetrievalConfig, *, cache_dir=None, available_byte_budget=None):
    config = RetrievalConfig.model_validate(config.model_dump())
    sources = snapshot_sources(snapshot, review_input.pr.head)
    parsed = parse_diff(review_input.diff)
    changed = {f.new_path for f in parsed if f.new_path}
    represented = {f.new_path: {l.new_line for h in f.hunks for l in h.lines if l.new_line is not None}
                   for f in parsed if f.new_path}
    query_text = '\n'.join(l.content for f in parsed for h in f.hunks for l in h.lines)
    query = tokens(query_text) | tokens(' '.join(sorted(changed)))
    analyses = {p: analyze(text, config.chunk_target_lines, cache_dir) for p, text in sources.items()}
    imports = set()
    for path in sorted(changed & sources.keys()):
        data = analyses[path]
        query.update(data['import_names'])
        package = module_name(path).split('.')[:-1]
        if PurePosixPath(path).name == '__init__.py':
            package = module_name(path).split('.')
        for module, level in data['imports']:
            prefix = package[:len(package) - level + 1] if level else []
            imports.add('.'.join(prefix + ([module] if module else [])))
    changed_parents = {str(PurePosixPath(p).parent) for p in changed}
    candidates = []
    for path, text in sources.items():
        lines = text.splitlines()
        for first, last in analyses[path]['ranges']:
            # Remove every represented HEAD line, including unchanged hunk context.
            segments, start = [], None
            for n in range(first, last + 2):
                allowed = n <= last and n not in represented.get(path, set())
                if allowed and start is None:
                    start = n
                elif not allowed and start is not None:
                    segments.append((start, n - 1)); start = None
            for start, end in segments:
                content = '\n'.join(lines[start - 1:end])
                if not content.strip():
                    continue
                score, reasons = 0, []
                module = module_name(path)
                if module and any(module == m or module.startswith(m + '.') for m in imports if m):
                    score += 100; reasons.append('imported_module')
                definitions = set(re.findall(r'^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)', content, re.M))
                hits = sorted(definitions & query)
                if hits:
                    score += 60; reasons.append('referenced_definition:' + ','.join(hits))
                if path in changed:
                    score += 30; reasons.append('same_file_outside_diff')
                elif str(PurePosixPath(path).parent) in changed_parents:
                    score += 20; reasons.append('nearby_module')
                overlap = len(tokens(content) & query)
                if overlap:
                    score += min(40, overlap * 2); reasons.append(f'token_overlap:{overlap}')
                name_overlap = len(tokens(path) & query)
                if name_overlap:
                    score += min(20, name_overlap * 5); reasons.append(f'path_overlap:{name_overlap}')
                candidates.append(ContextChunk(file=path, start_line=start, end_line=end, content=content,
                    content_sha256=digest(content.encode()), score=score, reasons=reasons,
                    selected=False, selection_reason='no_signal'))
    candidates.sort(key=lambda c: (-c.score, c.file, c.start_line, c.end_line, c.content_sha256))
    budget = min(config.context_byte_budget, max(0, available_byte_budget)) if available_byte_budget is not None else config.context_byte_budget
    selected = []
    for chunk in candidates:
        item = {k: getattr(chunk, k) for k in ('file', 'start_line', 'end_line', 'content')}
        if not chunk.score:
            continue
        if len(selected) >= config.max_chunks:
            chunk.selection_reason = 'max_chunks'
        elif context_cost(selected + [item]) > budget:
            chunk.selection_reason = 'budget'
        else:
            chunk.selected = True
            chunk.selection_reason = 'selected'
            selected.append(item)
    for chunk in candidates:
        if not chunk.selected:
            chunk.content = ''
    return RepositoryContext(head=review_input.pr.head, diff_sha256=digest(review_input.diff.encode()),
        archive_sha256=snapshot.archive_sha256, repository=snapshot.repository, ranking=VERSION,
        config=config.model_dump(), budget_unit='serialized_utf8_bytes',
        budget=budget, used_budget=context_cost(selected), chunks=candidates)
