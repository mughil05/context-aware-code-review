"""Frozen Python archive fixtures; all acquisition/model calls mocked or blocked."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reviewer.archives import extract_snapshot
from reviewer.config import load_config
from reviewer.github import Snapshot, GitHubClient
from reviewer.models import OpenAIReviewer
from reviewer.pipeline import prepare_context, review_prompt, run_baseline, run_review
from reviewer.retrieval import analyze, context_cost, digest, retrieve, snapshot_sources
from reviewer.schemas import ReviewInput, BaselineRun
from reviewer.storage import save_run
from reviewer.cli import main
from test_benchmark import ROOT, response
from test_schemas import pr_data

DIFF = ('diff --git a/src/pkg/main.py b/src/pkg/main.py\n'
        '--- a/src/pkg/main.py\n+++ b/src/pkg/main.py\n@@ -3 +3 @@\n-old()\n+load_user(user_id)\n')
FILES = {
    'src/pkg/main.py': 'from .users import load_user\n\nload_user(user_id)\n\ndef caller():\n    return load_user(1)\n',
    'src/pkg/users.py': 'def load_user(user_id):\n    return database.lookup(user_id)\n',
    'src/remote/helpers.py': 'def load_user(user_id):\n    return user_id\n',
    'src/pkg/neighbor.py': 'def nearby():\n    return 1\n',
    'remote/lexical.py': '# load_user user_id database lookup\nvalue = 1\n',
    'remote/unrelated.py': 'def zebras():\n    return 9\n',
}


def snapshot(root, files, head, name='snapshot'):
    archive_path = root / (name + '.tar.gz')
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        wrapper = tarfile.TarInfo('repo-head'); wrapper.type = tarfile.DIRTYPE
        tar.addfile(wrapper)
        for path, text in files.items():
            raw = text.encode() if isinstance(text, str) else text
            member = tarfile.TarInfo('repo-head/' + path); member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    archive_path.write_bytes(stream.getvalue())
    directory = root / name
    sha = extract_snapshot(stream.getvalue(), directory)
    return Snapshot('owner/repo', head, sha, archive_path, directory)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.inp = ReviewInput(pr=pr_data(), diff=DIFF, diff_complete=True)
        self.config = load_config(ROOT / 'configs/experiment.toml')
        self.config.review.request_byte_budget = 32000
        self.snap = snapshot(self.root, FILES, self.inp.pr.head)
        guard = patch('socket.socket', side_effect=AssertionError('offline only'))
        guard.start(); self.addCleanup(guard.stop)

    def provider(self, findings=None):
        return OpenAIReviewer('gpt-6-luna', self.config.review, api_key='offline-secret',
                              transport=Mock(return_value=response(findings)))

    def get(self, **kwargs):
        return retrieve(self.inp, self.snap, self.config.retrieval, cache_dir=self.root / 'cache', **kwargs)

    def test_import_symbol_lexical_and_combined_ranking(self):
        result = self.get()
        by_file = {c.file: c for c in result.chunks}
        best = by_file['src/pkg/users.py']
        self.assertEqual(result.chunks[0].file, best.file)
        self.assertIn('imported_module', best.reasons)
        self.assertIn('referenced_definition:load_user', best.reasons)
        self.assertTrue(any(r.startswith('token_overlap:') for r in best.reasons))
        self.assertIn('nearby_module', best.reasons)
        self.assertIn('referenced_definition:load_user', by_file['src/remote/helpers.py'].reasons)
        self.assertTrue(any(r.startswith('token_overlap:') for r in by_file['remote/lexical.py'].reasons))
        self.assertIn('nearby_module', by_file['src/pkg/neighbor.py'].reasons)
        self.assertFalse(by_file['remote/unrelated.py'].selected)

    def test_changed_head_lines_are_never_retrieved_again(self):
        result = self.get()
        for c in result.chunks:
            if c.file == 'src/pkg/main.py':
                self.assertFalse(c.start_line <= 3 <= c.end_line)
                self.assertIn('same_file_outside_diff', c.reasons)

    def test_deterministic_order_cache_and_archive_member_order(self):
        first, second = self.get(), self.get()
        self.assertEqual(first, second)
        other = snapshot(self.root, dict(reversed(list(FILES.items()))), self.inp.pr.head, 'reordered')
        result = retrieve(self.inp, other, self.config.retrieval)
        self.assertEqual(result.chunks, first.chunks)
        self.assertEqual([(c.file, c.start_line) for c in first.chunks],
                         [(c.file, c.start_line) for c in sorted(first.chunks, key=lambda c: (-c.score, c.file, c.start_line, c.end_line, c.content_sha256))])

    def test_budget_max_chunks_and_selection_metadata(self):
        self.config.retrieval.context_byte_budget = 300
        self.config.retrieval.max_chunks = 1
        result = self.get()
        self.assertLessEqual(result.used_budget, 300)
        self.assertEqual(result.used_budget, context_cost(result.payload()))
        self.assertLessEqual(len(result.payload()), 1)
        self.assertTrue(any(not c.selected for c in result.chunks))
        for c in result.chunks:
            if c.selected:
                self.assertEqual(c.content_sha256, digest(c.content.encode()))
            else:
                self.assertEqual(c.content, '')
        zero = self.get(available_byte_budget=0)
        self.assertEqual(zero.payload(), [])
        self.assertEqual(zero.used_budget, 0)

    def test_context_fits_remaining_request_budget_without_changing_diff(self):
        provider = self.provider()
        base = provider.payload(self.inp.diff, review_prompt())
        base_size = len(json.dumps(base, ensure_ascii=False).encode()) + 4096
        provider.config.request_byte_budget = base_size + 200
        context = prepare_context(self.inp, provider, self.snap, self.config.retrieval)
        self.assertLessEqual(context.used_budget, 136)
        full = provider.payload(self.inp.diff, review_prompt(), context=context.payload())
        self.assertLessEqual(len(json.dumps(full, ensure_ascii=False).encode()) + 4096,
                             provider.config.request_byte_budget)
        self.assertEqual(json.loads(full['input'][0]['content'])['diff'], self.inp.diff)
        provider._transport.assert_not_called()

    def test_wrong_context_identity_is_rejected_before_model_call(self):
        provider, context = self.provider(), self.get()
        context.head = '0'*40
        with self.assertRaises(ValueError):
            run_review(self.inp, provider, mode='retrieval', context=context)
        provider._transport.assert_not_called()

    def test_ast_chunks_fallback_and_cache_invalidation(self):
        cache = self.root / 'analysis-cache'
        text = 'def first():\n    return 1\n\nclass Second:\n    pass\n'
        data = analyze(text, 2, cache)
        with patch('reviewer.retrieval.ast.parse', side_effect=AssertionError('cache miss')):
            self.assertEqual(analyze(text, 2, cache), data)
        analyze(text + '\n# change', 2, cache)
        analyze(text, 3, cache)
        self.assertEqual(len(list(cache.glob('*.json'))), 3)
        self.assertTrue(all(b - a + 1 <= 2 for a, b in data['ranges']))
        self.assertTrue(analyze('def broken(\nmore\n', 1)['ranges'])
        for path in cache.glob('*.json'): path.write_text('malformed')
        self.assertEqual(analyze(text, 2, cache), data)

    def test_excludes_artifacts_secrets_labels_vendor_and_binary(self):
        banned = {'golden/labels.py': 'LABEL_SENTINEL', 'results/scoring.py': 'LABEL_SENTINEL',
                  'reference/data.py': 'LABEL_SENTINEL', '.cache/data.py': 'LABEL_SENTINEL',
                  'node_modules/m.py': 'LABEL_SENTINEL', 'vendor/m.py': 'LABEL_SENTINEL',
                  'build/m.py': 'LABEL_SENTINEL', 'dist/m.py': 'LABEL_SENTINEL',
                  'pkg/credentials.py': 'SECRET_SENTINEL', 'pkg/settings.py': 'SECRET_SENTINEL',
                  'pkg/runtime.py': 'OPENAI_API_KEY = "SECRET_SENTINEL"',
                  'pkg/private.py': '# -----BEGIN RSA PRIVATE KEY-----',
                  'pkg/generated.py': '# auto-generated\nLABEL_SENTINEL',
                  'pkg/binary.py': b'\x00\xff', 'labels.json': '{"golden":"LABEL_SENTINEL"}',
                  'README.md': 'LABEL_SENTINEL', 'pr.diff': DIFF}
        snap = snapshot(self.root, FILES | banned, self.inp.pr.head, 'excluded')
        sources = snapshot_sources(snap, self.inp.pr.head)
        self.assertEqual(set(sources), set(FILES))
        result = retrieve(self.inp, snap, self.config.retrieval)
        self.assertNotIn('LABEL_SENTINEL', result.model_dump_json())
        self.assertNotIn('SECRET_SENTINEL', result.model_dump_json())

    def test_archive_hash_head_and_links_fail_closed(self):
        with self.assertRaises(ValueError):
            retrieve(self.inp, replace(self.snap, commit='0'*40), self.config.retrieval)
        with self.assertRaises(ValueError):
            retrieve(self.inp, replace(self.snap, archive_sha256='0'*64), self.config.retrieval)
        # Snapshot directory mutations cannot influence retrieval from archive bytes.
        (self.snap.directory / 'src/pkg/users.py').write_text('SECRET_SENTINEL')
        self.assertNotIn('SECRET_SENTINEL', self.get().model_dump_json())
        link = self.root / 'linked.tar.gz'; link.symlink_to(self.snap.archive_path)
        with self.assertRaises(ValueError):
            retrieve(self.inp, replace(self.snap, archive_path=link), self.config.retrieval)

    def test_shared_prompt_contract_and_isolated_payloads(self):
        provider = self.provider()
        context = prepare_context(self.inp, provider, self.snap, self.config.retrieval, cache_dir=self.root / 'cache')
        baseline = run_baseline(self.inp, provider)
        wire_a = provider._transport.call_args.args[0]
        retrieval = run_review(self.inp, provider, mode='retrieval', context=context)
        wire_b = provider._transport.call_args.args[0]
        self.assertEqual(set(json.loads(wire_a['input'][0]['content'])), {'diff'})
        self.assertEqual(set(json.loads(wire_b['input'][0]['content'])), {'diff', 'repository_context'})
        self.assertEqual(wire_a['instructions'], wire_b['instructions'])
        self.assertEqual(wire_a['text'], wire_b['text'])
        self.assertEqual(baseline.settings, retrieval.settings)
        self.assertEqual(baseline.prompt_sha256, retrieval.prompt_sha256)
        self.assertIsNone(baseline.retrieval)
        self.assertEqual(retrieval.retrieval, context)
        self.assertIn('HEAD-side line range', review_prompt())
        self.assertIn('still point to the PR diff', review_prompt())
        self.assertIn('Do not report a finding if it cannot be', review_prompt())
        self.assertTrue(review_prompt().startswith('Review contract version: 3.'))
        path = save_run(retrieval, self.root / 'runs')
        self.assertEqual(path.name, 'retrieval.json')
        self.assertEqual(BaselineRun.model_validate_json(path.read_bytes()), retrieval)
        with self.assertRaises(ValueError): run_review(self.inp, provider, context=context)

    def test_retrieved_files_do_not_extend_finding_anchors(self):
        from test_benchmark import finding
        context = self.get()
        for path, line in [('src/pkg/users.py', 1), ('src/pkg/main.py', 5)]:
            bad = finding() | {'file': path, 'start_line': line, 'end_line': line}
            provider = self.provider([bad])
            run = run_review(self.inp, provider, mode='retrieval', context=context)
            self.assertEqual(run.result.status, 'failed')
            self.assertIn('invalid_reference', run.result.error)
        good = finding() | {'file': 'src/pkg/main.py', 'start_line': 3, 'end_line': 3}
        self.assertEqual(run_review(self.inp, self.provider([good]), mode='retrieval', context=context).result.status, 'success')

    def test_cli_retrieval_reuses_acquisition(self):
        acquired = SimpleNamespace(review_input=self.inp, artifact_directory=self.root, head_snapshot=self.snap)
        provider = self.provider()
        # Keep all cache writes in the test temporary directory.
        self.config.cache_dir = str(self.root / 'cache')
        with patch.dict('os.environ', {'REVIEW_PROVIDER': 'openai', 'REVIEW_MODEL': 'gpt-6-luna'}, clear=True), \
             patch('reviewer.cli.load_config', return_value=self.config), \
             patch('reviewer.cli.GitHubClient') as client, \
             patch('reviewer.cli.OpenAIReviewer', return_value=provider), redirect_stdout(io.StringIO()):
            client.return_value.acquire.return_value = acquired
            code = main(['review', '--repo', 'o/r', '--pr', '1', '--mode', 'retrieval',
                         '--input-token-limit', '32000', '--results-dir', str(self.root / 'results')])
        self.assertEqual(code, 0)
        self.assertTrue(client.return_value.acquire.call_args.kwargs['snapshots'])
        self.assertEqual(provider._transport.call_count, 1)
        self.assertTrue(json.loads(provider._transport.call_args.args[0]['input'][0]['content'])['repository_context'])

    def test_frozen_head_acquisition_uses_sha_archive_only(self):
        downloader = Mock(return_value=self.snap.archive_path.read_bytes())
        client = GitHubClient(token='', downloader=downloader)
        result = client.frozen_head(self.inp.pr, source_repository='owner/repo', cache_dir=self.root / 'frozen')
        self.assertEqual(result.commit, self.inp.pr.head)
        self.assertTrue(downloader.call_args.args[0].endswith('/tarball/' + self.inp.pr.head))
        self.assertEqual(result.archive_sha256, self.snap.archive_sha256)


if __name__ == '__main__':
    unittest.main()
