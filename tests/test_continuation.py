"""Frozen-prefix continuation tests: synthetic local inputs and mocked transports."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import socket
import unittest
from unittest.mock import patch

from reviewer.benchmark import Benchmark, Subset, digest
from reviewer.config import load_config
from reviewer.continuation import (ORIGIN, MODES, continue_frozen_run, validate_source)
from reviewer.evaluation import _persist_result, write_json
from reviewer.final_exclusions import TASK_ID, exclusion_record
from reviewer.models import OpenAIReviewer, ModelError
from reviewer.pipeline import run_review, run_full, review_settings, review_prompt
from reviewer.request_budget import BUDGET_METHOD
from reviewer.schemas import RepositoryContext, ContextChunk, ReviewInput
from reviewer.static_analysis import StaticAnalysisResult
from reviewer.verification import verification_prompt
from test_benchmark import DIFF, finding, response
from test_verification import verify_response

ROOT = Path(__file__).resolve().parents[1]


def tree_hash(root):
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.source = self.root / ORIGIN; self.source.mkdir()
        self.guard = patch.object(socket, 'create_connection', side_effect=AssertionError('network forbidden'))
        self.guard.start(); self.addCleanup(self.guard.stop)
        self.config = load_config(ROOT/'configs/experiment.toml'); self.config.review.max_output_tokens=8192
        self.calls=[]; self.vcalls=[]; self.serial=0
        self.reviewer=OpenAIReviewer('gpt-6-luna',self.config.review,api_key='offline',transport=self.transport)
        self.verifier=OpenAIReviewer('gpt-6-luna',self.config.review,api_key='offline',transport=self.vtransport)
        subset=Subset.model_validate_json((ROOT/'configs/reviewbench-final20.json').read_bytes())
        lock=json.loads((ROOT/'configs/reviewbench-final20.lock.json').read_text())
        subset=subset.model_copy(update={'tasks':[t.model_copy(update={'diff_sha256':digest(DIFF.encode())})
            if t.task_id!=TASK_ID else t for t in subset.tasks]})
        for t in lock['tasks']:
            t.update(diff_sha256=digest(DIFF.encode()),head_archive_sha256='a'*64)
            t.pop('acquisition_warnings',None)
        self.benchmark=Benchmark(subset,digest((ROOT/'configs/reviewbench-final20.json').read_bytes()),
            lock['benchmark_archive_sha256'],digest(json.dumps(lock).encode()),
            {t.task_id:{'pr_key':t.task_id,'pr':t.pr.model_dump(),'findings':[{'message':'GOLDEN_ONLY_SENTINEL','tp_fp':'tp'}]} for t in subset.tasks},lock)
        self.tasks=[t for t in subset.tasks if t.task_id!=TASK_ID]
        self.contexts={t.task_id:self.context(t) for t in self.tasks}
        for name in ('inputs','reference'):(self.source/name).mkdir()
        write_json(self.source/'subset.json',subset.model_dump())
        rows=[]
        for t in self.tasks:
            inp=ReviewInput(pr=t.pr,diff=DIFF,diff_complete=True)
            write_json(self.source/'inputs'/f'{t.task_id}.json',inp.model_dump())
            write_json(self.source/'reference'/f'{t.task_id}.json',self.benchmark.golden[t.task_id])
            for mode in MODES:
                rows.append({'task_id':t.task_id,'mode':mode,'pr':t.pr.model_dump(),
                    'diff_sha256':t.diff_sha256,'diff_base':None,'model':'gpt-6-luna',
                    'golden_sha256':t.golden_sha256,'status':'ready','scoring_status':'paused','quality_metrics':None})
        self.summary={'schema_version':1,'benchmark':'reviewbench','status':'failed',
            'modes':MODES,'mode':','.join(MODES),'tasks':rows,
            'benchmark_revision':subset.benchmark_revision,'benchmark_archive_sha256':self.benchmark.archive_sha256,
            'manifest_sha256':subset.manifest_sha256,'subset_sha256':self.benchmark.subset_sha256,
            'diff_lock_sha256':self.benchmark.diff_lock_sha256,'intended_task_count':20,
            'evaluable_task_count':19,'requested_reviews':76,'excluded_pre_execution_count':1,
            'excluded_tasks':[exclusion_record(next(t for t in subset.tasks if t.task_id==TASK_ID))],
            'scoring_status':'paused','quality_metrics':None,'requested_model':'gpt-6-luna',
            'verifier_model':'gpt-6-luna','verifier_provider':'openai',
            'review_settings':review_settings(self.reviewer),'verifier_settings':review_settings(self.verifier),
            'prompt_sha256':digest(review_prompt().encode()),'verifier_prompt_sha256':digest(verification_prompt().encode()),
            'review_prompt_version':'v3','verifier_prompt_version':'1',
            'context_byte_budget':6000,'static_signal_byte_budget':2000,
            'retrieval_config':self.config.retrieval.model_dump(),
            'static_analysis_config':self.config.static_analysis.model_dump(),
            'request_budget':lock['request_budget'],'completed_reviews':13,'reused_reviews':0}
        tools={}
        for n,row in enumerate(rows[:14]):
            t=self.tasks[n//4];mode=row['mode'];inp=ReviewInput(pr=t.pr,diff=DIFF,diff_complete=True)
            context=self.contexts[t.task_id];static=self.static(inp)
            if mode=='full':
                run=run_full(inp,self.reviewer,self.verifier,tools_run=tools[t.task_id],context=context,
                             static_analysis=static,signal_budget=2000)
            else:
                if n==13:
                    with patch.object(self.reviewer,'_transport',side_effect=ModelError('timeout','OpenAI request timed out; completion/billing is unknown')):
                        run=run_review(inp,self.reviewer,mode=mode,context=context)
                else:
                    run=run_review(inp,self.reviewer,mode=mode,context=context if mode!='baseline' else None,
                                   static_analysis=static if mode=='tools' else None)
                if mode=='tools':tools[t.task_id]=run
            mode_root=self.source/'modes'/mode
            (mode_root/'runs').mkdir(parents=True,exist_ok=True);(mode_root/'candidates').mkdir(parents=True,exist_ok=True)
            _persist_result(mode_root,t,run,row,None)
            for k in ('result_file','prediction_file'):
                if k in row:row[k]=str(Path('modes')/mode/row[k])
        write_json(self.source/'summary.json',self.summary)
        origin_hash = patch('reviewer.continuation.ORIGIN_SUMMARY_SHA256',
                            digest((self.source/'summary.json').read_bytes()))
        origin_hash.start(); self.addCleanup(origin_hash.stop)
        self.calls.clear();self.vcalls.clear()
        self.before=tree_hash(self.source)
        self.client=SimpleNamespace(frozen_head=lambda pr,**kwargs:SimpleNamespace(commit=pr.head,archive_sha256='a'*64))

    def context(self,t):
        return RepositoryContext(head=t.pr.head,diff_sha256=digest(DIFF.encode()),archive_sha256='a'*64,
            repository=t.source_repository,ranking='structural_lexical_v1',config=self.config.retrieval.model_dump(),
            budget_unit='serialized_utf8_bytes',budget=6000,used_budget=100,overall_request_byte_ceiling=73728,
            reserved_static_signal_byte_cap=2000,reserved_static_signal_estimator_bytes=8256,
            chunks=[ContextChunk(file='src/helper.py',start_line=1,end_line=1,content='def helper(): pass',
                content_sha256=digest(b'def helper(): pass'),score=1,reasons=['test'],selected=True,selection_reason='selected')])

    def static(self,inp,*args):
        return StaticAnalysisResult(tool='bandit',version='1.7.8',head=inp.pr.head,archive_sha256='a'*64,
            analyzed_files=['src/a.py'],config={'signal_byte_budget':2000},timeout_seconds=60,
            status='no_findings',error=None,findings=[],metrics={})

    def transport(self,payload,key,timeout):
        self.calls.append(deepcopy(payload));self.serial+=1
        self.assertEqual(timeout,60)
        result=response([finding()] if 'static_analysis_signals' in json.loads(payload['input'][0]['content']) else [])
        result['id']=f'resp_{self.serial}';return result

    def vtransport(self,payload,key,timeout):
        self.vcalls.append(deepcopy(payload));self.serial+=1
        result=verify_response();result['id']=f'verify_{self.serial}';return result

    def validate(self,source=None):
        return validate_source(source or self.source,self.benchmark,self.reviewer,self.verifier,
                               self.config.retrieval,self.config.static_analysis)

    def execute(self,source=None):
        with patch('reviewer.continuation.prepare_context',side_effect=lambda inp,*a,**kw:self.contexts[next(t.task_id for t in self.tasks if t.pr==inp.pr)]), \
             patch('reviewer.continuation.prepare_static_analysis',side_effect=self.static):
            return continue_frozen_run(source or self.source,self.benchmark,self.reviewer,self.verifier,self.client,
                self.root/'out',modes=MODES,retrieval_config=self.config.retrieval,
                static_config=self.config.static_analysis,cache_dir=self.root/'cache')

    def test_complete_prefix_copy_first_request_pairing_and_accounting(self):
        plan=self.validate();self.assertEqual(plan.prefix,13)
        failed=plan.runs[(self.tasks[3].task_id,'retrieval')]
        from reviewer.continuation import reviewer_payload
        expected=reviewer_payload(plan.inputs[self.tasks[3].task_id],failed,self.reviewer)
        path,report=self.execute()
        self.assertEqual(report['status'],'generation_complete_scoring_paused')
        self.assertEqual(self.calls[0],expected)
        self.assertEqual(len(self.calls),47) # 57 reviewer generations minus the 10 completed.
        self.assertEqual(report['completed_reviews'],76)
        self.assertEqual(report['reused_reviews'],13)
        self.assertEqual(report['newly_completed_observations'],63)
        self.assertEqual(report['requested_reviews'],76)
        self.assertEqual(report['reviewer_successful_responses'],57)
        self.assertEqual(report['reviewer_failed_attempts'],1)
        self.assertEqual(report['unknown_usage_billing_attempts'],1)
        self.assertEqual(report['excluded_tasks'][0]['task_id'],TASK_ID)
        self.assertEqual(tree_hash(self.source),self.before)
        for row in self.summary['tasks'][:13]:
            for field in ('result_file','prediction_file'):
                self.assertEqual((path.parent/row[field]).read_bytes(),(self.source/row[field]).read_bytes())
        history=report['historical_failed_observations'][0]
        self.assertEqual((path.parent/history['artifact_file']).read_bytes(),
                         (self.source/self.summary['tasks'][13]['result_file']).read_bytes())
        for payload in self.calls+self.vcalls:
            self.assertNotIn('GOLDEN_ONLY_SENTINEL',json.dumps(payload))

    def test_timeout_stops_and_second_continuation_reuses_prefix(self):
        with patch.object(self.reviewer,'_transport',side_effect=ModelError('timeout','completion/billing unknown')) as call:
            path,report=self.execute()
        self.assertEqual(call.call_count,1)
        self.assertEqual(report['completed_reviews'],13)
        self.assertEqual(report['historical_failed_observation_count'],1)
        self.assertEqual(report['unknown_usage_billing_attempts'],2)
        frozen=tree_hash(path.parent)
        plan=self.validate(path.parent);self.assertEqual(plan.prefix,13)
        final,done=self.execute(path.parent)
        self.assertEqual(done['completed_reviews'],76)
        self.assertEqual(done['historical_failed_observation_count'],2)
        self.assertEqual(done['unknown_usage_billing_attempts'],2)
        self.assertEqual(tree_hash(path.parent),frozen)

    def test_changed_boundary_payload_fails_before_call(self):
        original=self.reviewer.payload
        def changed(*args,**kwargs):
            payload=original(*args,**kwargs)
            payload['instructions']+='tampered'
            return payload
        with patch.object(self.reviewer,'payload',side_effect=changed):
            _,report=self.execute()
        self.assertEqual(report['status'],'failed')
        self.assertEqual(self.calls,[]);self.assertEqual(self.vcalls,[])
        self.assertEqual(tree_hash(self.source),self.before)

    def test_artifact_and_frozen_control_tampering_rejected(self):
        def check():
            with self.assertRaises(ValueError):self.validate()
            self.assertEqual(self.calls,[])
        for name in ('model','request_byte_budget','max_output_tokens','timeout_seconds'):
            with self.subTest(name=name):
                obj=self.reviewer if name=='model' else self.reviewer.config
                old=getattr(obj,name);setattr(obj,name,'different-model' if name=='model' else old+1)
                check();setattr(obj,name,old)
        for field in ('result_file','prediction_file'):
            p=self.source/self.summary['tasks'][0][field];old=p.read_bytes();p.write_bytes(old+b' ')
            check();p.write_bytes(old)
        original=deepcopy(self.summary)
        for key,value in [('diff_lock_sha256','0'*64),('completed_reviews',12),('prompt_sha256','0'*64),
                          ('context_byte_budget',5000),('static_signal_byte_budget',1000)]:
            self.summary=deepcopy(original);self.summary[key]=value
            (self.source/'summary.json').write_text(json.dumps(self.summary));check()
        (self.source/'summary.json').write_text(json.dumps(original))

    def test_nonprefix_success_rejected(self):
        self.summary['tasks'][15]['status']='success'
        (self.source/'summary.json').write_text(json.dumps(self.summary))
        with self.assertRaisesRegex(ValueError,'non-prefix'):self.validate()

    def test_partial_verifier_failure_replays_completed_decision_without_rebilling(self):
        original_transport = self.transport
        def two_proposals(payload,key,timeout):
            result=original_transport(payload,key,timeout)
            if 'static_analysis_signals' in json.loads(payload['input'][0]['content']):
                result['output'][0]['content'][0]['text']=json.dumps({'findings':[finding(),finding()]})
            return result
        vcount=0
        def fail_second(payload,key,timeout):
            nonlocal vcount
            vcount+=1
            if vcount==2: raise ModelError('timeout','completion/billing unknown')
            return self.vtransport(payload,key,timeout)
        with patch.object(self.reviewer,'_transport',side_effect=two_proposals), \
             patch.object(self.verifier,'_transport',side_effect=fail_second):
            path,report=self.execute()
        self.assertEqual(report['completed_reviews'],15)
        self.assertEqual(report['tasks'][15]['mode'],'full')
        self.assertEqual(report['tasks'][15]['status'],'failed')
        self.assertEqual(len(self.calls),2)  # #14 and #15 only.
        plan=self.validate(path.parent);self.assertEqual(plan.prefix,15)
        self.calls.clear();self.vcalls.clear()
        # Resume starts with D, using the persisted C and first verifier decision.
        with patch.object(self.reviewer,'_transport',side_effect=ModelError('timeout','stop after D')):
            final,done=self.execute(path.parent)
        self.assertEqual(done['completed_reviews'],16)
        self.assertEqual(len(self.vcalls),1)
        self.assertEqual(done['tasks'][15]['status'],'success')
        self.assertEqual(tree_hash(self.source),self.before)
        self.validate(final.parent)

    def test_completed_pair_tampering_rejected_even_with_updated_artifact_hash(self):
        row=self.summary['tasks'][3];path=self.source/row['result_file']
        raw=json.loads(path.read_text());raw['proposed_findings']=[]
        path.write_text(json.dumps(raw));row['result_sha256']=digest(path.read_bytes())
        (self.source/'summary.json').write_text(json.dumps(self.summary))
        with self.assertRaises(ValueError):self.validate()
        self.assertEqual(self.calls,[])

    def test_cli_explicit_frozen_flags_and_76_denominator(self):
        import io
        from contextlib import redirect_stdout
        from reviewer.cli import main
        args=['evaluate','--benchmark','reviewbench','--benchmark-archive','unused.tar.gz',
              '--subset','configs/reviewbench-final20.json','--config',str(ROOT/'configs/experiment.toml'),
              '--modes','baseline','retrieval','tools','full','--request-byte-budget','73728',
              '--context-byte-budget','6000','--max-output-tokens','8192',
              '--continue-run',str(self.source)]
        with patch.dict('os.environ',{'REVIEW_PROVIDER':'openai','REVIEW_MODEL':'gpt-6-luna',
                    'VERIFY_PROVIDER':'openai','VERIFY_MODEL':'gpt-6-luna','OPENAI_API_KEY':'offline'}), \
             patch('reviewer.benchmark.load_benchmark',return_value=self.benchmark), \
             patch('reviewer.cli.OpenAIReviewer',side_effect=[self.reviewer,self.verifier]), \
             patch('reviewer.continuation.continue_frozen_run',return_value=(Path('out/summary.json'),
                {'status':'generation_complete_scoring_paused','completed_reviews':76,
                 'requested_reviews':76,'reused_reviews':13,'model_attempts':0})) as runner:
            stream=io.StringIO()
            with redirect_stdout(stream):status=main(args)
        self.assertEqual(status,0);self.assertIn('76/76',stream.getvalue())
        self.assertEqual(runner.call_args.args[0],str(self.source))
        self.assertEqual(runner.call_args.kwargs['modes'],MODES)
        self.assertEqual(self.calls,[])
