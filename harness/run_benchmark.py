"""Real native-tree and serial-AR generation; all online method costs timed."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import traceback
import weakref
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness'))
from resources import ResourceGuard
import torch
torch.set_num_threads(1)
from refined_controller import RefinedController
from speed_controller import SpeedController
from terminal_update import TerminalUpdate
from lvspec_controller import LVSpecController
from compact_logits import CompactTreeLogits, tree_decoding


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f'Preserve evidence: {path}'
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def clean(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return clean(tensor.item() if tensor.numel() == 1 else tensor.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    return value


def input_digest(inputs):
    identity = hashlib.sha256()
    for key, value in sorted(inputs.items()):
        identity.update(key.encode())
        tensor = value.detach().cpu().contiguous()
        identity.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        identity.update(tensor.view(torch.uint8).numpy().tobytes())
    return identity.hexdigest()




class Runtime:
    def __init__(self, phase, shard):
        self.phase, self.shard = phase, shard
        self.label = f'{phase}_shard{shard}'
        self.config_path = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
        self.cfg = read(self.config_path)
        self.guard = ResourceGuard(self.label,ROOT)
        for name,identity in self.cfg['harness_sha256'].items():
            assert digest(ROOT/'harness'/name)==identity,name
        for task,entry in self.cfg['data'].items():
            for split in ['dev','test']:
                assert digest(entry[split+'_path'])==entry[split+'_sha256']
        self.verify_checkout()
        if phase == 'main':
            assert self.cfg['config_finalized_before_generation']
            assert read(ROOT/'audit/MAIN_READY.json')['status'] == 'pass'
        sys.path.insert(0, self.cfg['upstream'])
        os.chdir(self.cfg['upstream'])
        from vispec.model import spec_model_ours
        from transformers import AutoProcessor
        self.module = spec_model_ours
        self.model = spec_model_ours.SpecModel.from_pretrained(
            base_model_path=self.cfg['target'], spec_model_path=self.cfg['draft'],
            torch_dtype='auto', low_cpu_mem_usage=True, device_map='auto', **self.cfg['tree']).eval()
        self.processor = AutoProcessor.from_pretrained(self.cfg['target'],**self.cfg['processor'])
        self.tokenizer = self.model.tokenizer
        self.loaded_vispec_sources={}
        for name,mod in list(sys.modules.items()):
            if name.startswith('vispec') and getattr(mod,'__file__',None):
                path=Path(mod.__file__).resolve();assert path.is_relative_to(Path(self.cfg['upstream']).resolve()),(name,path)
                self.loaded_vispec_sources[name]=dict(path=str(path),sha256=digest(path))
        self.original = {key: getattr(self.module, key) for key in RefinedController.FUNCTIONS}
        self.counts, self.counter_handles = {}, []
        for name, module in [('target', self.model.base_model), ('draft', self.model.spec_layer), ('vision', self.model.base_model.visual)]:
            def count(mod, args, out, key=name):
                self.counts[key] = self.counts.get(key, 0)+1
            self.counter_handles.append(module.register_forward_hook(count))
        self.compact = False
        self.compact_stats, self.logit_refs = [], []
        self.meta = dict(config_sha256=digest(self.config_path), runner_sha256=digest(__file__),
            python=sys.version, executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'], gpu=torch.cuda.get_device_name(),
            target_dtype=str(self.model.base_model.dtype), model_class=type(self.model).__module__,
            attention_class=type(self.model.base_model.model.layers[-1].self_attn).__name__,
            phase=phase, source_sha256=self.cfg['harness_sha256'],
            loaded_vispec_sources=self.loaded_vispec_sources, processor_settings=self.cfg['processor'], AR_memory_includes_idle_loaded_Draft=True)
        write(ROOT/f'audit/{self.label}.loaded.json', self.meta)

    def restore(self):
        for key, function in self.original.items():
            setattr(self.module, key, function)
        if self.compact:
            self.module.tree_decoding = tree_decoding

    def wrap_evaluate(self, evaluate, audit):
        def wrapped(logits, candidates, processor):
            if not self.compact:
                return evaluate(logits, candidates, processor)
            assert isinstance(logits, CompactTreeLogits)
            self.logit_refs.append(weakref.ref(logits))
            reference = logits.materialize_reference() if audit else None
            if reference is not None:
                assert torch.equal(logits[:, :-1].argmax(-1), reference[:, :-1].argmax(-1))
            result = evaluate(logits, candidates, processor)
            if reference is not None:
                assert torch.equal(result[2], reference[result[0], result[1]])
            self.compact_stats.append(logits.audit_statistics())
            return result
        return wrapped

    def inputs(self, sample):
        from PIL import Image
        torch.cuda.synchronize()
        prep_started=time.perf_counter()
        assert digest(sample['image_path']) == sample['image_sha256']
        image = Image.open(sample['image_path']).convert('RGB')
        messages = [dict(role='system', content=[dict(type='text', text=self.cfg['system_prompt'])]),
                    dict(role='user', content=[dict(type='text', text=sample['prompt_text']), dict(type='image')])]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        result=self.processor(images=[image], text=prompt, return_tensors='pt').to('cuda:0')
        torch.cuda.synchronize()
        self.last_preprocessing_seconds=time.perf_counter()-prep_started
        return result

    @torch.inference_mode()
    def serial_ar(self, inputs, *, output_hidden_states=False):
        self.module.reset_tree_mode(self.model)
        if not hasattr(self.model, 'past_key_values'):
            cache, data, lengths = self.module.initialize_past_key_values(self.model.base_model)
            self.model.past_key_values, self.model.past_key_values_data, self.model.current_length_data = cache, data, lengths
        self.model.current_length_data.zero_()
        cache = self.model.past_key_values
        result = self.model.base_model(**inputs, past_key_values=cache, use_cache=True,
                                      output_hidden_states=output_hidden_states, return_dict=True)
        logits = result.logits[:, -1]
        del result
        torch.cuda.synchronize()
        start_decode = time.perf_counter()
        generated = []
        max_steps = self.cfg['max_length']-self.model.spec_layer.total_tokens-10
        for step in range(max_steps):
            token = logits.argmax(-1)
            generated.append(int(token))
            if (generated[-1] == self.tokenizer.eos_token_id
                    or len(generated) > self.cfg['max_new_tokens'] or step+1 >= max_steps):
                break
            result = self.model.base_model(input_ids=token[:, None], past_key_values=cache,
                use_cache=True, output_hidden_states=output_hidden_states, return_dict=True)
            logits = result.logits[:, -1]
            del result
        torch.cuda.synchronize()
        decode = time.perf_counter()-start_decode
        ids = torch.tensor([generated], device=inputs.input_ids.device, dtype=inputs.input_ids.dtype)
        return torch.cat((inputs.input_ids, ids), dim=1), len(generated), len(generated)-1, [0]*len(generated), decode

    def generate(self, sample, inputs, method, variant, repeat, order, audit, identity):
        self.guard.assert_free()
        self.compact = bool(variant.get('compact_logits', True)) and variant['family'] != 'AR'
        self.restore()
        self.compact_stats, self.logit_refs = [], []
        self.counts = {}
        controller = controller_ref = None
        terminal = None
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline, reserved_baseline = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                if variant.get('terminal_skip', False):
                    terminal = TerminalUpdate(self.original['update_inference_inputs'], prompt_length=inputs.input_ids.shape[1],
                        max_new_tokens=self.cfg['max_new_tokens'], eos_token_id=self.tokenizer.eos_token_id,audit=audit)
                    self.module.update_inference_inputs = terminal
                if variant['family'] not in ['AR', 'vispec']:
                    cls = LVSpecController if variant['family']=='LVSPEC' else SpeedController if variant.get('controller')=='speed' else RefinedController
                    controller = cls(self.module, self.model, variant,
                        inputs.input_ids, inputs.image_grid_thw, audit=audit,
                        max_new_tokens=self.cfg['max_new_tokens'], image_id=sample['image_id'], prompt_kind=self.task)
                    controller_ref = weakref.ref(controller)
                # Install only at module entry: never write a closure capturing
                # a bound method back onto that controller instance.
                self.module.evaluate_posterior = self.wrap_evaluate(self.module.evaluate_posterior, audit)
                if variant['family'] == 'AR':
                    out, new, index, acceptance, decode = self.serial_ar(inputs,
                        output_hidden_states=variant.get('ar_output_hidden_states', False))
                else:
                    out, new, index, acceptance, decode = self.model.specgenerate(**inputs,
                        temperature=0., max_new_tokens=self.cfg['max_new_tokens'], max_length=self.cfg['max_length'],
                        log=True, return_acceptance_len=True, return_decode_time=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            stats = clean(controller.final_stats()) if controller is not None else {}
            stats['terminal_update'] = clean(terminal.statistics()) if terminal is not None else dict(enabled=False)
            paths = clean(controller.acceptance_paths) if controller is not None else []
            counts = dict(self.counts)
        finally:
            if controller is not None:
                controller.restore()
            self.compact = False
            self.restore()
        controller = None
        assert controller_ref is None or controller_ref() is None, 'Request controller ownership cycle'
        assert all(ref() is None for ref in self.logit_refs), 'Native logits ownership cycle'
        if self.compact_stats and not audit:
            assert all(row['audit_reference_materializations'] == 0 for row in self.compact_stats)
        raw = out[0, inputs.input_ids.shape[1]:].tolist()
        budget = raw[:self.cfg['max_new_tokens']]
        eos = budget.index(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id in budget else None
        returned = budget[:eos] if eos is not None else budget
        completion = len(returned)+int(eos is not None)
        assert sum(acceptance)+len(acceptance) == len(raw)
        assert counts.get('target', 0) == (len(raw) if variant['family'] == 'AR' else len(acceptance)+1)
        assert counts.get('vision', 0) == 1
        if variant['family'] == 'AR':
            assert counts.get('draft', 0) == 0
        effective = rounds = committed = 0
        for accepted in acceptance:
            if committed >= completion:
                break
            rounds += 1
            effective += max(0, min(accepted, completion-committed-1))
            committed += accepted+1
        compact_stats = {key: sum(row[key] for row in self.compact_stats)
                         for key in sorted({key for row in self.compact_stats for key in row})}
        stats['compact_logits'] = compact_stats
        row = dict(task=self.task, prompt_kind='task_short', phase=self.phase, method=method,
            temperature=0., repeat=repeat, ordinal=sample['ordinal'], image_id=sample['image_id'],
            group_id=sample['group_id'],group_ordinal=sample['group_ordinal'], question_id=sample['question_id'], image_path=sample['image_path'],
            image_sha256=sample['image_sha256'], physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
            gpu=os.environ['CUDA_VISIBLE_DEVICES'], shard=self.shard, schedule_position=order,
            prompt_text=sample['prompt_text'], prompt_token_ids=inputs.input_ids[0].tolist(),
            image_grid_thw=inputs.image_grid_thw.tolist(), input_tensor_sha256=identity,
            raw_token_ids=raw, returned_token_ids=returned,
            output_text=self.tokenizer.decode(returned, skip_special_tokens=True).strip(),
            raw_output_text=self.tokenizer.decode(raw, skip_special_tokens=False),
            native_output_text=self.tokenizer.decode(raw, skip_special_tokens=False),
            prompt_tokens=inputs.input_ids.shape[1], seconds=seconds, decode_seconds=float(decode),
            input_preprocessing_seconds=getattr(self,'last_preprocessing_seconds',None),
            preprocessing_reused_for_paired_requests=True,
            completion_tokens=completion, acceptance_length=acceptance, acceptance_paths=paths,
            effective_accepted_draft_tokens=effective, effective_verification_rounds=rounds,
            eos_index=eos, cap_reached=eos is None and len(budget) == self.cfg['max_new_tokens'],
            native_new_tokens=int(new), native_index=int(index),
            baseline_allocated_bytes=baseline, baseline_reserved_bytes=reserved_baseline,
            peak_allocated_bytes=peak, peak_reserved_bytes=reserved,
            peak_increment_bytes=peak-baseline, peak_reserved_increment_bytes=reserved-reserved_baseline,
            signal_stats=stats, model_calls=counts, extra_target_forwards=0,
            request_lifecycle=dict(immediate_release_checked=True, live_controllers_after_request=0,
                                   live_logits_after_request=0, explicit_gc_or_empty_cache=False),
            diagnostic_recording=audit, config_sha256=self.meta['config_sha256'])
        self.guard.assert_free()
        return row

    def verify_checkout(self):
        import subprocess
        root=Path(self.cfg['upstream'])
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==self.cfg['upstream_commit']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(), 'Fresh upstream changed'
        manifest=read(ROOT/'audit/CLEAN_CHECKOUT.json')
        for name,expected in manifest['tracked_sha256'].items():assert digest(root/name)==expected,name

    def run(self):
        directory=ROOT/'results'/self.phase;directory.mkdir(parents=True,exist_ok=True)
        for self.task in self.cfg['tasks']:
            split='test' if self.phase=='main' else 'dev'
            samples=read(self.cfg['data'][self.task][split+'_path'])
            if self.phase=='parity':samples=[s for s in samples if s['group_ordinal']<2]
            samples=[s for s in samples if s['group_ordinal']%2==self.shard]
            methods=(self.cfg['methods_by_task'][self.task] if self.phase=='main' else
                     self.cfg['parity_methods'] if self.phase=='parity' else self.cfg['development_methods'])
            repeats=self.cfg['repetitions'] if self.phase!='parity' else 1
            target=directory/f'{self.phase}_{self.task}_shard{self.shard}.jsonl';assert not target.exists(),str(target)
            inputs=self.inputs(samples[0]);identity=input_digest(inputs);warmups=[]
            for name,variant in methods.items():
                for j in range(self.cfg['warmups'] if self.phase=='main' else 1):
                    warmups.append(self.generate(samples[0],inputs,name,variant,-1,j,False,identity))
            write(ROOT/f'audit/{self.label}_{self.task}.warmup.json',dict(requests=len(warmups),seconds=sum(r['seconds'] for r in warmups),lifecycle_checked_every_request=True))
            count,checks=0,[];started=time.perf_counter()
            with target.open('w',encoding='utf-8') as handle:
                for sample in samples:
                    inputs=self.inputs(sample);identity=input_digest(inputs)
                    for repeat in range(repeats):
                        scheduled=list(methods)
                        random.Random(self.cfg['order_seed']+self.cfg['tasks'].index(self.task)*1000003+sample['ordinal']*101+repeat*10007).shuffle(scheduled)
                        selected={}
                        for order,method in enumerate(scheduled):
                            try:
                                row=self.generate(sample,inputs,method,methods[method],repeat,order,self.phase=='parity',identity)
                            except Exception as error:
                                write(directory/f'{self.label}_{self.task}.failure.json',dict(exception=repr(error),traceback=traceback.format_exc(),image_id=sample['image_id'],question_id=sample['question_id'],method=method,repeat=repeat,completed_records=count,config_sha256=self.meta['config_sha256']))
                                raise
                            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();selected[method]=row;count+=1
                        if self.phase=='parity':
                            for left,right in self.cfg['parity_pairs']:
                                fields=['raw_token_ids','returned_token_ids','acceptance_length','eos_index','cap_reached']
                                result=dict(question_id=sample['question_id'],method=left,baseline=right,**{k:selected[left][k]==selected[right][k] for k in fields})
                                if selected[left]['acceptance_paths'] and selected[right]['acceptance_paths']:
                                    result['acceptance_paths']=selected[left]['acceptance_paths']==selected[right]['acceptance_paths']
                                result['target_calls']=selected[left]['model_calls'].get('target',0)==selected[right]['model_calls'].get('target',0)
                                assert all(v for k,v in result.items() if k not in ['question_id','method','baseline']),result
                                checks.append(result)
                    print(json.dumps(dict(phase=self.phase,task=self.task,shard=self.shard,ordinal=sample['ordinal'],group_ordinal=sample['group_ordinal'],records=count,seconds=time.perf_counter()-started)),flush=True)
            if checks:write(target.with_suffix('.checks.json'),dict(status='pass',checks=checks))
            self.verify_checkout()
            write(target.with_suffix('.complete.json'),dict(complete=True,task=self.task,phase=self.phase,records=count,questions=len(samples),result_sha256=digest(target),metadata=self.meta,seconds_excluding_load_warmup=time.perf_counter()-started))
        self.guard.finish()

    def main(self):
        try:
            self.run()
        finally:
            self.compact = False
            self.restore()
            for handle in self.counter_handles:
                handle.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['parity', 'development', 'main'], required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    Runtime(args.phase, args.shard).main()
