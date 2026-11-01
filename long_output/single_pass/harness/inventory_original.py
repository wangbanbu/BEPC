"""Read-only, label-blind inventory of the stopped three-repeat long run."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def inventory():
    now=datetime.now(timezone.utc).isoformat()
    manifest={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for folder in ('harness','configs','results','audit','data','logs') for p in sorted((OLD/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    for p in [OLD/'EXPERIMENT_PROTOCOL.md',OLD/'RUN_STATUS.json']:
        manifest[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    cfg=json.loads((OLD/'configs/design_config.json').read_text())
    files=[]; repeats=Counter(); seconds=Counter(); reusable=[]; failures=[]; malformed=[]; seen=set()
    for p in sorted((OLD/'results').rglob('*.jsonl')):
        counts=Counter(); times=Counter(); coverage=defaultdict(list); last_offset=0
        for line_number,line in enumerate(p.read_bytes().splitlines(keepends=True),1):
            offset=last_offset;last_offset+=len(line)
            if not line.strip():continue
            try:r=json.loads(line)
            except (UnicodeDecodeError,json.JSONDecodeError) as e:
                malformed.append(dict(path=str(p),line=line_number,byte_offset=offset,bytes=len(line),sha256=hashlib.sha256(line).hexdigest(),trailing=last_offset==p.stat().st_size,error=str(e)))
                continue
            key=(r['phase'],r['task'],r['shard'],r['ordinal'],r['method'],r['repeat'])
            if key in seen:raise AssertionError('duplicate original row '+str(key))
            seen.add(key)
            k=f"{r['phase']}|repeat{r['repeat']}";repeats[k]+=1;seconds[k]+=r.get('seconds',0.)
            ck=f"{r['method']}|repeat{r['repeat']}";counts[ck]+=1;times[ck]+=r.get('seconds',0.);coverage[ck].append(r['ordinal'])
            if r.get('error') or r.get('status') in ('failed','error'):failures.append(dict(path=str(p),line=line_number,key=list(key),error=r.get('error')))
            if r['phase']=='development' and r['repeat']==0:
                reusable.append(dict(task=r['task'],shard=r['shard'],ordinal=r['ordinal'],method=r['method'],repeat=0,question_id=r['question_id'],image_id=r['image_id'],source_path=str(p),source_line=line_number,source_byte_offset=offset,source_line_sha256=hashlib.sha256(line).hexdigest(),source_row_sha256=hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),source_config_sha256=r['config_sha256']))
        files.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,complete_marker_present=p.with_suffix('.complete.json').exists(),rows=sum(counts.values()),counts=dict(counts),model_request_seconds=dict(times),ordinal_coverage=dict(coverage)))
    warmups=[]
    for p in sorted((OLD/'audit').glob('*.warmup.json')):
        warmups.append(dict(path=str(p),sha256=sha(p),**json.loads(p.read_text())))
    pids={}
    for pid in [3113887,3114828,3114829]:
        p=Path('/proc')/str(pid)
        pids[str(pid)]=dict(proc_exists=p.exists(),observation_utc=now)
        if p.exists():pids[str(pid)]['stat']=p.joinpath('stat').read_text()
    expected={}
    for task in cfg['tasks']:
        samples=json.loads(Path(cfg['data'][task]['dev_path']).read_text())
        have={(r['ordinal'],r['method']) for r in reusable if r['task']==task}
        missing=[dict(ordinal=s['ordinal'],question_id=s['question_id'],method=m,shard=s['group_ordinal']%2) for s in samples for m in cfg['development_methods'] if (s['ordinal'],m) not in have]
        expected[task]=dict(expected_repeat0=len(samples)*len(cfg['development_methods']),reusable_repeat0=len(have),missing_repeat0=len(missing),missing=missing)
    result=dict(created_utc=now,original_root=str(OLD),superseded_run_status=json.loads((OLD/'RUN_STATUS.json').read_text()),stop_evidence=dict(original_pids=pids,user_requested_repetitions=1,actual_SIGTERM_action_evidence='root session action and superseded RUN_STATUS; this inventory independently observes PID absence'),selection_rule='Reuse every complete valid original development repeat 0 row by identity; no gold, score, output-length or quality based selection. Preserve all repeat 1/2 and partial bytes unchanged.',original_manifest=manifest,result_files=files,record_counts_by_phase_and_repeat=dict(repeats),model_request_seconds_by_phase_and_repeat=dict(seconds),warmups=warmups,warmup_requests=sum(x['requests'] for x in warmups),warmup_seconds=sum(x['seconds'] for x in warmups),development_repeat0_coverage=expected,reusable_rows=reusable,successful_reusable_repeat0=len(reusable),failed_rows=failures,partial_or_malformed_records=malformed,main_records=sum(v for k,v in repeats.items() if k.startswith('main|')),gold_contents_read=False,source_files_modified=False)
    need=ROOT/'audit/ORIGINAL_RUN_INVENTORY.json';need.parent.mkdir(parents=True,exist_ok=True)
    assert not need.exists(),'preserve existing inventory'
    need.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    md=['# 原三次重复运行库存','',f'UTC observation: `{now}`。只读原运行；未读取 gold 内容；不以质量筛选。','',f'可复用开发集 repeat 0：**{len(reusable)} / 1080**；原 main：**{result["main_records"]}**。', '', '| Task | Required repeat 0 | Reusable | Missing |','|---|---:|---:|---:|']
    md += [f'| {task} | {r["expected_repeat0"]} | {r["reusable_repeat0"]} | {r["missing_repeat0"]} |' for task,r in expected.items()]
    md += ['', '重复 1/2、预热和所有原始文件完整保留；不计入新的单次主比较。', '',f'原预热 {result["warmup_requests"]} 次，{result["warmup_seconds"]:.3f} 秒。解析异常或部分行 {len(malformed)} 条。', '', '源文件 SHA、行偏移、行 SHA、候选复用映射、各方法覆盖和各 repeat 时间见 JSON。停机 PID 独立观察见 stop_evidence；SIGTERM 的操作证据由 root 会话与旧 RUN_STATUS 保留。']
    need.with_suffix('.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:result[k] for k in ('successful_reusable_repeat0','record_counts_by_phase_and_repeat','model_request_seconds_by_phase_and_repeat','warmup_requests','warmup_seconds','main_records')}))

if __name__=='__main__':inventory()
