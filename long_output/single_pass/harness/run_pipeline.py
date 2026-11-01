"""Durable single-pass generation, scoring, statistical reporting and audit."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
LOGS=ROOT/'logs';STATUS=ROOT/'RUN_STATUS.json'
state=dict(status='starting',controller_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
 repetitions=1,max_new_tokens=512,development_requests=1080,main_requests=4200,
 reused_development_repeat0=1003,new_development_requests=77,stages=[])
def save(**fields):
 state.update(fields,updated_utc=datetime.now(timezone.utc).isoformat())
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');p.replace(STATUS)
def cpu(label,args):
 save(stage=label,status='running',active_gpu_children=[])
 log=LOGS/(label+'.log');assert not log.exists(),log
 started=time.time()
 with log.open('w',encoding='utf-8') as handle:
  process=subprocess.run([sys.executable,*args],cwd=ROOT,env=os.environ.copy(),stdout=handle,stderr=subprocess.STDOUT)
 state['stages'].append(dict(stage=label,exit_code=process.returncode,seconds=time.time()-started,log=str(log)));save()
 if process.returncode:raise RuntimeError(label+' failed: '+str(log))
def gpu(phase):
 save(stage=phase,status='running');children=[];started=time.time()
 for shard in (0,1):
  log=LOGS/f'{phase}_shard{shard}.log';assert not log.exists(),log
  handle=log.open('w',encoding='utf-8')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
  process=subprocess.Popen([sys.executable,'harness/run_single_pass.py','--phase',phase,'--shard',str(shard)],cwd=ROOT,
    env=env,stdout=handle,stderr=subprocess.STDOUT)
  children.append((process,handle,log,shard))
 while any(p.poll() is None for p,_,_,_ in children):
  save(stage_elapsed_seconds=time.time()-started,active_gpu_children=[dict(pid=p.pid,shard=s,exit_code=p.poll(),log=str(log)) for p,_,log,s in children]);time.sleep(10)
 failures=[]
 for p,handle,log,shard in children:
  code=p.wait();handle.close();state['stages'].append(dict(stage=phase,shard=shard,exit_code=code,log=str(log)))
  if code:failures.append(dict(shard=shard,exit_code=code,log=str(log)))
 save(active_gpu_children=[],stage_elapsed_seconds=time.time()-started)
 if failures:raise RuntimeError(json.dumps(failures))
def main():
 assert not STATUS.exists(),'Preserve existing controller state'
 preflight=ROOT/'audit/PREFLIGHT_SPEC_VALIDATION.json'
 assert json.loads(preflight.read_text())['status']=='pass','Spec environment preflight must pass'
 LOGS.mkdir(parents=True,exist_ok=True);save()
 try:
  gpu('development')
  cpu('development_selection',['harness/select_lvspec.py'])
  cpu('main_freeze',['harness/prepare_single.py','--stage','freeze'])
  gpu('main')
  cpu('final_analysis',['harness/analyze_results.py','--stage','all'])
  cpu('final_validation',['harness/validate_single.py','--phase','all'])
  cpu('full_report',['harness/compile_full_report.py'])
  save(status='completed',stage='completed',completed_utc=datetime.now(timezone.utc).isoformat(),report=str(ROOT/'reports/FULL_EXPERIMENT_REPORT.md'))
 except BaseException as error:
  save(status='failed',error=repr(error),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
