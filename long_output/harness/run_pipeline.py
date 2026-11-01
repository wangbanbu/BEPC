"""Durable sequential execution of the frozen long-output comparison."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
LOGS=ROOT/'logs'
STATUS=ROOT/'RUN_STATUS.json'
state=dict(status='starting',controller_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
           max_new_tokens=512,development_requests=3240,main_requests=12600,stages=[])

def save(**fields):
 state.update(fields,updated_utc=datetime.now(timezone.utc).isoformat())
 temporary=STATUS.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temporary.replace(STATUS)

def run_cpu(label,arguments):
 save(status='running',stage=label,active_gpu_children=[])
 log=LOGS/(label+'.log');assert not log.exists(),log
 started=time.time()
 with log.open('w',encoding='utf-8') as handle:
  result=subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=os.environ.copy())
 state['stages'].append(dict(stage=label,exit_code=result.returncode,elapsed_seconds=time.time()-started,log=str(log)))
 save()
 if result.returncode:raise RuntimeError(label+' failed; see '+str(log))

def run_gpu(phase):
 save(status='running',stage=phase)
 children=[];started=time.time()
 for shard in (0,1):
  path=LOGS/f'{phase}_shard{shard}.log';assert not path.exists(),path
  handle=path.open('w',encoding='utf-8')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
  process=subprocess.Popen([sys.executable,'harness/run_benchmark.py','--phase',phase,'--shard',str(shard)],cwd=ROOT,
      stdout=handle,stderr=subprocess.STDOUT,env=env)
  children.append((process,handle,path,shard))
 save(active_gpu_children=[dict(pid=p.pid,shard=shard,log=str(log)) for p,_,log,shard in children])
 while any(p.poll() is None for p,_,_,_ in children):
  save(stage_elapsed_seconds=time.time()-started,
       active_gpu_children=[dict(pid=p.pid,shard=shard,exit_code=p.poll(),log=str(log)) for p,_,log,shard in children])
  time.sleep(10)
 failures=[]
 for process,handle,log,shard in children:
  code=process.wait();handle.close()
  state['stages'].append(dict(stage=phase,shard=shard,exit_code=code,log=str(log)))
  if code:failures.append(dict(shard=shard,exit_code=code,log=str(log)))
 save(active_gpu_children=[],stage_elapsed_seconds=time.time()-started)
 if failures:raise RuntimeError('GPU failures: '+json.dumps(failures))

def main():
 LOGS.mkdir(parents=True,exist_ok=True)
 assert not STATUS.exists(),'Preserve existing pipeline evidence'
 save()
 try:
  run_gpu('parity')
  run_cpu('parity_gate',['harness/prepare_experiment.py','--stage','parity-gate'])
  run_cpu('parity_validation',['harness/validate_long.py','--phase','parity','--output','audit/PARITY_VALIDATION.json'])
  run_gpu('development')
  run_cpu('development_selection',['harness/select_lvspec.py'])
  run_cpu('main_freeze',['harness/prepare_experiment.py','--stage','freeze'])
  run_gpu('main')
  run_cpu('final_analysis',['harness/analyze_results.py','--stage','all'])
  run_cpu('final_validation',['harness/validate_long.py','--phase','all'])
  save(status='completed',stage='completed',completed_utc=datetime.now(timezone.utc).isoformat())
 except BaseException as error:
  save(status='failed',error=repr(error),traceback=traceback.format_exc())
  raise

if __name__=='__main__':main()
