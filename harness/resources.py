"""Record exclusive GPU use without changing model timing or allocations."""
import os,json,subprocess,threading,time
from pathlib import Path
class ResourceGuard:
    def __init__(self,label,root):
        self.label,self.root=label,Path(root);self.gpu=os.environ['CUDA_VISIBLE_DEVICES'];self.errors=[];self.checks=0;self.stop=threading.Event()
        self.check();self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
    def check(self):
        s=subprocess.check_output(['nvidia-smi','-i',self.gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        foreign=[int(x.strip()) for x in s.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()];self.checks+=1
        if foreign:self.errors.append(dict(time=time.time(),foreign_pids=foreign))
    def monitor(self):
        while not self.stop.wait(2):
            try:self.check()
            except Exception as ex:self.errors.append(dict(time=time.time(),error=repr(ex)))
    def assert_free(self):
        if self.errors:raise RuntimeError(f'GPU resource conflict: {self.errors}')
    def finish(self):
        self.stop.set();self.thread.join(timeout=5);self.check()
        p=self.root/'audit'/f'{self.label}.gpu_resource.json';assert not p.exists();p.write_text(json.dumps(dict(physical_gpu=self.gpu,pid=os.getpid(),checks=self.checks,events=self.errors),indent=2)+'\n')
        self.assert_free()
