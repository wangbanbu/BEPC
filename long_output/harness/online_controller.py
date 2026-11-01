"""Single-score online gates, with exact native Target/Draft/KV computation.

Every relaxed method uses the same continuous-prefix/longest-leaf rule and EOS
guard. Full future captions, task references and Harmful labels are unavailable.
"""
import math
import torch
from epcc_online import OnlineEPCC
from visual_online import BatchVisualScores


def js_divergence(z,q):
    assert z.shape==q.shape and z.ndim==1,'Shared aligned vocabulary required'
    p=z.float().log_softmax(-1);r=q.float().log_softmax(-1)
    mid=torch.logaddexp(p,r)-math.log(2.)
    return .5*((p.exp()*(p-mid)).sum()+(r.exp()*(r-mid)).sum())


class DraftDistributions:
    """Exact native beam indexing, captured pre-LogSoftmax logits on GPU."""
    def __init__(self,model):
        self.layer=model.spec_layer;self.original=self.layer.topK_genrate
        self.handle=self.layer.logsoftmax.register_forward_hook(self.observe)
        self.layer.topK_genrate=self.tree;self.paths=[];self.distributions={};self.active=False
    def tree(self,*args,**kwargs):
        ids=args[1] if len(args)>1 else kwargs['input_ids']
        self.paths=[(int(ids[0,-1]),)];self.beam=None;self.distributions={};self.active=True
        try:return self.original(*args,**kwargs)
        finally:self.active=False
    def observe(self,module,args,out):
        if not self.active:return
        assert len(self.paths)==len(args[0])
        for path,row in zip(self.paths,args[0]):self.distributions[path]=row.detach()
        k=self.layer.top_k;top=out.topk(k,dim=-1)
        if self.beam is None:
            self.paths=[self.paths[0]+(int(t),) for t in top.indices[0]];self.beam=top.values[0]
        else:
            scores=top.values+self.beam[:,None];best=scores.reshape(-1).topk(k)
            parents=(best.indices//k).tolist();tokens=top.indices.reshape(-1)[best.indices].tolist()
            self.paths=[self.paths[p]+(t,) for p,t in zip(parents,tokens)];self.beam=best.values
    def close(self):
        self.handle.remove();self.layer.topK_genrate=self.original;self.distributions={}


class Controller:
    FUNCTIONS=['initialize_tree','tree_decoding','evaluate_posterior','update_inference_inputs']
    def __init__(self,runtime,model,variant,prompt_ids,image_grid,*,collect=False,audit=False,max_new_tokens=128):
        self.runtime=runtime;self.model=model;self.variant=variant;self.family=variant['family'];self.collect=collect
        self.original={k:getattr(runtime,k) for k in self.FUNCTIONS};self.handles=[];self.states={};self.active=False
        self.eos=model.tokenizer.eos_token_id;self.prompt_ids=prompt_ids;self.grid=image_grid
        self.visible_boundary=prompt_ids.shape[-1]+int(max_new_tokens)
        self.records=[];self.stats=dict(rounds=0,unique_score_requests=0,missing_scores=0,accepted_mismatches=0,accepted_draft=0,contexts=0)
        self.epcc=OnlineEPCC(model).install() if collect or self.family=='EPCC' else None
        self.visual=BatchVisualScores(model.base_model,prompt_ids,with_ccve=collect or self.family=='CCVE',check_cache=audit) if collect or self.family in ['mass','CCVE'] else None
        self.draft=DraftDistributions(model) if collect or self.family=='T4' else None
        if collect or self.family=='P1':
            for layer in [14,28]:self.handles.append(model.base_model.model.layers[layer-1].register_forward_hook(self.hook(layer)))
        runtime.initialize_tree=self.prefill;runtime.tree_decoding=self.tree
        runtime.evaluate_posterior=self.evaluate;runtime.update_inference_inputs=self.update
    def hook(self,layer):
        def capture(module,args,out):
            if self.active:self.states[layer]=out[0].detach()
        return capture
    def prefill(self,*args,**kwargs):
        if self.epcc:self.epcc.begin_prefill(args[0],self.grid)
        result=self.original['initialize_tree'](*args,**kwargs)
        if self.epcc:self.epcc.finish_prefill()
        return result
    def tree(self,model,tokens,cache,positions,ids,retrieve):
        self.tokens=tokens;self.ids=ids;self.retrieve=retrieve;self.score_cache={};self.states={}
        if self.epcc:self.epcc.begin_tree(tokens,ids,retrieve,model.base_model.model.tree_mask,positions)
        if self.visual:self.visual.enable(True)
        self.active=True
        try:result=self.original['tree_decoding'](model,tokens,cache,positions,ids,retrieve)
        finally:
            self.active=False
            if self.visual:self.visual.enable(False)
        if self.epcc:self.epcc.finish_tree()
        self.stats['rounds']+=1
        return result
    def score(self,family,row,column,logits,candidates):
        parent=int(self.retrieve[row,column-1]);a=int(self.targets[row][column-1]);d=self.candidate_rows[row][column]
        key=(family,parent,a,d if family not in ['T4','mass'] else -1)
        if key in self.score_cache:return self.score_cache[key]
        if family=='T2':value=logits[row,column-1,a].float()-logits[row,column-1,d].float()
        elif family=='P1':
            head=self.model.base_model.lm_head.weight;norm=self.model.base_model.model.norm
            m=[]
            for layer in [14,28]:
                h=self.states[layer][0,parent].float();y=h*torch.rsqrt(h.square().mean()+norm.variance_epsilon)*norm.weight.float();m.append((head[a].float()*y).sum()-(head[d].float()*y).sum())
            value=m[1]-m[0]
        elif family=='T4':
            path=tuple(self.candidate_rows[row][:column]);q=self.draft.distributions.get(path)
            if q is None:value=logits.new_tensor(float('nan'),dtype=torch.float32)
            else:value=js_divergence(logits[row,column-1],q)
        elif family in ['mass','CCVE']:
            if not self.visual.prepared:self.visual.prepare()
            device=logits.device
            result=self.visual.scores(torch.tensor([parent],device=device),torch.tensor([a],device=device),torch.tensor([d],device=device))
            value=result['visual_attention_mass' if family=='mass' else 'CCVE'][0]
        elif family=='EPCC':
            if parent not in self.epcc.contexts:self.stats['contexts']+=1
            value=self.epcc.score_rows(row,column,a,d)
        else:raise ValueError(family)
        number=float(value);self.score_cache[key]=number;self.stats['unique_score_requests']+=1
        if not math.isfinite(number):self.stats['missing_scores']+=1
        return number
    def evaluate(self,logits,candidates,processor):
        assert processor is None
        self.candidate_rows=candidates.tolist();self.targets=logits[:,:-1].argmax(-1).tolist()
        if self.collect:
            native=self.original['evaluate_posterior'](logits,candidates,processor);b,L=int(native[0]),int(native[1]);col=L+1
            if col<candidates.shape[1]:
                d=self.candidate_rows[b][col];a=self.targets[b][col-1]
                prefix=self.candidate_rows[b][:col]
                eos_before=self.eos in prefix
                if d>=0 and a!=d and a!=self.eos and d!=self.eos and not eos_before and self.ids.shape[1]+col<self.visible_boundary:
                    record=dict(round=self.stats['rounds']-1,parent_node=int(self.retrieve[b,col-1]),candidate_node=int(self.retrieve[b,col]),target_id=a,draft_id=d,parent_absolute_position=self.ids.shape[1]+col-1,candidate_absolute_position=self.ids.shape[1]+col,prefix_length=self.ids.shape[1]+col)
                    for family in ['T2','P1','T4','mass','CCVE','EPCC']:
                        value=self.score(family,b,col,logits,candidates);record[family]=value if math.isfinite(value) else None
                    self.records.append(record)
            return native
        width=candidates.shape[1]-1;allowed=[];lengths=[]
        for row,seq in enumerate(self.candidate_rows):
            flags=[False]*width;length=0;eos_seen=seq[0]==self.eos
            for j in range(width):
                d=seq[j+1];a=self.targets[row][j]
                if d<0:break
                if d==a:accept=True
                elif eos_seen or self.ids.shape[1]+j+1>=self.visible_boundary:accept=False
                elif a==self.eos or d==self.eos:accept=False
                else:
                    value=self.score(self.family,row,j+1,logits,candidates)
                    accept=math.isfinite(value) and self.variant.get('threshold') is not None and value<=self.variant['threshold']
                flags[j]=accept
                if not accept:break
                length+=1
                if d==self.eos:eos_seen=True
            allowed.append(flags);lengths.append(length)
        best=max(range(len(lengths)),key=lengths.__getitem__);L=lengths[best]
        self.stats['accepted_mismatches']+=sum(self.candidate_rows[best][j+1]!=self.targets[best][j] for j in range(L))
        self.stats['accepted_draft']+=L
        return torch.tensor(best,device=logits.device),torch.tensor(L,device=logits.device),logits[best,L]
    def update(self,*args,**kwargs):
        old_ids,candidates,best,L,retrieve=args[:5]
        result=self.original['update_inference_inputs'](*args,**kwargs)
        if self.epcc:self.epcc.commit(best,L,retrieve,candidates,new_input_ids=result[0])
        return result
    def restore(self):
        for key,value in self.original.items():setattr(self.runtime,key,value)
        for h in self.handles:h.remove()
        if self.epcc:self.stats['contexts']=getattr(self.epcc,'constructed_contexts',self.stats['contexts']);self.epcc.restore()
        if self.visual:self.visual.close()
        if self.draft:self.draft.close()
        self.states={}
