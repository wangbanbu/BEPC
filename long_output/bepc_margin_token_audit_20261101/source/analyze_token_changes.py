"""Read-only paired token audit; never modifies or rescores production outputs."""
from __future__ import annotations
import base64
import difflib
import hashlib
import html
import io
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = '/data/wbb/model/Qwen2.5-VL-7B-Instruct'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    freeze = json.loads((ROOT/'audit/SOURCE_FREEZE.json').read_text())
    for f in freeze['files']:
        assert sha(f['path']) == f['sha256'], f['path']
    pairs = json.loads((ROOT/'data/paired_records.json').read_text())
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    rows, spans, ledger, browser = [], [], [], []
    seen = set()
    for pair in pairs:
        task, qid = pair['task'], pair['question_id']
        assert (task, qid) not in seen
        seen.add((task, qid))
        b, m = pair['bepc'], pair['margin']
        for key in ['prompt_text', 'prompt_token_ids', 'image_sha256',
                    'input_tensor_sha256', 'image_grid_thw', 'config_sha256']:
            assert b[key] == m[key], (task, qid, key)
        ledgers = {}
        for method, r in [('bepc', b), ('margin', m)]:
            flat = []
            lg = []
            for path in r['acceptance_paths']:
                assert len(path['tokens']) == len(path['nodes']) == path['accepted_draft']+1
                for j, (t, node) in enumerate(zip(path['tokens'], path['nodes'])):
                    entry = dict(task=task, question_id=qid, image_id=str(pair['image_id']),
                                 method=method, raw_token_index=len(flat), token_id=t,
                                 token_piece=dec([t]), round=path['round'], node_id=node,
                                 block_offset=j, role='target_root' if j == 0 else 'draft_node',
                                 returned_visible=len(flat)<len(r['returned_token_ids']))
                    flat.append(t)
                    lg.append(entry)
            assert flat == r['raw_token_ids'], (task, qid, method, 'path flatten')
            assert flat[:len(r['returned_token_ids'])] == r['returned_token_ids']
            assert tok.decode(r['returned_token_ids'], skip_special_tokens=True).strip() == r['output_text']
            ledgers[method] = lg
            ledger.extend(lg)
        bt, mt = b['returned_token_ids'], m['returned_token_ids']
        lcp = 0
        while lcp < min(len(bt),len(mt)) and bt[lcp] == mt[lcp]:
            lcp += 1
        token_changed = bt != mt
        text_changed = b['output_text'] != m['output_text']
        first_path = None
        for i in range(max(len(b['acceptance_paths']),len(m['acceptance_paths']))):
            bp = b['acceptance_paths'][i] if i<len(b['acceptance_paths']) else None
            mp = m['acceptance_paths'][i] if i<len(m['acceptance_paths']) else None
            keys = ['row','nodes','tokens','accepted_draft']
            if bp is None or mp is None or any(bp[k]!=mp[k] for k in keys):
                first_path = i
                break
        bs, ms = pair['bepc_score'], pair['margin_score']
        qa = task != 'coco'
        score_b, score_m = (bs.get('score'), ms.get('score')) if qa else (None, None)
        score_delta = float(score_b)-float(score_m) if qa else None
        row = dict(task=task, question_id=qid, image_id=str(pair['image_id']),
                   image_path=pair['image_path'], output_text_changed=text_changed,
                   output_tokens_changed=token_changed, common_token_prefix=lcp,
                   first_divergent_token_index=lcp if token_changed else None,
                   bepc_first_piece=dec(bt[lcp:lcp+1]) if token_changed else None,
                   margin_first_piece=dec(mt[lcp:lcp+1]) if token_changed else None,
                   common_text_tail=dec(bt[max(0,lcp-30):lcp]),
                   first_selected_path_divergence_round=first_path,
                   bepc_tokens=len(bt), margin_tokens=len(mt),
                   bepc_answer=bs.get('extracted_answer'), margin_answer=ms.get('extracted_answer'),
                   surface_answer_changed=(bs.get('extracted_answer')!=ms.get('extracted_answer')) if qa else None,
                   normalized_answer_changed=(bs.get('parsed_answer')!=ms.get('parsed_answer')) if qa else None,
                   bepc_score=score_b, margin_score=score_m, score_delta=score_delta,
                   bepc_cap_reached=b['cap_reached'], margin_cap_reached=m['cap_reached'],
                   bepc_eos_index=b['eos_index'], margin_eos_index=m['eos_index'])
        for method, ts in [('bepc',bt),('margin',mt)]:
            lg = ledgers[method]
            entry = lg[lcp] if token_changed and lcp<len(ts) else {}
            row[method+'_first_token_role'] = entry.get('role')
            row[method+'_first_token_round'] = entry.get('round')
            if qa and token_changed:
                text = dec(ts)
                final = text.lower().rfind('final answer:')
                row[method+'_first_change_in_final_answer_line'] = final>=0 and len(dec(ts[:lcp]))>=final
        if not qa:
            for metric in ['CIDEr','SPICE','CHAIRs','CHAIRi']:
                row['bepc_'+metric], row['margin_'+metric] = bs.get(metric), ms.get(metric)
        chunks = {'bepc':[], 'margin':[]}
        opcodes = difflib.SequenceMatcher(a=mt,b=bt,autojunk=False).get_opcodes()
        changed_token_sum = 0
        for span_id, (tag,i,j,k,l) in enumerate(opcodes):
            for method, ts, start, end in [('margin',mt,i,j),('bepc',bt,k,l)]:
                text = html.escape(dec(ts[start:end]))
                chunks[method].append(text if tag=='equal' else '<mark>'+text+'</mark>')
            if tag == 'equal':
                continue
            changed_token_sum += max(j-i,l-k)
            spans.append(dict(task=task,question_id=qid,image_id=str(pair['image_id']),
                              span_id=span_id,operation=tag,margin_start=i,margin_end=j,
                              bepc_start=k,bepc_end=l,margin_text=dec(mt[i:j]),bepc_text=dec(bt[k:l]),
                              margin_token_ids=mt[i:j],bepc_token_ids=bt[k:l]))
        row['alignment_change_size'] = changed_token_sum
        rows.append(row)
        im = Image.open(pair['image_path']).convert('RGB')
        im.thumbnail((320,240))
        buff = io.BytesIO()
        im.save(buff,format='JPEG',quality=72)
        thumb='data:image/jpeg;base64,'+base64.b64encode(buff.getvalue()).decode()
        cat = ('gain' if score_delta>0 else 'loss' if score_delta<0 else 'tie') if qa else 'coco'
        browser.append(dict(task=task,qid=qid,changed=text_changed,category=cat,
                            title=f'{task} / {qid} | '+(f'official {score_m} → {score_b}' if qa else f"CHAIRi {ms.get('CHAIRi')} → {bs.get('CHAIRi')}"),
                            prompt=pair['prompt_text'],image=thumb,**chunks,
                            note=f'first token index={lcp if token_changed else None}; first selected-path fork round={first_path}'))
    pd.DataFrame(rows).to_parquet(ROOT/'data/token_changes.parquet',index=False)
    pd.DataFrame(rows).to_csv(ROOT/'data/token_changes.csv',index=False)
    pd.DataFrame(spans).to_parquet(ROOT/'data/token_change_spans.parquet',index=False)
    pd.DataFrame(ledger).to_parquet(ROOT/'data/token_ledger.parquet',index=False)
    summary = dict(status='complete',paired_questions=len(rows),same_inputs_verified=len(rows),
                   path_flatten_verified_requests=len(pairs)*2,
                   all_sources_unchanged=len(freeze['files']),
                   alignment='difflib.SequenceMatcher(autojunk=False), deterministic alignment, NOT minimum edit distance',
                   attribution='Output differences and chunk paths only; gate attribution requires same-prefix diagnostics.',
                   tasks={})
    for task in ['scienceqa','textvqa','coco']:
        rs=[r for r in rows if r['task']==task]
        summary['tasks'][task]=dict(n=len(rs),text_changed=sum(r['output_text_changed'] for r in rs),
           tokens_changed=sum(r['output_tokens_changed'] for r in rs),
           selection_path_changed=sum(r['first_selected_path_divergence_round'] is not None for r in rs),
           bepc_cap_reached=sum(r['bepc_cap_reached'] for r in rs),
           margin_cap_reached=sum(r['margin_cap_reached'] for r in rs))
        if task!='coco':
            summary['tasks'][task].update(surface_answer_changed=sum(r['surface_answer_changed'] for r in rs),
                normalized_answer_changed=sum(r['normalized_answer_changed'] for r in rs),
                improved=sum(r['score_delta']>0 for r in rs),degraded=sum(r['score_delta']<0 for r in rs),
                score_tie=sum(r['score_delta']==0 for r in rs),
                changed_answer_first_divergence_in_body=sum(bool(r['surface_answer_changed']) and not r.get('bepc_first_change_in_final_answer_line',False) and not r.get('margin_first_change_in_final_answer_line',False) for r in rs))
    (ROOT/'audit/TOKEN_ALIGNMENT_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    document = '''<!doctype html><meta charset="utf-8"><title>BEPC–Margin Token Audit</title>
<style>body{font:15px system-ui;margin:24px;background:#f4f6f9;color:#152238}header{position:sticky;top:0;background:#f4f6f9;padding:12px 0;border-bottom:1px solid #bbb}input,select{padding:8px;margin-right:8px}article{background:white;border:1px solid #ddd;border-radius:9px;padding:16px;margin:18px 0}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}.txt{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}mark{background:#ffdf91}img{max-height:200px;max-width:320px}details{margin:8px 0}.note{color:#536278}@media(max-width:800px){.cols{grid-template-columns:1fr}}</style>
<header><h2>BEPC 与 Margin：全部 300 题 token 对齐</h2><p>黄色为确定性 token-ID 对齐的差异。首次分叉后的共同措辞不代表共同因果前缀。浏览器不证明关键 token 保护。</p><input id="q" placeholder="题号/正文关键词"><select id="task"><option value="">全部任务</option><option>scienceqa</option><option>textvqa</option><option>coco</option></select><select id="cat"><option value="">全部评分关系</option><option value="gain">QA 改善</option><option value="loss">QA 退化</option><option value="tie">QA 同分</option><option value="coco">COCO</option></select><label><input type="checkbox" id="diff" checked>仅生成文本变化</label><span id="count"></span></header><main id="root"></main><script>
const DATA=__DATA__;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){let q=document.querySelector('#q').value.toLowerCase(),t=document.querySelector('#task').value,c=document.querySelector('#cat').value,d=document.querySelector('#diff').checked;let arr=DATA.filter(x=>(!t||x.task===t)&&(!c||x.category===c)&&(!d||x.changed)&&(!q||[x.title,x.bepc.join(''),x.margin.join('')].join(' ').toLowerCase().includes(q)));document.querySelector('#count').textContent=arr.length+' pairs';document.querySelector('#root').innerHTML=arr.map(x=>`<article id="${esc(x.task+'_'+x.qid)}"><h3>${esc(x.title)}</h3><img src="${x.image}"><details><summary>原始任务指令</summary><pre class="txt">${esc(x.prompt)}</pre></details><p class="note">${esc(x.note)}</p><div class="cols"><div><h4>Margin</h4><div class="txt">${x.margin.join('')}</div></div><div><h4>BEPC</h4><div class="txt">${x.bepc.join('')}</div></div></div></article>`).join('');}
for(let id of ['q','task','cat','diff'])document.getElementById(id).addEventListener('input',render);render();
</script>'''
    document=document.replace('__DATA__',json.dumps(browser,ensure_ascii=False).replace('</','<\\/'))
    (ROOT/'reports/TOKEN_DIFF_BROWSER.html').write_text(document)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
