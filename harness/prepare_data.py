"""Independent local official-data subsets; no inference or quality-based selection.

Historical files are inspected only for identity fields and generation markers.
Runtime records contain no gold labels; official labels are stored separately.
"""
from __future__ import annotations
import argparse
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zlib
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
LOCAL=Path('/data/wbb/data')
UPSTREAM=ROOT/'upstream/ViSpec'
HISTORY=[Path('/data/wbb/vispec-official-reproduction'),Path('/data/wbb/ViSpec')]
TASKS=['mme','gqa','textvqa','scienceqa','coco']
DEFAULT_SEED=2026103109
SYSTEM='You are a helpful assistant.'
GENERATION=re.compile(r'"(?:model_id|answer_id|raw_token_ids|returned_token_ids|output_text|generated_text|generated_token_ids|event_id|method|response|prediction)"\s*:')
IDS=re.compile(r'"(question_id|questionId|qid|pid|image_id|imageId|cocoid)"\s*:\s*(?:"([^"\\]*)"|(-?\d+))')
PATHS=re.compile(r'"(?:image_path|image)"\s*:\s*"([^"\\]+\.(?:jpg|jpeg|png))"',re.I)
HASHES=re.compile(r'"(image_sha256|image_rgb_sha256)"\s*:\s*"([0-9a-f]{64})"')
CATEGORIES=['artwork','celebrity','code_reasoning','color','commonsense_reasoning','count','existence','landmark','numerical_calculation','OCR','position','posters','scene','text_translation']
MMEPATH=re.compile(r'('+'|'.join(CATEGORIES)+r')/(?:images/)?([^"\s/]+\.(?:jpg|jpeg|png))',re.I)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def dump(p,value,preserve=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if preserve:assert not p.exists(),f'Preserve prepared data: {p}'
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def canonical(value):
 s=str(value).strip();return str(int(s)) if re.fullmatch(r'\d+',s) else s.lower()
def order(seed,*parts):return hashlib.sha256(':'.join(map(str,[seed,*parts])).encode()).hexdigest()
def task_hint(text):
 x=text.lower()
 for task,patterns in [('textvqa',['textvqa']),('scienceqa',['scienceqa','/sqa/','_sqa','sqa_']),('gqa',['gqa']),('mme',['mme']),('coco',['coco','caption','visual-margin'])]:
  if any(p in x for p in patterns):return task
 return None


def scan_history():
 used={t:dict(questions=set(),images=set()) for t in TASKS}
 unknown=dict(questions=set(),images=set())
 hashes=set();rgb=set();paths=set();sources=[]
 for base in HISTORY:
  for suffix in ['*.jsonl','*.json']:
   for path in base.rglob(suffix):
    relative=path.relative_to(base)
    if any(p in relative.parts for p in ['.git','cache','reference','upstream','site-packages','node_modules']):continue
    # Prepared HM/benchmark input subsets are conservatively excluded too,
    # but never exclude complete official corpus metadata just for existing.
    prepared=('inputs' in relative.parts and 'harmful_mismatch' in relative.parts)
    if suffix=='*.json' and not prepared and not any(p.lower().startswith(('result','output')) for p in relative.parts):continue
    json_generation=False
    if suffix=='*.json' and not prepared:
     try:
      with path.open(encoding='utf-8-sig') as prefix:json_generation=bool(GENERATION.search(prefix.read(256*1024)))
     except (UnicodeError,OSError):continue
     if not json_generation:continue
    found={t:dict(questions=set(),images=set()) for t in TASKS};unknown_found=dict(questions=set(),images=set())
    eligible_rows=0
    try:
     with path.open(encoding='utf-8-sig') as f:
      for line in f:
       if not prepared and not json_generation and not GENERATION.search(line):continue
       values=IDS.findall(line)
       if not values and not PATHS.search(line):continue
       eligible_rows+=1
       hint=task_hint(str(relative))
       task_match=re.search(r'"(?:task|dataset|task_name)"\s*:\s*"([^"\\]+)"',line)
       if task_match:hint=task_hint(task_match.group(1)) or hint
       dest=found[hint] if hint else unknown_found
       for key,quoted,number in values:
        value=quoted if quoted else number
        dest['questions' if key in ['question_id','questionId','qid','pid'] else 'images'].add(canonical(value))
       for image in re.findall(r'COCO_(?:val|train)2014_(\d+)',line):found['coco']['images'].add(canonical(image))
       for category,name in MMEPATH.findall(line):
        exact=next(c for c in CATEGORIES if c.lower()==category.lower())
        found['mme']['images'].add((exact+'/'+name).lower())
       for key,value in HASHES.findall(line):(rgb if key=='image_rgb_sha256' else hashes).add(value)
       paths.update(PATHS.findall(line))
    except (UnicodeError,OSError):continue
    count=sum(len(v[k]) for v in found.values() for k in ['questions','images'])+sum(len(v) for v in unknown_found.values())
    if not count:continue
    for t in TASKS:
     for key in ['questions','images']:used[t][key].update(found[t][key])
    for key in ['questions','images']:unknown[key].update(unknown_found[key])
    sources.append(dict(path=str(path),sha256=sha(path),identity_rows=eligible_rows,
        prepared_input_conservatively_excluded=prepared,
        task_counts={t:{key:len(v[key]) for key in v} for t,v in found.items() if any(v.values())},
        unknown_task_counts={key:len(v) for key,v in unknown_found.items()}))
 # Ambiguous identity namespaces are conservatively matched to each official
 # candidate universe; this may exclude extra items, never supplies labels.
 for t in TASKS:
  for key in ['questions','images']:used[t][key].update(unknown[key])
 serial={t:{key:sorted(v) for key,v in values.items()} for t,values in used.items()}
 dump(ROOT/'audit/DATA_PRIOR_IDENTITIES.json',dict(roots=list(map(str,HISTORY)),used=serial,
      image_sha256=sorted(hashes),image_rgb_sha256=sorted(rgb),image_paths=sorted(paths),sources=sources,
      selection_uses_quality_or_generated_caption_values=False,ambiguous_ids_conservatively_matched=True))
 return used,hashes,rgb,sources


def scan_history_fast():
 """Native regex identity extraction; file-level eligibility is conservative.
 Compared with row eligibility, generation-bearing JSONL files may contribute
 additional identity-only metadata rows. This can exclude extra candidates.
 """
 command=['rg','-l','--null','--glob','*.jsonl','--glob','*.json',GENERATION.pattern,*map(str,HISTORY)]
 result=subprocess.run(command,check=False,capture_output=True)
 assert result.returncode in [0,1],result.stderr.decode()[:1000]
 candidates={Path(v.decode()) for v in result.stdout.split(b'\0') if v}
 for base in HISTORY:
  candidates.update(base.glob('research/harmful_mismatch/**/inputs/*.json'))
 eligible=[]
 for path in sorted(candidates):
  base=next(base for base in HISTORY if path.is_relative_to(base));relative=path.relative_to(base)
  if any(p in relative.parts for p in ['.git','cache','reference','upstream','site-packages','node_modules']):continue
  prepared='inputs' in relative.parts and 'harmful_mismatch' in relative.parts
  if path.suffix=='.json' and not prepared and not any(p.lower().startswith(('result','output')) for p in relative.parts):continue
  eligible.append(path)
 print(json.dumps({'fast_identity_files':len(eligible)}),flush=True)
 used={t:dict(questions=set(),images=set()) for t in TASKS};unknown=dict(questions=set(),images=set());hashes=set();rgb=set();imagepaths=set();counts=Counter()
 expression='(?:'+IDS.pattern+')|(?:'+PATHS.pattern+')|(?:'+HASHES.pattern+')|(?:'+MMEPATH.pattern+')|COCO_(?:val|train)2014_[0-9]+'
 for begin in range(0,len(eligible),100):
  batch=eligible[begin:begin+100]
  result=subprocess.run(['rg','--no-heading','--with-filename','--only-matching','-e',expression,*map(str,batch)],capture_output=True,check=False)
  assert result.returncode in [0,1],result.stderr.decode()[:1000]
  for line in result.stdout.decode(errors='replace').splitlines():
   filename,match=line.split(':',1);path=Path(filename);base=next(base for base in HISTORY if path.is_relative_to(base));hint=task_hint(str(path.relative_to(base)));dest=used[hint] if hint else unknown
   for key,quoted,number in IDS.findall(match):dest['questions' if key in ['question_id','questionId','qid','pid'] else 'images'].add(canonical(quoted or number))
   for ident in re.findall(r'COCO_(?:val|train)2014_(\d+)',match):used['coco']['images'].add(canonical(ident))
   for category,name in MMEPATH.findall(match):used['mme']['images'].add((category+'/'+name).lower())
   for key,value in HASHES.findall(match):(rgb if key=='image_rgb_sha256' else hashes).add(value)
   imagepaths.update(PATHS.findall(match));counts[filename]+=1
 for task in TASKS:
  for key in ['questions','images']:used[task][key].update(unknown[key])
 for key in ['questions','images']:used['mme'][key].update(re.sub(r'/images/','/',v) for v in list(used['mme'][key]))
 def source_info(path):return dict(path=str(path),sha256=sha(path),identity_matches=counts[str(path)])
 with ThreadPoolExecutor(max_workers=8) as executor:sources=list(executor.map(source_info,[p for p in eligible if counts[str(p)]]))
 dump(ROOT/'audit/DATA_PRIOR_IDENTITIES_FAST.json',dict(roots=list(map(str,HISTORY)),used={t:{k:sorted(v) for k,v in row.items()} for t,row in used.items()},image_sha256=sorted(hashes),image_rgb_sha256=sorted(rgb),image_paths=sorted(imagepaths),sources=sources,selection_uses_quality_or_generated_caption_values=False,ambiguous_ids_conservatively_matched=True,file_level_generation_eligibility_conservative=True))
 return used,hashes,rgb,sources


def load_prior(refresh=False):
 path=ROOT/'audit/DATA_PRIOR_IDENTITIES_FAST.json'
 if refresh or not path.exists():return scan_history_fast()
 d=read(path);used={('coco' if t=='coco_caption' else t):{key:set(v) for key,v in row.items()} for t,row in d['used'].items()}
 for key in ['questions','images']:
  used['mme'][key].update(re.sub(r'/images/','/',v) for v in list(used['mme'][key]))
 for value in d.get('image_paths',[]):
  for category,name in MMEPATH.findall(value):
   used['mme']['images'].add((category+'/'+name).lower())
 return used,set(d['image_sha256']),set(d['image_rgb_sha256']),d['sources']


def supplement_parquet_identities(used,hashes,rgb):
 sources=[]
 for base in HISTORY:
  for path in base.rglob('*.parquet'):
   relative=path.relative_to(base)
   if any(v in relative.parts for v in ['reference','upstream','cache','.git']):continue
   names=pq.read_schema(path).names
   columns=[v for v in names if v in ['question_id','qid','pid','image_id','cocoid','image_sha256','image_rgb_sha256']]
   if not columns:continue
   hint=task_hint(str(relative));before=sum(len(x[k]) for x in used.values() for k in ['questions','images'])
   for row in pq.read_table(path,columns=columns).to_pylist():
    for key,value in row.items():
     if value is None:continue
     if key in ['image_sha256','image_rgb_sha256']:(rgb if key=='image_rgb_sha256' else hashes).add(str(value));continue
     for task in [hint] if hint else TASKS:used[task]['questions' if key in ['question_id','qid','pid'] else 'images'].add(canonical(value))
   sources.append(dict(path=str(path),sha256=sha(path),identity_columns=columns,additional_identity_entries=sum(len(x[k]) for x in used.values() for k in ['questions','images'])-before))
 dump(ROOT/'audit/DATA_PARQUET_IDENTITY_SUPPLEMENT.json',dict(sources=sources,only_identity_columns_read=True))
 return sources


def image_info_bytes(data):
 with Image.open(io.BytesIO(data)) as img:
  img=img.convert('RGB');return dict(image_sha256=hashlib.sha256(data).hexdigest(),image_rgb_sha256=hashlib.sha256(img.tobytes()).hexdigest(),image_size=list(img.size))
def image_info_path(path):return image_info_bytes(Path(path).read_bytes())


def zip_index(path):
 """Read complete local ZIP entries even if a local download lacks its directory.
 No bytes are rewritten. Selected entries later require inflate+size+CRC+PIL.
 """
 size=path.stat().st_size;records={};status='local_headers_complete'
 with path.open('rb') as f:
  while f.tell()+30<=size:
   start=f.tell();hdr=f.read(30)
   if hdr[:4] in [b'PK\x01\x02',b'PK\x05\x06']:break
   if hdr[:4]!=b'PK\x03\x04':status='stopped_at_nonlocal_header';break
   _,version,flags,method,mtime,mdate,crc,csize,usize,nlen,xlen=struct.unpack('<IHHHHHIIIHH',hdr)
   name=f.read(nlen).decode('utf-8');extra=f.read(xlen);offset=f.tell()
   if flags&8:status='unsupported_data_descriptor';break
   if offset+csize>size:status='truncated_last_member';break
   if not name.endswith('/'):
    assert name not in records;records[name]=dict(offset=offset,compressed_size=csize,uncompressed_size=usize,crc32=crc,method=method)
   f.seek(csize,1)
 return records,dict(path=str(path),bytes=size,complete_local_members=len(records),status=status,central_directory_required=False)

def zip_bytes(path,entry):
 if entry.get('remote'):
  url=entry['url'];offset=entry['header_offset']
  def ranged(start,end):
   response=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=60)
   response.raise_for_status();assert response.status_code==206 and response.headers['Content-Range'].startswith(f'bytes {start}-{end}/') and len(response.content)==end-start+1
   return response.content
  header=ranged(offset,offset+29);assert header[:4]==b'PK\x03\x04'
  values=struct.unpack('<IHHHHHIIIHH',header);assert values[6:9]==(entry['crc32'],entry['compressed_size'],entry['uncompressed_size'])
  start=offset+30+values[-2]+values[-1];compressed=ranged(start,start+entry['compressed_size']-1)
 else:
  with path.open('rb') as f:f.seek(entry['offset']);compressed=f.read(entry['compressed_size'])
 raw=compressed if entry['method']==0 else zlib.decompress(compressed,-15) if entry['method']==8 else None
 assert raw is not None and len(raw)==entry['uncompressed_size'] and zlib.crc32(raw)&0xffffffff==entry['crc32']
 image_info_bytes(raw)
 return raw


def is_used(task,qid,image_id,used,info=None,hashes=None,rgb=None):
 return (canonical(qid) in used[task]['questions'] or canonical(image_id) in used[task]['images'] or
         info is not None and (info['image_sha256'] in (hashes or set()) or info['image_rgb_sha256'] in (rgb or set())))


def gqa_pool(used):
 # Match prior questions from both labelled official evaluation splits to image
 # identities, then use val_balanced for this fresh experiment.
 for filename in ['testdev_balanced_questions.json','val_balanced_questions.json','test_balanced_questions.json']:
  payload=read(LOCAL/'GQA/questions'/filename)
  for qid,row in payload.items():
   if canonical(qid) in used['gqa']['questions']:used['gqa']['images'].add(canonical(row['imageId']))
 payload=read(LOCAL/'GQA/questions/val_balanced_questions.json');pool=[]
 for qid,row in payload.items():
  image_id=row['imageId'];path=LOCAL/'GQA/images'/f'{image_id}.jpg'
  if is_used('gqa',qid,image_id,used) or not path.exists():continue
  pool.append(dict(task='gqa',question_id=qid,image_id=image_id,question=row['question'],image_path=str(path),
     source=str(LOCAL/'GQA/questions/val_balanced_questions.json'),source_split='val_balanced',label=dict(answer=row['answer']),source_question_id=qid))
 return pool


def textvqa_pool(used):
 path=LOCAL/'textvqa/TextVQA_0.5.1_val.json';payload=read(path)
 assert payload['dataset_type']=='val' and payload['dataset_name']=='textvqa' and payload['dataset_version']=='0.5.1'
 rows=payload['data']
 for row in rows:
  if canonical(row['question_id']) in used['textvqa']['questions']:used['textvqa']['images'].add(canonical(row['image_id']))
 archive=LOCAL/'textvqa/train_val_images.zip';index,status=zip_index(archive);pool=[]
 remote_index_path=HISTORY[0]/'reference/textvqa-author-zip-index.json';remote_index=read(remote_index_path) if remote_index_path.exists() else {}
 status['official_remote_member_index']=str(remote_index_path);status['official_remote_index_sha256']=sha(remote_index_path) if remote_index else None
 for row in rows:
  assert len(row['answers'])==10
  qid,image_id=row['question_id'],row['image_id']
  if is_used('textvqa',qid,image_id,used):continue
  pathimage=LOCAL/'textvqa/train_images'/f'{image_id}.jpg';member=f'train_images/{image_id}.jpg'
  if not pathimage.exists() and member not in index and image_id not in remote_index:continue
  entry=index.get(member)
  if entry is None and image_id in remote_index:
   value=remote_index[image_id];entry=dict(remote=True,url='https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip',header_offset=value['offset'],compressed_size=value['compressed'],uncompressed_size=value['size'],crc32=value['crc'],method=value['method'])
  pool.append(dict(task='textvqa',question_id=qid,image_id=image_id,question=row['question'],
      image_path=str(pathimage) if pathimage.exists() else None,archive=str(archive),archive_entry=entry,
      source=str(path),source_split='official_val',label=dict(answers=row['answers']),source_question_id=qid))
 return pool,status


def scienceqa_pool(used,hashes,rgb):
 folder=UPSTREAM/'vispec/data/scienceqa';problems=read(folder/'problems.json');splits=read(folder/'pid_splits.json')
 path=next((LOCAL/'ScienceQA/data').glob('test-*.parquet'));table=pq.read_table(path);rows=table.to_pylist()
 assert len(rows)==len(splits['test']);pool=[];image_rows=[]
 for index,(qid,row) in enumerate(zip(splits['test'],rows)):
  official=problems[str(qid)]
  for key in ['question','choices','answer']:assert official[key]==row[key],('ScienceQA pid/parquet mapping mismatch',qid,key)
  if not row['image'] or not row['image']['bytes']:continue
  data=row['image']['bytes'];info=image_info_bytes(data);image_rows.append((qid,info))
  pool.append(dict(task='scienceqa',question_id=str(qid),image_id='scienceqa:test:'+str(qid),question=row['question'],hint=row['hint'],choices=row['choices'],
      image_bytes=data,image_info=info,source=str(path),source_row=index,source_split='official_image_test',
      source_question_id=str(qid),label=dict(answer_index=row['answer'],answer_letter='ABCDE'[row['answer']],choices=row['choices'])))
 prior_rgb={info['image_rgb_sha256'] for qid,info in image_rows if canonical(qid) in used['scienceqa']['questions']}
 return [row for row in pool if not is_used('scienceqa',row['question_id'],row['image_id'],used,row['image_info'],hashes,rgb|prior_rgb)]


def mme_pool(used,hashes,rgb):
 folder=LOCAL/'MME/MME_Benchmark_release_version/MME_Benchmark';pool=[]
 for category in CATEGORIES:
  for path in sorted((folder/category).rglob('*.txt')):
   candidates=[path.with_suffix(s) for s in ['.jpg','.png','.jpeg']]+[(folder/category/'images'/path.stem).with_suffix(s) for s in ['.jpg','.png','.jpeg']]
   image=next((v for v in candidates if v.exists()),None)
   if image is None:continue
   key=category+'/'+image.name; lines=path.read_text().splitlines();assert len(lines)==2,(path,len(lines))
   info=image_info_path(image)
   known=key.lower() in used['mme']['questions'] or key.lower() in used['mme']['images']
   if category in ['existence','count','color','position']:
    known=known or canonical(image.stem) in used['coco']['images']
   if known or info['image_sha256'] in hashes or info['image_rgb_sha256'] in rgb:continue
   questions=[]
   for i,line in enumerate(lines):
    question,answer=line.rsplit('\t',1);assert answer.strip().lower() in ['yes','no']
    questions.append(dict(task='mme',question_id=key+'::'+str(i),source_question_id=key,image_id=key,question=question,
      image_path=str(image),category=category,source=str(path),source_split='official_MME',image_info=info,
      label=dict(answer=answer.strip().lower(),official_question=question,official_image=image.name,category=category)))
   pool.append(dict(category=category,image_id=key,image_info=info,questions=questions))
 return pool


def coco_pool(used):
 source=HISTORY[0]/'reference/dataset_coco.json';metadata=read(source)['images'];idmap={canonical(x['imgid']):canonical(x['cocoid']) for x in metadata}
 used['coco']['images'].update(idmap[q] for q in used['coco']['questions'] if q in idmap)
 pool=[]
 for row in metadata:
  ident=row['cocoid']
  if row['split']!='test' or is_used('coco',row['imgid'],ident,used):continue
  pool.append(dict(task='coco',question_id=str(ident),source_question_id=f'COCO_val2014_{ident:012d}.jpg',
      image_id=ident,cocoid=ident,source=str(source),source_split='Karpathy_test',question='',
      label=dict(cocoid=ident,sentences=row['sentences'])))
 return pool


def prompt(row):
 task=row['task'];q=row.get('question','')
 if task in ['gqa','textvqa']:return q+'\nAnswer the question using a single word or phrase.'
 if task=='mme':
  q=re.sub(r'(?:\s*Please answer yes or no\.?\s*)+$','',q,flags=re.I).strip()
  return q+'\nPlease answer yes or no.'
 if task=='scienceqa':
  choice=' '.join(f'({"ABCDE"[i]}) {v}' for i,v in enumerate(row['choices']))
  return f'Question: {q}\nContext: {row.get("hint","").strip() or "N/A"}\nOptions: {choice}\nAnswer with the option letter only.'
 return 'Describe this image in one concise sentence.'


def choose(pool,task,seed,hashes,rgb):
 rows=sorted(pool,key=lambda row:order(seed,task,row['question_id']));groups=[];seen=set();seen_rgb=set()
 if task=='textvqa':
  prefetch=[];unique=set()
  for row in rows:
   if row['image_id'] in unique:continue
   unique.add(row['image_id']);prefetch.append(row)
   if len(prefetch)==160:break
  def load_candidate(row):
   data=Path(row['image_path']).read_bytes() if row.get('image_path') else zip_bytes(Path(row['archive']),row['archive_entry'])
   return row['image_id'],data,image_info_bytes(data)
  with ThreadPoolExecutor(max_workers=8) as executor:loaded={key:(data,info) for key,data,info in executor.map(load_candidate,prefetch)}
  for row in rows:
   if row['image_id'] in loaded:row['image_bytes'],row['image_info']=loaded[row['image_id']]
 for row in rows:
  key=canonical(row['image_id'])
  if key in seen:continue
  info=row.get('image_info')
  if row.get('image_path'):info=info or image_info_path(row['image_path'])
  elif task=='textvqa':
   if info is None:row['image_bytes']=zip_bytes(Path(row['archive']),row['archive_entry']);info=image_info_bytes(row['image_bytes'])
  if info:
   row['image_info']=info
   if info['image_sha256'] in hashes or info['image_rgb_sha256'] in rgb or info['image_rgb_sha256'] in seen_rgb:continue
   seen_rgb.add(info['image_rgb_sha256'])
  seen.add(key);groups.append([row])
  if len(groups)==120:break
 assert len(groups)==120,(task,'insufficient fresh available image groups',len(groups))
 return groups[:20],groups[20:]


def choose_mme(pool,seed):
 bycat={cat:sorted([x for x in pool if x['category']==cat],key=lambda x:order(seed,'mme',x['image_id'])) for cat in CATEGORIES}
 cats=sorted(CATEGORIES,key=lambda c:order(seed,'mme-category',c));used=set();selected={}
 for split,n in [('dev',20),('test',100)]:
  quotas={cat:n//len(cats)+int(i<n%len(cats)) for i,cat in enumerate(cats)};groups=[]
  for cat in cats:
   available=[x for x in bycat[cat] if x['image_info']['image_rgb_sha256'] not in used]
   assert len(available)>=quotas[cat],('MME insufficient category',cat,len(available),quotas[cat])
   taken=0
   for row in available:
    if row['image_info']['image_rgb_sha256'] in used:continue
    used.add(row['image_info']['image_rgb_sha256']);groups.append(row['questions']);taken+=1
    if taken==quotas[cat]:break
   assert taken==quotas[cat],('MME insufficient unique images',cat,taken,quotas[cat])
  selected[split]=sorted(groups,key=lambda group:order(seed,'mme-'+split,group[0]['image_id']))
 return selected['dev'],selected['test']


def materialize(groups,task):
 if task=='coco':
  wanted={row['image_id'] for group in groups for row in group}
  locations={}
  for path in sorted((LOCAL/'COCO/data').glob('val-*.parquet')):
   table=pq.read_table(path,columns=['question_id'])
   for i,name in enumerate(table['question_id'].to_pylist()):
    ident=int(re.fullmatch(r'COCO_val2014_(\d+)\.jpg',name).group(1))
    if ident in wanted:assert ident not in locations;locations[ident]=(path,i)
  assert set(locations)==wanted
  for path in sorted({v[0] for v in locations.values()}):
   table=pq.read_table(path,columns=['image'])
   for group in groups:
    for row in group:
     if locations[row['image_id']][0]==path:
      index=locations[row['image_id']][1];row['image_bytes']=table['image'][index].as_py()['bytes'];row['source_parquet']=str(path);row['source_row']=index
 for group in groups:
  for row in group:
   if not row.get('image_path'):
    data=row.pop('image_bytes',None)
    if data is None:data=zip_bytes(Path(row['archive']),row['archive_entry'])
    suffix='.jpg' if task in ['coco','textvqa'] else '.png'
    destination=ROOT/'data/assets'/task/(str(row['question_id']).replace('/','_').replace(':','_')+suffix)
    destination.parent.mkdir(parents=True,exist_ok=True);assert not destination.exists();destination.write_bytes(data)
    row['image_path']=str(destination)
   row['image_info']=image_info_path(row['image_path'])


def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['inventory','prepare'],default='inventory');p.add_argument('--seed',type=int,default=DEFAULT_SEED);p.add_argument('--refresh-history',action='store_true');p.add_argument('--allow-history-reuse',nargs='*',default=['mme'],choices=TASKS);a=p.parse_args()
 used,hashes,rgb,sources=load_prior(a.refresh_history)
 supplement=supplement_parquet_identities(used,hashes,rgb)
 pools={};pools['gqa']=gqa_pool(used);pools['textvqa'],zipstatus=textvqa_pool(used)
 pools['scienceqa']=scienceqa_pool(used,hashes,rgb);pools['mme']=mme_pool(used,hashes,rgb);pools['coco']=coco_pool(used)
 fresh_counts={t:len(v) for t,v in pools.items()}
 empty={t:dict(questions=set(),images=set()) for t in TASKS}
 for task in a.allow_history_reuse:
  if task=='mme':pools[task]=mme_pool(empty,set(),set())
  elif task=='scienceqa':pools[task]=scienceqa_pool(empty,set(),set())
  elif task=='gqa':pools[task]=gqa_pool(empty)
  elif task=='textvqa':pools[task],_=textvqa_pool(empty)
  elif task=='coco':pools[task]=coco_pool(empty)
 inventory=dict(stage=a.stage,seed=a.seed,task_candidate_rows={t:len(v) for t,v in pools.items()},
  task_candidate_image_groups={t:len({str(x['image_id']) for x in v}) for t,v in pools.items()},
  MME_categories=Counter(x['category'] for x in pools['mme']),textvqa_archive=zipstatus,history_sources=len(sources),
  fresh_candidate_rows_before_explicit_reuse=fresh_counts,explicit_history_reuse_tasks=a.allow_history_reuse,
  prior_identity_counts={t:{k:len(v) for k,v in row.items()} for t,row in used.items()},
  prompts=dict(system=SYSTEM,max_new_tokens=128,qa_suffix='Answer the question using a single word or phrase.',mme_suffix='Please answer yes or no.',sqa_no_caption_lecture_solution=True,coco='Describe this image in one concise sentence.'),
  labels_are_separate_from_model_inputs=True,old_quality_outputs_used_for_selection=False)
 dump(ROOT/'audit/DATA_INVENTORY.json',inventory);print(json.dumps(inventory,ensure_ascii=False),flush=True)
 if a.stage=='inventory':return
 chosen={}
 for task in TASKS:
  chosen[task]=choose_mme(pools[task],a.seed) if task=='mme' else choose(pools[task],task,a.seed,set() if task in a.allow_history_reuse else hashes,set() if task in a.allow_history_reuse else rgb)
  materialize(chosen[task][0]+chosen[task][1],task)
 manifests={};global_ordinal={'dev':0,'test':0}
 for task in TASKS:
  for split,groups in zip(['dev','test'],chosen[task]):
   records=[];labels={};image_hashes=set()
   for group_ordinal,group in enumerate(groups):
    for row in group:
     qid=str(row['question_id']);gid=f'{task}:{group_ordinal}:{row["image_id"]}';info=row['image_info'];image_hashes.add(info['image_rgb_sha256'])
     record=dict(task=task,ordinal=len(records),global_ordinal=global_ordinal[split],group_ordinal=group_ordinal,
      question_id=qid,image_id=row['image_id'],image_path=row['image_path'],group_id=gid,
      prompt_text=prompt(row),system_prompt=SYSTEM,history_reuse_required=task in a.allow_history_reuse,fresh_to_project=task not in a.allow_history_reuse,**info)
     if 'category' in row:record['category']=row['category']
     if task=='coco':record['cocoid']=row['cocoid']
     records.append(record);global_ordinal[split]+=1
     labels[qid]=dict(row['label'],source=row['source'],source_split=row['source_split'],source_question_id=row['source_question_id'],
       source_row=row.get('source_row'),source_parquet=row.get('source_parquet'),image_archive=row.get('archive'),image_archive_entry=row.get('archive_entry'),image_sha256=info['image_sha256'])
   runtime=ROOT/f'data/{task}/{split}.json';gold=ROOT/f'data/{task}/{split}_labels.json'
   dump(runtime,records,preserve=True);dump(gold,labels,preserve=True)
   manifests[f'{task}/{split}']=dict(path=str(runtime),sha256=sha(runtime),labels_path=str(gold),labels_sha256=sha(gold),
     groups=len(groups),questions=len(records),unique_rgb_images=len(image_hashes))
  devhash={r['image_info']['image_rgb_sha256'] for g in chosen[task][0] for r in g};testhash={r['image_info']['image_rgb_sha256'] for g in chosen[task][1] for r in g}
  assert not devhash&testhash,(task,'dev/test image overlap')
 dump(ROOT/'audit/DATA_MANIFEST.json',dict(seed=a.seed,tasks=TASKS,aliases=dict(sqa='scienceqa',coco_caption='coco'),splits=manifests,
  development_groups_per_task=20,test_groups_per_task=100,MME_questions_per_group=2,all_generation_repetitions=3,
  runtime_records_contain_gold_labels=False,explicit_history_reuse_tasks=a.allow_history_reuse,prompt_agreement='root standard short prompts confirmed before preparation',
  official_helper_explanation_prompts_replaced=True,old_results_quality_not_used=True),preserve=True)
 print(json.dumps(dict(status='prepared',manifests=manifests)),flush=True)

if __name__=='__main__':main()
