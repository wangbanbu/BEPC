"""Map reviewed QA semantic-unit candidates to original generated token spans.
These are descriptive output annotations, never online signals or harmful labels.
"""
import json
import re
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
tok=AutoTokenizer.from_pretrained('/data/wbb/model/Qwen2.5-VL-7B-Instruct')
pairs={(p['task'],p['question_id']):p for p in json.loads((ROOT/'data/paired_records.json').read_text())}
review=json.loads((ROOT/'audit/QA_CASE_REVIEW.json').read_text())['cases']
rows=[]
for case in review:
    pair=pairs[case['task'],case['question_id']]
    for method in ['bepc','margin']:
        ids=pair[method]['returned_token_ids']
        prefixes=[tok.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False) for i in range(len(ids)+1)]
        text=prefixes[-1]
        boundaries=[len(p) for p in prefixes]
        stable=all(text.startswith(p) for p in prefixes)
        for unit in case['critical_semantic_units']:
            flags=re.IGNORECASE
            if not any(c.isalnum() for c in unit):
                pattern=re.escape(unit)
                rule='literal_punctuation'
            elif re.fullmatch(r'\d+(?:\.\d+)?',unit):
                pattern=r'(?<![\w.])'+re.escape(unit)+r'(?!\w|\.\d|,\d)'
                rule='complete_numeric_surface_not_decimal_or_thousands_prefix'
            elif re.fullmatch(r'[A-Z]',unit):
                pattern=r'(?<!\w)'+re.escape(unit)+r'(?!\w)'
                flags=0
                rule='case_sensitive_single_letter_candidate_not_verified_option_role'
            else:
                pattern=r'(?<!\w)'+re.escape(unit)+r'(?!\w)'
                rule='case_insensitive_word_bounded_literal'
            matches=list(re.finditer(pattern,text,flags=flags))
            base=dict(task=case['task'],question_id=case['question_id'],image_id=str(case['image_id']),
                method=method,semantic_unit_candidate=unit,classification_source='model-assisted full-output review',
                role='descriptive_only_not_online_signal_or_new_ground_truth',prefix_decoder_stable=stable,
                matching_rule=rule,semantic_role_requires_contextual_review=True)
            if not matches:
                rows.append(dict(base,status='not_present'))
            for i,match in enumerate(matches):
                a,b=match.span()
                overlapping=[j for j in range(len(ids)) if boundaries[j]<b and boundaries[j+1]>a]
                start,end=min(overlapping),max(overlapping)+1
                rows.append(dict(base,status='mapped' if stable else 'unicode_boundary_uncertain',occurrence=i,
                    char_start=a,char_end=b,exact_surface=text[a:b],token_start=start,token_end_exclusive=end,
                    token_ids=json.dumps(ids[start:end]),token_decoded_span=tok.decode(ids[start:end],skip_special_tokens=False),
                    context=text[max(0,a-50):min(len(text),b+50)],
                    starts_inside_token=boundaries[start]!=a,ends_inside_token=boundaries[end]!=b))
df=pd.DataFrame(rows)
df.to_csv(ROOT/'data/qa_critical_semantic_units.csv',index=False)
print(df.status.value_counts().to_json())
