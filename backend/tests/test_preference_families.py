"""Multi-negative opt-in retains binding, duplicate, and split protections."""
import copy
import pytest
from training.preference_validation import reviewed_pair_digest,validate_pairs,validate_partitions

def pair(i=0):
    row={'id':f'p{i}','prompt':'same scene','chosen':'character answer','rejected':f'candidate {i}',
         'metadata':{'schema_version':'persona-preference-v1','persona':'kisaki','source_group':'scene',
         'source_ids':['source'],'evidence_text_sha256':['evidence'],'review_sha256':f'review{i}',
         'chosen_candidate_id':'original','rejected_candidate_id':f'model{i}',
         'human_final_approved':True,'review_status':'approved',
         'preference_family':{'id':'family','max_negatives':4,'review_method':'individually_reviewed_distinct_negatives'}}}
    return bind(row)

def bind(row):
    row['metadata']['pair_content_sha256']=reviewed_pair_digest(row)
    return row

def test_reviewed_distinct_family_is_accepted():
    validate_pairs([pair(i) for i in range(4)],split='train')

@pytest.mark.parametrize('change',[
    lambda r:r['metadata'].pop('preference_family'),
    lambda r:r.update(chosen='different chosen'),
    lambda r:r.update(rejected='candidate 0'),
    lambda r:r['metadata'].update(rejected_candidate_id='model0'),
    lambda r:r['metadata'].update(source_ids=['other']),
    lambda r:r['metadata']['preference_family'].update(id='other'),
    lambda r:r.update(prompt='other prompt'),
])
def test_family_mismatches_fail_even_when_rebound(change):
    second=pair(1);change(second);bind(second)
    with pytest.raises(ValueError):validate_pairs([pair(),second],split='train')

def test_family_limit_and_unbound_mutation_fail():
    with pytest.raises(ValueError):validate_pairs([pair(i) for i in range(5)],split='train')
    row=pair();row['metadata']['preference_family']['max_negatives']=3
    with pytest.raises(ValueError,match='binding mismatch'):validate_pairs([row],split='train')

def test_family_cannot_cross_partitions():
    with pytest.raises(ValueError,match='cross-split'):validate_partitions([pair(0)],[pair(1)])

def test_legacy_duplicates_still_fail():
    rows=[pair(i) for i in range(2)]
    for row in rows:row['metadata'].pop('preference_family');bind(row)
    with pytest.raises(ValueError,match='duplicate normalized'):validate_pairs(rows,split='train')
