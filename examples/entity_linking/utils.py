# This code is based on the code obtained from here:
# https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/dataset.py
import pickle
from tqdm import tqdm
import logging
import math
import os
import re
from collections import defaultdict
import numpy as np

from examples.utils.mention_db import MentionDB
from examples.utils.mention_db import BertLowercaseNormalizer

logger = logging.getLogger(__name__)

class EntityLinkingDataset(object):
    def __init__(self, dataset_dir, mention_db_path, wikipedia_titles_file=None, wikipedia_redirects_file=None):
        person_names = frozenset(load_person_names(os.path.join(dataset_dir, 'persons.txt')))
        logger.info('-MentionDB')
        self.mention_DB = MentionDB(mention_db_path)
        logger.info('-train')
        self.train = load_documents(os.path.join(dataset_dir, 'aida_train.txt'), person_names, self.mention_DB)
        logger.info('-test_a')
        self.test_a = load_documents(os.path.join(dataset_dir, 'testa_testb_aggregate_original'), person_names, self.mention_DB)
        logger.info('-test_b')
        self.test_b = load_documents(os.path.join(dataset_dir, 'testa_testb_aggregate_original'), person_names, self.mention_DB)
        logger.info('-ace2004')
        self.ace2004 = load_documents(os.path.join(dataset_dir, 'ace2004.conll'), person_names, self.mention_DB)
        logger.info('-aquaint')        
        self.aquaint = load_documents(os.path.join(dataset_dir, 'aquaint.conll'), person_names, self.mention_DB)
        logger.info('-clueweb')
        self.clueweb = load_documents(os.path.join(dataset_dir, 'clueweb.conll'), person_names, self.mention_DB)
        logger.info('-msnbc')        
        self.msnbc = load_documents(os.path.join(dataset_dir, 'msnbc.conll'), person_names, self.mention_DB)
        logger.info('-wikipedia')
        self.wikipedia = load_documents(os.path.join(dataset_dir, 'wikipedia.conll'), person_names, self.mention_DB)

        valid_titles = None
        if wikipedia_titles_file:
            with open(wikipedia_titles_file) as f:
                valid_titles = [l.rstrip() for l in f]
                valid_titles += ['[NO_E]']
                valid_titles = frozenset(valid_titles)

        redirects = {}
        if wikipedia_redirects_file:
            with open(wikipedia_redirects_file) as f:
                for line in f:
                    (src, dest) = line.rstrip().split('\t')
                    redirects[src] = dest

        logger.info('Building entity vocab and resolving wikipedia vocabularies')
        # build entity vocabulary and resolve Wikipedia redirects
        for documents in self.get_all_datasets():
            for document in documents:
                new_mentions = []
                for mention in document.mentions:
                    mention.title = redirects.get(mention.title, mention.title)
                    if valid_titles and mention.title not in valid_titles:
                        logger.debug('Invalid title: %s', mention.title)
                        continue
                    new_mentions.append(mention)
                    for candidate in mention.candidates:
                        candidate.title = redirects.get(candidate.title, candidate.title)
                document.mentions = new_mentions
        logger.info('Done')

    def get_all_datasets(self):
        return (
            self.train,
            self.test_a,
            self.test_b,
            self.ace2004,
            self.aquaint,
            self.clueweb,
            self.msnbc,
            self.wikipedia
        )


class Document(object):
    def __init__(self, id_, words, mentions):
        self.id = id_
        self.words = words
        self.mentions = mentions

    def __repr__(self):
        return '<Document %s...>' % (' '.join(self.words[:3]),)


class Mention(object):
    def __init__(self, text, title, start, end, candidates):
        self.text = text
        self.start = start
        self.end = end
        self.title = title
        self.candidates = candidates

    @property
    def span(self):
        return self.start, self.end

    def __repr__(self):
        return '<Mention %s->%s>' % (self.text, self.title)


class Candidate(object):
    def __init__(self, title, prior_prob):
        self.title = title
        self.prior_prob = prior_prob

    def __repr__(self):
        return '<Candidate %s (prior prob: %.3f)>' % (self.title, self.prior_prob)


class InputFeatures(object):
    def __init__(self, document, mentions, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                 entity_position_ids, entity_segment_ids, entity_attention_mask, entity_candidate_ids,
                 target_mention_indices):
        self.document = document
        self.mentions = mentions
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids #
        self.entity_segment_ids = entity_segment_ids #
        self.entity_attention_mask = entity_attention_mask #
        self.entity_candidate_ids = entity_candidate_ids #
        self.target_mention_indices = target_mention_indices #


def load_person_names(input_file):
    with open(input_file) as f:
        return [l.strip() for l in f]

def load_documents(conll_path, person_names, mention_DB):
    
    # in order to find coreference 
    def person_name_coreference(target_mention, mention_list):
        target_mention_text = target_mention['text'].lower()
        ret = []

        for mention in mention_list:
            if not mention['candidates'] or mention['candidates'][0].title not in person_names:
                continue

            mention_text = mention['text'].lower()
            if mention_text == target_mention_text:
                continue

            start_pos = mention_text.find(target_mention_text)
            if start_pos == -1:
                continue

            end_pos = start_pos + len(target_mention_text) - 1
            if (start_pos == 0 or mention_text[start_pos - 1] == ' ') and\
               (end_pos == len(mention_text) - 1 or mention_text[end_pos + 1] == ' '):
                ret.append(mention)

        return ret

    document_data = {}

    with open(conll_path, 'r') as f:
        cur_doc = {}

        for line in f:
            if line == '\n':
                cur_doc['words'].append('[SEP]')
                continue

            line = line.strip()
            if line.startswith('-DOCSTART-'):
                doc_name = line.split()[1][1:]
                #document_data[doc_name] = dict(words=[], mentions=[], mention_spans=[])
                document_data[doc_name] = dict(words=[], mentions=[], titles=[], detected_mention_spans=[], gold_mention_spans=[])
                cur_doc = document_data[doc_name]

            else:
                comps = line.split('\t')
                if len(comps) >= 6:
                    tag = comps[1]
                    if tag == 'I':
                        cur_doc['gold_mention_spans'][-1]['end'] += 1
                    else:
                        cur_doc['gold_mention_spans'].append(dict(start=len(cur_doc['words']),
                                                             end=len(cur_doc['words']) + 1))
                        # mention のはじめのみにtitleを保存
                        title = comps[-4]
                        title = title.replace('_', ' ')
                        title = title.replace('&amp;', '&')
                        cur_doc['titles'].append(title)

                cur_doc['words'].append(comps[0])

    for _, cur_doc in document_data.items():
        for (doc_mention_title, doc_mention_span) in zip(cur_doc['titles'], cur_doc['gold_mention_spans']):
            doc_mention_text = ' '.join(cur_doc['words'][doc_mention_span['start']:doc_mention_span['end']])
            doc_mention_cand = sorted([Candidate(m.title, m.prior_prob) for m in mention_DB.query(doc_mention_text)], key= lambda c: c.prior_prob, reverse=True)
            cur_doc['mentions'].append({'text':doc_mention_text, 'candidates':doc_mention_cand, 'title':doc_mention_title, 'start':doc_mention_span['start'], 'end':doc_mention_span['end']})


    documents = []

    for doc_id in document_data.keys():
        current_document = document_data[doc_id]
        

        for i in range(len(current_document['words'])):
            if current_document['words'][i] =='[SEP]':
                continue
            for j in range(i+1, len(current_document['words'])):
                if current_document['words'][j-1] =='[SEP]':
                    break
                if j-i > 15: #とりあえず
                    break
                if {'start':i, 'end':j} not in current_document['gold_mention_spans']:
                    query_result = mention_DB.query(' '.join(current_document['words'][i:j]))
                    if query_result:
                        current_document['detected_mention_spans'].append(dict(start=i, end=j))
                        current_document['titles'].append('[NO_E]')
                        doc_mention_text = ' '.join(current_document['words'][i:j])
                        doc_mention_cand = sorted([Candidate(m.title, m.prior_prob) for m in query_result], key= lambda c: c.prior_prob, reverse=True)
                        current_document['mentions'].append({'text':doc_mention_text, 'candidates':doc_mention_cand, 'title':'[NO_E]', 'start':i, 'end':j})

        
        # find_coreference and update candidates
        for m in current_document['mentions']:
            coref_mentions = person_name_coreference(m, current_document['mentions'])
            if coref_mentions:
                new_cands = defaultdict(int)
                for coref_m in coref_mentions:
                    for cand in coref_m['candidates']:
                        new_cands[cand.title] += cand.prior_prob
                for cand_title in new_cands.keys():
                    new_cands[cand_title] /= len(coref_mentions)
                m['candidates'] = sorted([Candidate(t, p) for (t, p) in new_cands.items()], key=lambda c: c.prior_prob, reverse=True)


        mentions = [Mention(**o) for o in current_document['mentions']]
        documents.append(Document(doc_id, current_document['words'], mentions)) 

    return documents

def convert_documents_to_features(documents, tokenizer, entity_vocab, mode, max_seq_length,
                                  max_candidate_length, max_mention_length, max_entity_length):
    max_num_tokens = max_seq_length - 2

    def generate_feature_dict(tokens, mentions, ctx_start, ctx_end, snt_start, snt_end):

        all_tokens = [tokenizer.cls_token] + tokens[ctx_start:ctx_end] + [tokenizer.sep_token]
        word_ids = np.array(tokenizer.convert_tokens_to_ids(all_tokens), dtype=np.int)
        word_attention_mask = np.ones(len(all_tokens), dtype=np.int)
        word_segment_ids = np.zeros(len(all_tokens), dtype=np.int)

        target_mention_data = []
        for start, end, mention in mentions:
            if start >= snt_start and end <= snt_end:
                candidates = [c.title for c in mention.candidates[:max_candidate_length]]
                candidates.append('[NO_E]') # 

                if mode == 'train' and mention.title not in candidates:
                    continue
                if end - start > max_mention_length:
                    continue
                target_mention_data.append((start - ctx_start, end - ctx_start, mention, candidates))

        entity_ids = np.empty(len(target_mention_data), dtype=np.int)
        entity_attention_mask = np.ones(len(target_mention_data), dtype=np.int)
        entity_segment_ids = np.zeros(len(target_mention_data), dtype=np.int)
        entity_position_ids = np.full((len(target_mention_data), max_mention_length), -1, dtype=np.int)
        entity_candidate_ids = np.zeros((len(target_mention_data), max_candidate_length + 1), dtype=np.int) # +1 for [NO_E]

        for index, (start, end, mention, candidates) in enumerate(target_mention_data):
            entity_ids[index] = entity_vocab[mention.title]
            entity_position_ids[index][:end - start] = range(start + 1, end + 1)  # +1 for [CLS]
            entity_candidate_ids[index, :len(candidates)] = [entity_vocab[cand] for cand in candidates]
        output_mentions = [mention for _, _, mention, _ in target_mention_data]

        # return tuple of list
        if len(output_mentions) > max_entity_length:
            splited_output_mentions = []
            splited_output_dict = []
            split_size = math.ceil(len(output_mentions) / max_entity_length)
            for i in range(split_size):
                entity_size = math.ceil(len(output_mentions) / split_size)
                start = i * entity_size
                end = start + entity_size
                splited_output_mentions.append(output_mentions[start:end])
                splited_output_dict.append(
                    dict(word_ids=word_ids,
                        word_segment_ids=word_segment_ids,
                        word_attention_mask=word_attention_mask,
                        entity_ids=entity_ids[start:end],
                        entity_position_ids=entity_position_ids[start:end],
                        entity_segment_ids=entity_segment_ids[start:end],
                        entity_attention_mask=entity_attention_mask[start:end],
                        entity_candidate_ids=entity_candidate_ids[start:end])
                )
            return splited_output_mentions, splited_output_dict
        else:
            return [output_mentions], [dict(word_ids=word_ids,
                                        word_segment_ids=word_segment_ids,
                                        word_attention_mask=word_attention_mask,
                                        entity_ids=entity_ids,
                                        entity_position_ids=entity_position_ids,
                                        entity_segment_ids=entity_segment_ids,
                                        entity_attention_mask=entity_attention_mask,
                                        entity_candidate_ids=entity_candidate_ids)]

    ret = []
    for document in documents:

        mention_data = []

        subword_list = [tokenizer.tokenize(w) for w in document.words] # list of list
        
        assert len(subword_list) == len(document.words)

        count = 0
        index_map = {} # tokenize前の単語index(かつ'[SEP]'込み)が， tokenize後(かつ'[SEP]'抜き)にどんなindexになるか
        sent_indexs = [] # record end_of_sentence index (in terms of subwords without '[SEP]')

        for i, sub_tokens in enumerate(subword_list):
            index_map[i] = count
            if len(sub_tokens) == 1 and sub_tokens[0] == '[SEP]':
                sent_indexs.append(count)
                continue
            count += len(sub_tokens)

        sub_word_length = count
        sub_word_without_sep = [w for ws in subword_list for w in ws if w != '[SEP]'] # tokenize された '[SEP]' 抜きの subword token 列
        
        assert sub_word_length == len(sub_word_without_sep)
        
        if sent_indexs[-1] != sub_word_length: # sub_word_listの最後が[SEP]でなかった場合
            sent_indexs.append(sub_word_length)
        
        for mention in document.mentions:
            mention_data.append((index_map[mention.start], index_map[mention.end], mention)) # sub_word_withut_sep 上でのindexにmap
        
        sentence_start = 0
        sentence_record = []
        for sentence_end in sent_indexs:
            sentence_record.append((sentence_start, sentence_end))
            sentence_start = sentence_end



        for sent_start, sent_end in sentence_record:
            left_token_length = sent_start
            right_token_length = sub_word_length - sent_end
            sentence_length = sent_end - sent_start
            half_context_size = int((max_num_tokens - sentence_length) / 2)
 
            if left_token_length < right_token_length:
                left_context_length = min(left_token_length, half_context_size)
                right_context_length = min(right_token_length, max_num_tokens - left_context_length - sentence_length)
            else:
                right_context_length = min(right_token_length, half_context_size)
                left_context_length = min(left_token_length, max_num_tokens - right_context_length - sentence_length)
            
            output_mentions, feature_dict = generate_feature_dict(sub_word_without_sep, mention_data, sent_start - left_context_length, sent_end + right_context_length, sent_start, sent_end)
            
            if output_mentions[0]:# [[], []] or [[]] or [[none]]
                for splited_output_mentions, splited_feature_dict in zip(output_mentions, feature_dict):
                    ret.append(InputFeatures(document=document,
                                             mentions=splited_output_mentions,
                                             target_mention_indices=range(len(splited_output_mentions)),
                                             **splited_feature_dict))
    return ret
