import logging
import os
import json
import warnings

import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

BASE_PATH = "/opt/ml/processing"
INPUT_PATH = os.path.join(BASE_PATH, "input")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '<|endoftext|>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'


def preprocess(data, bos_token, sept_token, answer):
    protagonist = data['characters'][0]['id']
    sentences = []
    length = len(data['scenes'][0]['items'])
    scences = data['scenes'][0]['items'].copy()
    idx = 0
    tmp_id = 0
    while True:
        tmp_sentence = ''
        for content in scences:
            if content['object_type'] == 'text' and content['character_id'] == protagonist \
                    and bos_token not in tmp_sentence:
                tmp_sentence = bos_token + tmp_sentence + content['object']['text']
                if idx < length - 1:
                    if data['scenes'][0]['items'][idx + 1]['object_type'] != 'text':
                        tmp_sentence = ''
                        idx = idx + 1
                        break
                    elif data['scenes'][0]['items'][idx + 1]['character_id'] != protagonist:
                        tmp_id = data['scenes'][0]['items'][idx + 1]['character_id']
                if idx == length - 1:
                    tmp_sentence = ''
                    idx = idx + 1
                    break
                idx = idx + 1
                continue

            if content['object_type'] == 'text' and content[
                'character_id'] == protagonist and bos_token in tmp_sentence:
                tmp_sentence = tmp_sentence + sept_token + content['object']['text']
                # 대화만 넣어야 됨
                if idx < length - 1:
                    if data['scenes'][0]['items'][idx + 1]['object_type'] != 'text':
                        tmp_sentence = ''
                        idx = idx + 1
                        break
                    elif data['scenes'][0]['items'][idx + 1]['character_id'] != protagonist:
                        tmp_id = data['scenes'][0]['items'][idx + 1]['character_id']
                if idx == length - 1 and answer not in tmp_sentence:
                    tmp_sentence = ''
                    idx = idx + 1
                    break
                idx = idx + 1
                continue
            if data['scenes'][0]['items'][idx - 1]['object_type'] != 'text':
                idx = idx + 1
                break
            if content['object_type'] == 'text' and idx != 0 and content['character_id'] == tmp_id and \
                    data['scenes'][0]['items'][idx - 1]['character_id'] == protagonist and answer not in tmp_sentence:
                tmp_sentence = tmp_sentence + answer + content['object']['text']
                if idx < length - 1:
                    if data['scenes'][0]['items'][idx + 1]['object_type'] == 'text' and \
                            data['scenes'][0]['items'][idx + 1]['character_id'] == protagonist:
                        idx = idx + 1
                        break
                idx = idx + 1
                continue
            if content['object_type'] == 'text' and idx != 0 and content['character_id'] == tmp_id and \
                    data['scenes'][0]['items'][idx - 1]['character_id'] == protagonist and answer in tmp_sentence:
                tmp_sentence = tmp_sentence + sept_token + content['object']['text']
                if idx < length - 1:
                    if data['scenes'][0]['items'][idx + 1]['object_type'] == 'text' and \
                            data['scenes'][0]['items'][idx + 1]['character_id'] == protagonist:
                        idx = idx + 1
                        break
                idx = idx + 1
                continue
            idx = idx + 1
            break
        if tmp_sentence != '':
            sentences.append(tmp_sentence)
        scences = data['scenes'][0]['items'][idx:]
        # idx = idx + 1
        if idx == length:
            break
    return sentences


def main():
    logger.info("Starting preprocessing")
    logger.info("Reading input data")
    # df = pd.read_csv(os.path.join(INPUT_PATH, "data.csv"))
    # logger.info("Input data shape: {}".format(df.shape))
    sentence_sum = []
    for file in os.listdir('INPUT_PATH'):
        if 'chatie' in file:
            with open(os.path.join('./drive/chat/', file), 'r') as f:
                test = f.readline()
            data = json.loads(test)
            sentences = preprocess(data, bos_token=BOS, sept_token='<|sept|>', answer=S_TKN)
            sentence_sum = sentence_sum + sentences
    with open(os.path.join('OUTPUT_PATH', 'data.json'), 'w') as f:
        json.dump(sentence_sum, f)
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
