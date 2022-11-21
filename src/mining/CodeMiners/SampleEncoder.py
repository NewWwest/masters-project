
from transformers import AutoTokenizer
import unicodedata
import regex as re


splitter_regex = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|((?<![a-zA-Z0-9])(?=[a-zA-Z0-9])|(?<=[a-zA-Z0-9])(?![a-zA-Z0-9]))')
model_input_size = 512
max_tokenizer_input = 128*1024 #for RAM paging issues 

class SampleEncoder:
    def __init__(self, tokenizer_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def process_sample(self, sample_object):
        commit_title = sample_object['commit_title']
        sample = sample_object['commit_sample']

        if sample == None or len(sample) == 0:
            return None

        message_tokens = self.tokenizer.tokenize(commit_title)
        normalized_tokens = splitter_regex.split(str(unicodedata.normalize('NFKD', sample).encode('ascii', 'ignore')))
        normalized_tokens = [t for t in normalized_tokens if t]
        to_tokenize = ' '.join(normalized_tokens)
        to_tokenize = to_tokenize[0:min(len(to_tokenize), max_tokenizer_input)]

        code_tokens = self.tokenizer.tokenize(to_tokenize)
        code_tokens_max_size = model_input_size - 3 - len(message_tokens)
        code_tokens = code_tokens[0:min(code_tokens_max_size, len(code_tokens))]

        tokens = [self.tokenizer.cls_token] + message_tokens + [self.tokenizer.sep_token] + code_tokens + [self.tokenizer.sep_token]
        if len(tokens) < model_input_size:
            tokens = tokens + [self.tokenizer.pad_token] * (model_input_size-len(tokens))

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens_ids








