from typing import List, Union, Dict , OrderedDict
import warnings
import re
import time
from tqdm import tqdm
from torch.utils.data import Dataset , DataLoader

class SymbolicQEDTokenizer:
    def __init__(self, df=None, index_token_pool_size=100, special_symbols=None, unk_idx=1, to_replace=True):
        self.amps = df.amp.tolist() if df is not None else None
        self.sqamps = df.sqamp.tolist() if df is not None else None
        if index_token_pool_size < 50:
            warnings.warn(f"Index token pool size ({index_token_pool_size}) may be insufficient. Consider using at least 50-100 tokens for symbolic tasks.", UserWarning)
        self.index_pool = [f"INDEX_{i}" for i in range(index_token_pool_size)]
        self.particle_index_pool = [f"PINDEX_{i}" for i in range(index_token_pool_size)]
        self.special_symbols = special_symbols or ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
        self.unk_idx = unk_idx
        self.to_replace = to_replace
        self.pattern_underscore_curly = re.compile(r'\b[\w]+(?:_[\w]+)*_{')
        self.pattern_mass = re.compile(r'\bm_([a-z]+)\b')
        self.pattern_mandelstam = re.compile(r'\bs_(\d{2,})\b')
        self.pattern_momentum = re.compile(r'\bp_(\d+)\b')
        self.pattern_single_s = re.compile(r'\bs_(\d+)\b(?!\d)')
        self.pattern_exponent = re.compile(r'\^(\w+|\([^)]+\))')
        self.pattern_special = re.compile(r'_([uv])|\\(\w+_\d+|\w+\b)')
        self.pattern_num_123 = re.compile(r'\b(?![psijkl]_)(?!MOMENTUM_)(?!MASS_)(?!P_)(?!S_)(?!MANDELSTAM_)\w+_\d+\b')
        self.pattern_particle = re.compile(r'(?P<prefix>\b(?:\w+_)?)?(?P<target>[ijkl]_\d+\b)')

    def preprocess_expression(self, expr):
        expr = expr.replace(' * ', '*').replace(' / ', '/').replace(' ^ ', '^')
        expr = expr.replace(' + ', '+').replace(' - ', '-')
        expr = expr.replace("+-", "-")
        expr = expr.replace("-+", "-")
        expr = ' '.join(expr.split())
        expr = expr.replace('me', 'm_e')
        return expr

    @staticmethod
    def remove_whitespace(expression: str):
        return re.sub(r'\s+', '', expression)

    def protect_structures(self, ampl: str):
        protected = []
        return ampl, protected

    def physics_aware_replace(self, ampl: str, is_source: bool = True):
        ampl = self.remove_whitespace(ampl)
        ampl = re.sub(r'\bi\b(?!\w)', 'I_UNIT', ampl)
        ampl = re.sub(r'\be\b(?=\^|[+\-*/()| ])', 'E_CHARGE', ampl)
        ampl = ampl.replace('reg_prop', 'REG_PROP')
        ampl = self.pattern_mandelstam.sub(r'MANDELSTAM_\1', ampl)
        ampl = self.pattern_momentum.sub(r'P_\1', ampl)
        ampl = self.pattern_single_s.sub(r'S_\1', ampl)
        ampl = ampl.replace('(*)', 'CONJ')
        return ampl

    def replace_indices(self, ampl: str, is_source: bool = True):
        if not self.to_replace:
            return ampl
        index_pool = iter(self.index_pool)
        particle_index_pool = iter(self.particle_index_pool)
        index_pool_set = set(self.index_pool) if is_source else set()
        ampl = self.pattern_mandelstam.sub(lambda m: f'MANDELSTAM_{m.group(1)}', ampl)

        def get_unique_matches(pattern):
            matches = list(OrderedDict.fromkeys(pattern.findall(ampl)))
            return [m for m in matches if m not in index_pool_set]

        def replace_particle_tokens():
            nonlocal ampl
            matches = list(OrderedDict.fromkeys(m.group('target') for m in sorted(self.pattern_particle.finditer(ampl), key=lambda m: m.start())))
            try:
                mapping = {m: next(particle_index_pool) for m in matches}
            except StopIteration:
                raise RuntimeError("particle_index_pool exhausted. Increase the size of the particle_index_pool.")
            for key in sorted(mapping.keys(), key=len, reverse=True):
                ampl = ampl.replace(key, mapping[key])

        matches = get_unique_matches(self.pattern_num_123)
        try:
            for match in matches:
                ampl = ampl.replace(match, next(index_pool))
        except StopIteration:
            raise RuntimeError("index_pool exhausted. Increase pool size.")
        replace_particle_tokens()
        return ampl

    def tokenize_expression(self, ampl: str, protected: List[str], is_source: bool = True):
        ampl = ampl.replace('\\\\', '\\')
        def replace_special(match):
            if match.group(1):
                return f' _ {match.group(1)} '
            elif match.group(2):
                return f' \\ {match.group(2)} '
        ampl = self.pattern_special.sub(replace_special, ampl)
        if is_source:
            ampl = self.pattern_underscore_curly.sub(lambda match: f' {match.group(0)} ', ampl)
            for symbol in ['{', '}', ',']:
                ampl = ampl.replace(symbol, f' {symbol} ')
        for symbol in ['/', '+', '-', '*', '(', ')', '^']:
            ampl = ampl.replace(symbol, f' {symbol} ')
        ampl = self.pattern_exponent.sub(r' ^ \1 ', ampl)
        ampl = ampl.replace('_PINDEX', '_ PINDEX').replace('_INDEX', '_ INDEX')
        ampl = ampl.replace('REG_PROP', ' reg_prop ')
        ampl = re.sub(r' +', ' ', ampl).strip()
        tokens = [token for token in ampl.split(' ') if token]
        final_tokens = []
        for token in tokens:
            if token.startswith('PROTECTED_'):
                try:
                    idx = int(token.split('_')[1])
                    final_tokens.append(protected[idx])
                except (IndexError, ValueError):
                    final_tokens.append(token)
            else:
                final_tokens.append(token)
        return final_tokens

    def src_tokenize(self, ampl: str):
        try:
            ampl = self.preprocess_expression(ampl)
            ampl, protected = self.protect_structures(ampl)
            ampl = self.physics_aware_replace(ampl, is_source=True)
            ampl = self.replace_indices(ampl, is_source=True)
            return self.tokenize_expression(ampl, protected, is_source=True)
        except Exception as e:
            warnings.warn(f"Source tokenization failed for '{ampl}': {e}")
            return [self.special_symbols[self.unk_idx]]

    def tgt_tokenize(self, sqampl: str):
        try:
            sqampl = self.preprocess_expression(sqampl)
            sqampl, protected = self.protect_structures(sqampl)
            sqampl = self.physics_aware_replace(sqampl, is_source=False)
            sqampl = self.replace_indices(sqampl, is_source=False)
            return self.tokenize_expression(sqampl, protected, is_source=False)
        except Exception as e:
            warnings.warn(f"Target tokenization failed for '{sqampl}': {e}")
            return [self.special_symbols[self.unk_idx]]

    def build_src_vocab(self):
        if self.amps is None:
            return set()
        vocab_set = set()
        start_time = time.time()
        for expr in tqdm(self.amps, desc="Processing source vocab"):
            vocab_set.update(self.src_tokenize(expr))
        end_time = time.time()
        print(f"Source vocab built in {end_time - start_time:.2f} seconds, size: {len(vocab_set)}")
        return vocab_set

    def build_tgt_vocab(self):
        if self.sqamps is None:
            return set()
        vocab_set = set()
        start_time = time.time()
        for expr in tqdm(self.sqamps, desc="Processing target vocab"):
            vocab_set.update(self.tgt_tokenize(expr))
        end_time = time.time()
        print(f"Target vocab built in {end_time - start_time:.2f} seconds, size: {len(vocab_set)}")
        return vocab_set

class SymbolicVocab:
    def __init__(self, tokens: set, special_symbols: list, bos_idx: int, pad_idx: int, eos_idx: int, unk_idx: int, sep_idx: int):
        self.token_list = special_symbols + sorted(list(tokens))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.token_list)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.sep_idx = sep_idx
        self.unk_tok = special_symbols[unk_idx]
        self.pad_tok = special_symbols[pad_idx]
        self.bos_tok = special_symbols[bos_idx]
        self.eos_tok = special_symbols[eos_idx]
        self.sep_tok = special_symbols[sep_idx]

    def encode(self, tokens: list):
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: list, include_special_tokens: bool = True):
        if include_special_tokens:
            return [self.idx_to_token.get(idx, self.unk_tok) for idx in indices]
        return [self.idx_to_token.get(idx, self.unk_tok) for idx in indices if idx not in {self.pad_idx, self.bos_idx, self.eos_idx, self.sep_idx}]

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.idx_to_token.get(item, self.unk_tok)
        return self.token_to_idx.get(item, self.unk_idx)