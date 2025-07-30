from collections import OrderedDict
import re
from tqdm import tqdm
import warnings


class Tokenizer:
    """
    Tokenizer for processing symbolic mathematical expressions.
    """

    def __init__(self, df, index_token_pool_size, special_symbols, UNK_IDX, to_replace):
        self.amps = df.amp.tolist() if df is not None else None
        self.sqamps = df.sqamp.tolist() if df is not None else None

        # Issue warnings if token pool sizes are too small
        if index_token_pool_size < 30:
            warnings.warn(
                f"Index token pool size ({index_token_pool_size}) is small. "
                "Consider increasing it.",
                UserWarning
            )

        # Generate token pools
        self.index_pool = [f"INDEX_{i}" for i in range(index_token_pool_size)]
        self.particle_index_pool = [f"PINDEX_{i}" for i in range(index_token_pool_size)]

        # Regular expression patterns for token replacement
        self.pattern_particle = re.compile(r'(?P<prefix>\b(?:\w+_)?)?(?P<target>[ijkl]_\d+\b)')
        self.pattern_num_123 = re.compile(r'\b(?![psijkl]_)(?!MOMENTUM_)(?!\w+_\w+_)\w+_\d+\b')
        self.pattern_index = re.compile(r'\b\w+_\w+_\d{2,}\b')
        self.pattern_special = re.compile(r'\b\w+_+\w+\b\\')
        self.pattern_underscore_curly = re.compile(r'\b[\w]+(?:_[\w]+)*_{')
        self.pattern_mx = re.compile(r'\bm_\w+\b')
        self.pattern_mass = re.compile(r'\b\w+_\w\b')
        self.pattern_s = re.compile(r'\b\w+_\d{2,}\b')

        self.special_symbols = special_symbols
        self.UNK_IDX = UNK_IDX
        self.to_replace = to_replace

    @staticmethod
    def remove_whitespace(expression):
        """Remove all forms of whitespace from the expression."""
        return re.sub(r'\s+', '', expression)

    @staticmethod
    def split_expression(expression):
        """Split the expression by space delimiter."""
        return re.split(r' ', expression)

    def build_tgt_vocab(self):
        """Build set of unique tokens for target sequences."""
        vocab_set = set()
        for eqn in tqdm(self.sqamps, desc='Processing target vocab'):
            vocab_set.update(self.tgt_tokenize(eqn))
        return sorted(vocab_set)
    
    def build_src_vocab(self):
        """Build set of unique tokens for source sequences."""
        vocab_set = set()
        for diag in tqdm(self.amps, desc='Processing source vocab'):
            vocab_set.update(self.src_tokenize(diag))
        return sorted(vocab_set)

    def tgt_replace(self, sqampl):
        """Replace momentum terms."""
        sqampl = self.remove_whitespace(sqampl)
        sqampl = re.sub(r'p_(\d+)', r'MOMENTUM_\1', sqampl)
        sqampl = re.sub(r's_(\d+)', r'MOMENTUM_\1', sqampl)

        return sqampl
        
    def src_replace(self, ampl):
        """Replace indexed and momentum variables with tokenized equivalents."""
        ampl = self.remove_whitespace(ampl)

        # Pools for replacement
        index_pool = iter(self.index_pool)
        particle_index_pool = iter(self.particle_index_pool)

        ampl = ampl.replace('\\\\', '\\').replace('\\', r' \ ').replace('%', ' % ')
        ampl = re.sub(r'p_(\d+)', r'MOMENTUM_\1', ampl)
        ampl = re.sub(r's_(\d+)', r'MOMENTUM_\1', ampl)

        def get_unique_matches(pattern):
            """Extract ordered unique string matches using a compiled pattern."""
            return list(OrderedDict.fromkeys(pattern.findall(ampl)))

        def map_and_replace(matches, pool, pool_name):
            """Replace matches using the given pool."""
            nonlocal ampl
            try:
                mapping = {match: next(pool) for match in matches}
            except StopIteration:
                raise RuntimeError(
                    f"{pool_name} exhausted. Increase the size of the {pool_name}."
                )

            for key in sorted(mapping.keys(), key=len, reverse=True):
                ampl = ampl.replace(key, mapping[key])

        def replace_particle_tokens():
            """Replace particle tokens using named group 'target'."""
            nonlocal ampl
            matches = list(OrderedDict.fromkeys(
                m.group('target')
                for m in sorted(self.pattern_particle.finditer(ampl), key=lambda m: m.start())
            ))

            try:
                mapping = {m: next(particle_index_pool) for m in matches}
            except StopIteration:
                raise RuntimeError(
                    "particle_index_pool exhausted. Increase the size of the particle_index_pool."
                )

            for key in sorted(mapping.keys(), key=len, reverse=True):
                ampl = ampl.replace(key, mapping[key])

        map_and_replace(get_unique_matches(self.pattern_num_123), index_pool, "index_pool")
        replace_particle_tokens()

        return ampl

    def src_tokenize(self, ampl):
        """Tokenize source expression, optionally applying replacements."""
        ampl = self.src_replace(ampl) if self.to_replace else ampl
        ampl = self.remove_whitespace(ampl)
        ampl = ampl.replace('\\\\', '\\').replace('\\', r' \ ').replace('%', ' % ')
        ampl = ampl.replace("(*)", " CONJ ")
        ampl = ampl.replace("(theta_W)", "_theta_W")

        for pattern in [self.pattern_underscore_curly, self.pattern_mx]:
            ampl = pattern.sub(lambda match: f' {match.group(0)} ', ampl)
        
        for symbol in self.special_symbols:
            ampl = ampl.replace(symbol, f" {symbol} ")
        
        for symbol in ['/', '+', '-', '*', ',', '^', '%', '}', '(', ')']:
            ampl = ampl.replace(symbol, f' {symbol} ')

        ampl = ampl.replace("_PINDEX", "_ PINDEX").replace("_INDEX", "_ INDEX")
        ampl = ampl.replace("reg_prop", " reg_prop ")
        ampl = re.sub(r' {2,}', ' ', ampl)

        return [token for token in self.split_expression(ampl) if token]

    def tgt_tokenize(self, sqampl):
        """Tokenize target expression."""
        sqampl = self.remove_whitespace(sqampl)
        sqampl = self.tgt_replace(sqampl) if self.to_replace else sqampl
        sqampl = sqampl.replace("(theta_W)", "_theta_W")
        
        for symbol in self.special_symbols:
            sqampl = sqampl.replace(symbol, f" {symbol} ")
        
        for symbol in ['/', '+', '-', '*', ',', '^', '%', '}', '(', ')']:
            sqampl = sqampl.replace(symbol, f' {symbol} ')

        for pattern in [self.pattern_mass, self.pattern_s]:
            sqampl = pattern.sub(lambda match: f' {match.group(0)} ', sqampl)

        sqampl = sqampl.replace("reg_prop", " reg_prop ")
        sqampl = re.sub(r' {2,}', ' ', sqampl)

        return [token for token in self.split_expression(sqampl) if token]




class Vocab:
    def __init__(self, tokens, special_symbols, bos_idx, pad_idx, eos_idx, unk_idx, sep_idx, term_idx):
        """
        Initializes the vocabulary.

        Args:
            tokens (Iterable[str]): Collection of tokens to include in the vocabulary.
            special_symbols (List[str]): List of special tokens like <PAD>, <UNK>, etc.
            *_idx (int): Expected index for the respective special token.
        """
        tokens = list(tokens)
        for i in range(len(special_symbols)):
            try:
                tokens.remove(special_symbols[i])
            except:
                pass
        
        self.token_list = special_symbols + tokens

        self.token_to_idx = {token: idx for idx, token in enumerate(self.token_list)}
        
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        # Assertions to verify special symbol positions
        assert self.token_to_idx[special_symbols[unk_idx]] == unk_idx, "UNK index mismatch"
        assert self.token_to_idx[special_symbols[pad_idx]] == pad_idx, "PAD index mismatch"
        assert self.token_to_idx[special_symbols[bos_idx]] == bos_idx, "BOS index mismatch"
        assert self.token_to_idx[special_symbols[eos_idx]] == eos_idx, "EOS index mismatch"
        assert self.token_to_idx[special_symbols[sep_idx]] == sep_idx, "SEP index mismatch"
        for i in range(sep_idx+1,len(special_symbols)):
            assert self.token_to_idx[special_symbols[i]] == term_idx[i - (sep_idx+1)], "TERM index mismatch"

        # Store special token values and indices
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.sep_idx = sep_idx
        self.term_idx = term_idx

        self.unk_tok = special_symbols[unk_idx]
        self.pad_tok = special_symbols[pad_idx]
        self.bos_tok = special_symbols[bos_idx]
        self.eos_tok = special_symbols[eos_idx]
        self.sep_tok = special_symbols[sep_idx]

        self.special_indices = set(self.token_to_idx[sym] for sym in special_symbols)


    def encode(self, tokens):
        """Convert a list of tokens to their corresponding indices."""
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices, include_special_tokens=True):
        """Convert a list of indices back to their corresponding tokens."""
        if include_special_tokens:
            return [self.idx_to_token.get(idx, self.unk_tok) for idx in indices]
        else:
            return [self.idx_to_token.get(idx, self.unk_tok) for idx in indices if idx not in self.special_indices]

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, item):
        """Access token by index or index by token."""
        if isinstance(item, int):
            return self.idx_to_token.get(item, self.unk_tok)
        return self.token_to_idx.get(item, self.unk_idx)

    def tokens(self):
        return self.token_list

