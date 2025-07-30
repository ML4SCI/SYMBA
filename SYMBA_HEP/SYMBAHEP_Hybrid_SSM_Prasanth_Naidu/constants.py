# Special token indices
BOS_IDX = 0  # Beginning of Sequence
PAD_IDX = 1  # Padding
EOS_IDX = 2  # End of Sequence
UNK_IDX = 3  # Unknown Token
SEP_IDX = 4  # Separator Token

T_IDX = [i for i in range(5,25)]


# Special token symbols
SPL_TERM_SYMBOLS = [f'<T{i}>' for i in range(20)]
SPECIAL_SYMBOLS = ['<BOS>', '<PAD>', '<EOS>', '<UNK>', '<SEP>'] + SPL_TERM_SYMBOLS
