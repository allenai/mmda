# word predictors

### summary

Due to PDF formatting, words might be represented as separate tokens after PDF parsing (e.g. `tok-` and `en` instead of `token`).  One way to turn these back into regular words is to undo this partitioning.

1. First we identify candidate tokens for merging.  A good heuristic is one that finds row-ending token with a hyphen `t_1` alongside its candidate merge partner `t_2` beginning the subsequent row.  For more complex documents, rows might be insufficient; we might require information about blocks and their semantic categories to identify candidates (e.g. two-column formats, or injected figures and captions might make make candidate selection tricky.)

    - Strategy: Assume PDF token stream in reading order; Use rows and page-breaks only
    - Strategy: Assume PDF token stream in reading order; Consider all blocks with same category; then use rows and page-breaks

2. Next we consider whether we should merge these candidates into a single word keeping the hyphen, removing the hyphen, or not merging the two at all.

    - Strategy: Check spelling of both segments; if valid words, keep separate; else merge
    - Strategy: Check both segments against dictionary of English words; if valid words, keep separate; else merge

 
### dependencies

1. [SymSpellPy](https://github.com/mammothb/symspellpy) is Python binding for [SymSpell](https://github.com/wolfgarbe/SymSpell), which is a popular method for fast spelling correction.  MIT License. 

