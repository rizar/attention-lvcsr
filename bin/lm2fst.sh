create_lexicon.py $1

cat $1                 | \
    grep -v '<s> <s>'   | \
    grep -v '</s> <s>'   | \
    grep -v '</s> </s>'   | \
    arpa2fst -             | \
    fstprint                | \
    eps2disambig.pl          | \
    s2eps.pl                  | \
    fstcompile                   \
        --isymbols=words.txt      \
        --osymbols=words.txt       \
        --keep_isymbols=false       \
        --keep_osymbols=false      | \
    fstrmepsilon                    | \
    fstarcsort --sort_type=ilabel      \
    > G.fst

add_lex_disambig.pl lexicon.txt lexicon_disambig.txt

make_lexicon_fst.pl                            \
    lexicon_disambig.txt                       |\
    fstcompile                                   \
        --isymbols=characters.txt                 \
        --osymbols=words.txt                       \
        --keep_isymbols=false --keep_osymbols=false|\
    fstaddselfloops  "echo 2 |" "echo 5 |"         | \
    fstarcsort --sort_type=olabel                  |  \
    fstrelabel --relabel_isymbols=relabel.txt          \
    > L_disambig.fst

fsttablecompose L_disambig.fst G.fst         |\
    fstdeterminizestar --use-log=true        | \
    fstminimizeencoded > LG.fst
