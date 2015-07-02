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

disambig_symbols=`add_lex_disambig.pl lexicon.txt lexicon_disambig.txt`

for symbol in $disambig_symbols
do
    last_number=`tail -n 1 characters.txt | grep -oP '(?<=. )[0-9]+'`
    echo \#$symbol $(($last_number + 1)) >> characters.txt
    echo $(($last_number + 1)) 0 >> relabel.txt
done

make_lexicon_fst.pl                            \
    lexicon_disambig.txt                       |\
    fstcompile                                   \
        --isymbols=characters.txt                 \
        --osymbols=words.txt                       \
        --keep_isymbols=false --keep_osymbols=false|\
    fstaddselfloops  \
        "echo `grep -oP '(?<=#0 )[0-9]+' characters.txt` |" \
        "echo `grep -oP '(?<=#0 )[0-9]+' words.txt` |"         | \
    fstarcsort --sort_type=olabel                  |  \
    fstrelabel --relabel_isymbols=relabel.txt          \
    > L_disambig.fst

fsttablecompose L_disambig.fst G.fst         |\
    fstdeterminizestar --use-log=true        | \
    fstminimizeencoded > LG.fst
