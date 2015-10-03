#!/usr/bin/env python
"""
Read an LM in ARPA format wnd write the following files:
- words.txt containing words and their numerical ids
- characters.txt containing characters and their ids
- lexicon.txt spelling out each word into characters
"""

import sys


def main(filename):
    chars = {'<eps>': 0, '<spc>': 1, '#0': 2}
    with open(filename) as f, open("lexicon.txt", "w") as f_lexicon, \
            open("words.txt", "w") as f_words, \
            open("characters.txt", "w") as f_characters:
        for char, ind in chars.items():
            f_characters.write("{} {}\n".format(char, ind))

        f_words.write("<eps> 0\n<UNK> 1\n</s> 2\n<s> 3\n<spc> 4\n#0 5\n")
        n_words = 6

        # Step 1: Wait until we read \data
        for line in f:
            if line.startswith("\\data"):
                break

        # Step 2: Wait until we read \1-gram
        for line in f:
            if line.startswith("\\1-gram"):
                break

        # Step 3: Read all 1-grams
        for line in f:
            if line.strip() == "":
                continue
            if line.startswith('\\2-grams') or line.startswith('\\end'):
                break
            line = line.split()
            if len(line) not in [2, 3]:
                # max robustness to junk, please note that you may or may not
                # have a backoff-weight
                continue
            word = line[1]
            if word.startswith("<") or word.startswith("#"):
                continue
            f_words.write("{} {}\n".format(word.strip(), n_words))
            n_words += 1

            f_lexicon.write("{} {}\n".format(word, " ".join(word)))
            for char in word:
                if char not in chars:
                    chars[char] = len(chars)
                    f_characters.write("{} {}\n".format(char, chars[char]))


if __name__ == '__main__':
    main(sys.argv[1])
