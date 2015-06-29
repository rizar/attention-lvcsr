#!/usr/bin/env python
import sys

def main(filename):
    data_started = False
    one_gram_started = False
    chars = {'<eps>': 0, '<spc>': 1, '#0': 2, '#1': 3}
    with open(filename) as f, open("lexicon.txt", "w") as f_lexicon, \
            open("words.txt", "w") as f_words, \
            open("characters.txt", "w") as f_characters:
        for char, ind in chars.items():
            f_characters.write("{} {}\n".format(char, ind))

        f_words.write("<eps> 0\n<UNK> 1\n</s> 2\n<s> 3\n<spc> 4\n#0 5\n")
        n_words = 6
        for line in f:
            # Step 3: Read all 1-grams
            if one_gram_started:
                if line.strip() == "":
                    break
                _, word, _ = line.split()
                if word.startswith("<") or word.startswith("#"):
                    continue
                f_words.write("{} {}\n".format(word.strip(), n_words))
                n_words += 1

                f_lexicon.write("{} {}\n".format(word, " ".join(word)))
                for char in word:
                    if char not in chars:
                        chars[char] = len(chars)
                        f_characters.write("{} {}\n".format(char, chars[char]))

            # Step 1: Wait until we read \data
            if line.startswith("\\data"):
                data_started = True
            # Step 2: Wait until we read \1-gram
            if line.startswith("\\1-gram") and data_started:
                one_gram_started = True


if __name__ == '__main__':
    main(sys.argv[1])
