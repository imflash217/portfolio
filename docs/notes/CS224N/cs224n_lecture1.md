# Lecture `#1`

## Word Meaning

???+ note "NLTK example: synonyms of word `good`"

    ```python
    --8<-- "../CS224N/01_word_meaning.py"
    ```

    ```
    ❯ python 01_word_meaning.py
    noun: good
    noun: goodgoodness
    noun: goodgoodness
    noun: commoditytrade_goodgood
    adj: good
    adj (s): fullgood
    adj: good
    adj (s): estimablegoodhonorablerespectable
    adj (s): beneficialgood
    adj (s): good
    adj (s): goodjustupright
    adj (s): adeptexpertgoodpracticedproficientskillfulskilful
    adj (s): good
    adj (s): deargoodnear
    adj (s): dependablegoodsafesecure
    adj (s): goodrightripe
    adj (s): goodwell
    adj (s): effectivegoodin_effectin_force
    adj (s): good
    adj (s): goodserious
    adj (s): goodsound
    adj (s): goodsalutary
    adj (s): goodhonest
    adj (s): goodundecomposedunspoiledunspoilt
    adj (s): good
    adv: wellgood
    adv: thoroughlysoundlygood
    ```
???+ bug "Problems with toolkits like `WordNet`"
    - [x] Great as a resource but missing `nuance`
      - [ ] Example: `proficent` is listed as a synonym for `good`; but it is correct only in some contexts.
    - [x] Missing new meaning of words
      - [ ] Example: new slang words etc like `wicked`, `badass`, `nifty` etc.
      - [ ] IMPOSSIBLE to keep up-to-date
    - [x] Very Subjective
    - [x] Requires human labour to curate and maintain
    - [x] Can't compute accurate word-similarity.










<!-- ############################################################################################################ -->
???+ quote "Author Disclaimer"
    `Author: Vinay Kumar (@imflash217)`

    `Date: 02/February/2021`

    The contents of this article were originally published at the references below. I have assembled it for my own understanding. Feel free to reuse and tag along the references. :+1:

## `References`
[^1]:

<!-- ############################################################################################################ -->