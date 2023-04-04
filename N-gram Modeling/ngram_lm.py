import math, random

'''
author: Caleb L'Italien
latest version: 01/20/2023
I affirm that I have carried out my academic endeavors with full
academic honesty. [Caleb L'Italien]
'''

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']


def start_pad(c):
    ''' Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 '''
    return '~' * c


def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the character '''
    ngram_list = list()
    text = start_pad(c) + text
    index = c
    while index <= len(text) - 1:
        prev = text[index - c: index]
        tuple = prev, text[index]

        ngram_list.append(tuple)
        index += 1
    return ngram_list


def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the entire text
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


def create_ngram_model_lines(model_class, path, c=2, k=0):
    '''Creates and returns a new n-gram model trained line by line on the
        text found in the path file. '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model


################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''
    def __init__(self, c, k):
        self._c = c
        self._k = k
        self._vocab = set()
        self._ngrams = {}
        self._context_counts = {}


    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self._vocab


    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for char in text:
            self.get_vocab().add(char)

        new_ngrams = ngrams(self.__get_c(), text)
        for ngram in new_ngrams:
            if ngram in self.__get_ngrams().keys():
                self.__get_ngrams()[ngram] += 1
            else:
                self.__get_ngrams()[ngram] = 1
            if ngram[0] in self.__get_context_counts():
                self.__get_context_counts()[ngram[0]] += 1
            else:
                self.__get_context_counts()[ngram[0]] = 1


    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        matching_char_count = 0
        if context in self.__get_context_counts().keys():
            ngram = context, char
            if ngram in self.__get_ngrams().keys():
                matching_char_count = self.__get_ngrams()[ngram]
            return (matching_char_count + self.__get_k()) / (self.__get_context_counts()[context] + self.__get_k() *
                                                             len(self.__get_vocab()))
        return 1/self.__vocab_size()


    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        vocabulary = sorted(self.__get_vocab())
        probability = 0
        index = 0
        while probability <= r and index < len(vocabulary):
            prob_ngram = self.prob(context, vocabulary[index])
            probability += prob_ngram
            index += 1
        return vocabulary[index - 1]


    def random_text(self, length):
        ''' Returns text of the specified character length based on the n-grams learned by
        this model '''
        random_string = start_pad(self.__get_c())
        for i in range(length):
            random_string += str(self.random_char(random_string[len(random_string) - self.__get_c():]))
        return random_string


    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        ngram_dict = ngrams(self.__get_c(), text)
        probability = 0
        for ngram in ngram_dict:
            prob = self.prob(ngram[0], ngram[1])
            if prob == 0:
                return float('inf')
            probability += math.log(prob)
        return math.exp(-(1 / len(text)) * probability)


    def set_k(self, k):
        ''' Sets the k-smoothing value '''
        self._k = k


    def __get_k(self):
        ''' Returns the k-smoothing value '''
        return self._k


    def __vocab_size(self):
        ''' Returns teh vocab size '''
        return len(self._vocab)


    def __get_vocab(self):
        ''' Returns the vocab set '''
        return self._vocab


    def __get_c(self):
        ''' Returns the n-gram model order '''
        return self._c


    def __get_ngrams(self):
        ''' Returns the dictionary of ngrams learned by this model '''
        return self._ngrams


    def __get_context_counts(self):
        ''' Returns the dictionary of contexts learned by this model '''
        return self._context_counts


################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation and add-k smoothing '''
    def __init__(self, c, k):
        super().__init__(c, k)
        self.__lam_list = []
        self.__create_base_lam_list()


    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self._vocab


    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for char in text:
            self.__get_vocab().add(char)
        new_ngrams = self.__create_ngram_list(text)

        for ngram_order in new_ngrams:
            for ngram in ngram_order:
                if ngram in self.__get_ngrams():
                    self.__get_ngrams()[ngram] += 1
                else:
                    self.__get_ngrams()[ngram] = 1
                if ngram[0] in self.__get_context_counts():
                    self.__get_context_counts()[ngram[0]] += 1
                else:
                    self.__get_context_counts()[ngram[0]] = 1


    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        probability = 0
        for i in range(self._c + 1):
            prob = super().prob(context[len(context) - i:], char) * self.__get_lam_list()[i]
            probability += prob
        return probability


    def set_lambda(self, lam_values):
        ''' Sets the lambda values to a list of new values '''
        self.__lam_list = lam_values


    def __create_base_lam_list(self):
        ''' Sets each of the lambda values equal '''
        for i in range(self.__get_c() + 1):
            self.__lam_list.append(1 / (self.__get_c() + 1))


    def __create_ngram_list(self, text):
        ''' Creates a list of new ngrams '''
        highest_ngram = self.__get_c()
        new_ngrams = []
        while highest_ngram >= 0:
            ngram_order = ngrams(highest_ngram, text)
            highest_ngram -= 1
            new_ngrams.append(ngram_order)
        return new_ngrams


    def __get_k(self):
        ''' Returns the k-smoothing value '''
        return self._k


    def __vocab_size(self):
        ''' Returns the vocab size '''
        return len(self._vocab)


    def __get_vocab(self):
        ''' Returns the vocab set '''
        return self._vocab


    def __get_c(self):
        ''' Returns the n-gram model order '''
        return self._c


    def __get_ngrams(self):
        ''' Returns the dictionary of ngrams learned by this model '''
        return self._ngrams


    def __get_context_counts(self):
        ''' Returns the dictionary of contexts learned by this model '''
        return self._context_counts

    def __get_lam_list(self):
        ''' Returns the list of lambda values '''
        return self.__lam_list


################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.

def base_tests_NgramModel():
    base_test_ngrams()
    base_test_get_vocab_and_update()
    base_test_prob()
    base_test_random_char()
    base_test_random_text()
    base_test_perplexity()
    base_test_perplexity()
    base_test_add_k()
    base_test_NgramModel()

def base_tests_NgramModelWithInterpolation():
    base_test_get_vocab_and_update_int()
    base_test_prob_int()
    base_test_add_k_int()

def base_test_ngrams():
    print('ngrams')
    print(ngrams(1, 'abc') == [('~', 'a'), ('a', 'b'), ('b', 'c')])
    print(ngrams(2, 'abc') == [('~~', 'a'), ('~a', 'b'), ('ab', 'c')])
    print('---')

def test_ngrams(c, text):
    print(ngrams(c, text))

def base_test_get_vocab_and_update():
    print('get_vocab and update')
    m = NgramModel(1,0)
    m.update('abab')
    print(m.get_vocab() == {'a', 'b'})
    print('---')

def test_get_vocab_and_update(c, k, text):
    m = NgramModel(c, k)
    m.update(text)
    print(m.get_vocab())

def base_test_prob():
    print('prob')
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.prob('a', 'b') == 1.0)
    print(m.prob('~', 'c') == 0.0)
    print(m.prob('b', 'c') == 0.5)
    print('---')

def test_prob(c, k, text, context, char):
    m = NgramModel(c, k)
    m.update(text)
    print(m.prob(context, char))

def base_test_random_char():
    print('random_char')
    m = NgramModel(0,0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print([m.random_char('') for i in range(25)] == ['a', 'c', 'c', 'a', 'b', 'b', 'b', 'c', 'a', 'a', 'c', 'b', 'c',
                                                     'a', 'b', 'b', 'a', 'd', 'd', 'a', 'a', 'b', 'd', 'b', 'a'])
    print('---')

def test_random_char(c, k, text):
    m = NgramModel(c, k)
    m.update(text)
    print([m.random_char('') for i in range(25)])

def base_test_random_text():
    print('random_text')
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print(m.random_text(25) == '~abcdbabcdabababcdddabcdba')
    print('---')

def test_random_text(c, k, text):
    m = NgramModel(c, k)
    m.update(text)
    print(m.random_text(25))

def base_test_perplexity():
    print('perplexity')
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.perplexity('abcd') == 1.189207115002721)
    print(m.perplexity('abca') == math.inf)
    print(m.perplexity('abcda') == 1.515716566510398)
    print('---')

def test_perplexity(c, k, text, perplexity_text):
    m = NgramModel(c, k)
    m.update(text)
    print(m.perplexity(perplexity_text) == 1.189207115002721)

def base_test_add_k():
    print('add-k')
    m = NgramModel(1, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob('a', 'a') == 0.14285714285714285)
    print(m.prob('a', 'b') == 0.5714285714285714)
    print(m.prob('c', 'd') == 0.4)
    print(m.prob('d', 'a') == 0.25)
    print('---')

def test_add_k(c, k, text, context, char):
    m = NgramModel(c, k)
    m.update(text)
    print(m.prob(context, char))

def base_test_NgramModel():
    print('NgramModel')
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    print(m.random_text(250))
    print('---')
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    print(m.random_text(250))
    print('---')
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    print(m.random_text(250))
    print('---')
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    print(m.random_text(250))
    print('---')

def test_NgramModel(input_text, c, k):
    m = create_ngram_model(NgramModel, input_text, c, k)
    print(m.random_text(1000))

def base_test_get_vocab_and_update_int():
    print('get_vocab and update - int')
    m = NgramModelWithInterpolation(1,0)
    m.update('abab')
    print(m.get_vocab() == {'a', 'b'})
    print('---')

def test_get_vocab_and_update_int(c, k, text):
    m = NgramModelWithInterpolation(c, k)
    m.update(text)
    print(m.get_vocab())

def base_test_prob_int():
    print('prob - int')
    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    print(m.prob('a', 'a') == 0.25)
    print(m.prob('a', 'b') == 0.75)
    print('---')

def test_prob_int(c, k, text, context, char):
    m = NgramModelWithInterpolation(c, k)
    m.update(text)
    print(m.prob(context, char))

def base_test_add_k_int():
    print('add-k - int')
    m = NgramModelWithInterpolation(2, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob('~a', 'b'))
    print(m.prob('ba', 'b'))
    print(m.prob('~c', 'd'))
    print(m.prob('bc', 'd'))
    print('---')

def test_add_k_int(c, k, text, context, char):
    m = NgramModelWithInterpolation(c, k)
    m.update(text)
    print(m.prob(context, char))

def test_lam_values(c, k, text, context, char, lam_list):
    m = NgramModelWithInterpolation(c, k)
    m.update(text)
    m.set_lambda(lam_list)
    print(m.prob(context, char))