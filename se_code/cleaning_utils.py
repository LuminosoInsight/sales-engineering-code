'''
Utilities that might be helpful for data preprocessing.
'''
import re

class RepeatShortener:
    '''
    Tweets sometimes contain strings like "kkkkkk" or "lolololol".  This class
    can be used to reduce the repeated segments so variations on the same theme
    become identical.

    Sample usage:

        In [2]: shortener = RepeatShortener()

        In [3]: shortener('lol')
        Out[3]: 'lol'

        In [4]: shortener('loooooooooooool')
        Out[4]: 'looool'

        In [5]: shortener('kekekekekekekek')
        Out[5]: 'kekekekek'
    '''
    def __init__(self, repeats=4):
        '''
        Constructor.  Optionally pass the number of repeats to collapse to;
        defaults to 4 (from the predecessor "fourshortener").
        '''
        self.reg = re.compile(r'(\S+?)\1{%d,}' % repeats)
        self.repl = r'\1' * repeats

    def __call__(self, s):
        '''
        Return the string with repeats shortened.
        '''
        return self.reg.sub(self.repl, s)
