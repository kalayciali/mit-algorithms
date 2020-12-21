# Document distance problem
# 3gram implementation with python dict.

# LCS dynamic programming implementation
# I tried with recursive implementation it is tooo slow
# compared to dynamic programming.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import string
import sys
from concurrent import futures


STOP_WORDS = list(stopwords.words('english'))


def readFile(filename):
    """
    Read whole file and return it
    """
    try:
        with open(filename, 'r', errors='ignore') as f:
            whole_doc = f.read()
            # for different languages uncomment below
            whole_doc = bytes(whole_doc, 'UTF-8')
            whole_doc = whole_doc.decode('ascii', 'ignore')
            return whole_doc
    except IOError:
        print("io error occurred")
        sys.exit()

def isCommon(ngram):
    '''
    ngram is 3gram
    remove most common words in order to improve accuracy
    '''
    commonWords = ['THE', 'BE', 'AND', 'OF', 'A', 'IN', 'TO', 'HAVE', 'IT', 'I', 'THAT', 'FOR', 'YOU', 'HE', 'WITH', 'ON', 'DO', 'SAY', 'THIS', 'THEY', 'IS', 'AN', 'AT', 'BUT', 'WE', 'HIS', 'FROM', 'THAT', 'NOT', 'BY', 'SHE', 'OR', 'AS', 'WHAT', 'GO', 'THEIR', 'CAN', 'WHO', 'GET', 'IF', 'WOULD', 'HER', 'ALL', 'MY', 'MAKE', 'ABOUT', 'KNOW', 'WILL', 'AS','UP', 'ONE', 'TIME', 'HAS', 'BEEN', 'THERE', 'YEAR', 'SO', 'THINK', 'WHEN', 'WHICH', 'THEM', 'SOME', 'ME', 'PEOPLE', 'TAKE', 'OUT', 'INTO', 'JUST', 'SEE', 'HIM', 'YOUR', 'COME', 'COULD', 'NOW', 'THAN', 'LIKE', 'OTHER', 'HOW', 'THEN', 'ITS', 'OUR', 'TWO', 'MORE', 'THESE', 'WANT', 'WAY', 'LOOK', 'FIRST', 'ALSO', 'NEW', 'BECAUSE', 'DAY', 'MORE', 'USE', 'NO', 'MAN', 'FIND', 'HERE', 'THING', 'GIVE', 'MANY', 'WELL']
    for word in ngram:
        word = word.upper()
        if word in commonWords:
            return True
    return False

def lineClean(line):
    '''
    Clean line and return as word list
    '''
    translation = line.maketrans(string.punctuation+string.ascii_uppercase," " *len(string.punctuation)+string.ascii_lowercase)
    line = line.translate(translation)
    tokens = word_tokenize(line)
    tokens = [ w for w in tokens if not w in STOP_WORDS ]
    return tokens

def getTokens(doc):
    # get all words from doc
    sentence_list = sent_tokenize(doc)
    trigrams = {}
    STOP_WORDS = set(stopwords.words('english'))
    all_tokens = []
    for line in sentence_list:
        tokens = lineClean(line)
        all_tokens.extend(tokens)
    return all_tokens

def cleanDoc(doc):
    '''
    Sentence by sentence analyze doc
    return 3grams as dict
    '''
    sentence_list = sent_tokenize(doc)
    trigrams = {}
    STOP_WORDS = set(stopwords.words('english'))

    for line in sentence_list:
        tokens = lineClean(line)

        for i in range(len(tokens)-2):
            t = ( tokens[i], tokens[i+1], tokens[i+2] )
            if not isCommon(t):
                if t in trigrams:
                    trigrams[t] += 1
                else:
                    trigrams[t] = 1
    return trigrams


def compareTrigrams(D1, D2):
    '''
    D1 is orig trigrams dict
    D1[(w1, w2, w3)] = freq. of trigram
    D2 is tested for plaguarism
    '''
    same_occur = 0

    for key in D1:
        if key in D2:
            same_occur += 1

    sizeD1 = 0
    for val in D1.itervalues()
        sizeD1 += val

    sizeD2 = 0
    for val in D2.itervalues():
        sizeD2 += val

    jaccord_coeff = same_occur / (sizeD1 + sizeD2)
    containment_measure = same_occur / sizeD2
    print('jaccord_coeff is {:.2f}%'.format(100 * jaccord_coeff))
    print('containment_measure is {:.2f}%'.format(100 * containment_measure))
    return (jaccord_coeff, containment_measure)


def longComSubseq(s1, s2):
    '''
    longest common subsequence dynamic programming implementation
    find LCS words within two lists
    sent1 = [w1, w2, ..]
    '''
    dp = [[None]*(len(s1)+1) for i in range(len(s2)+1)]

    for i in range(len(s2)+1):
        for j in range(len(s1)+1):
            if (i == 0 or j== 0):
                dp[i][j] = 0
            elif s2[i-1] == s1[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[len(s2)][len(s1)]


def LcsWorkerFunc(sent_pair):
    orig_sent, plag_sent = sent_pair
    orig = lineClean(orig_sent)
    plag = lineClean(plag_sent)
    comm_longest = longComSubseq(orig, plag)
    return comm_longest


def LCS(doc1, doc2):
    '''
    for each sentence pair find longComSubseq
    score = sum of max lcs for each plag sentence / len(plag_sentences)
    '''
    # doc1 is orig 
    # doc2 is for test of plaguarism


    max_lcs = 0
    sum_lcs = 0

    sentences_orig = sent_tokenize(doc1)
    sentences_plag = sent_tokenize(doc2)

    for plag in sentences_plag:
        pairs = []
        for orig in sentences_orig:
            # within each plag sentence 
            # check it with orig sentence
            # find max longest for each plag sentence
            # add them up
            pairs.append((orig, plag))

        with futures.ProcessPoolExecutor() as executor:
            comm_longests = executor.map(LcsWorkerFunc, pairs)
        max_lcs = max(comm_longests)
        sum_lcs +=max_lcs
        max_lcs = 0

    score = sum_lcs / len(getTokens(doc2))
    print("LCS score is {:.2f}%".format(100 * score))
    return score

def readAndClean(doc):
    doc = readFile(doc)
    D = cleanDoc(doc)
    return D

def main():
    if len(sys.argv) != 3:
        print("Please use it like this:")
        print("python filename.py orig_file.txt plag_file.txt")
    else:
        print("If it near 0, there is no plagiarism")
        print("If it near 100, the second doc is near copy of first one")
        orig = sys.argv[1]
        plag = sys.argv[2]
        orig_freq_dic = readAndClean(orig)
        plag_freq_dic = readAndClean(plag)
        compareTrigrams(orig_freq_dic, plag_freq_dic)

        # It works slow
        LCS(readFile(orig), readFile(plag))

if __name__ == "__main__":
    main()



