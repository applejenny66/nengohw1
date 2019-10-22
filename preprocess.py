# -*- coding: utf-8 -*
# preprocess.py

import nltk
from nltk.tokenize import word_tokenize
import csv

def downloadscript():
    nltk.download()

def paragraph():
    count = 0
    paragraph = ""
    with open('./paragraph.csv') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            print(row)
            paragraph = paragraph + str(row)
            count += 1
        print ("count : ", count)
        return (paragraph)

def word_seg(para):
    results = word_tokenize(para)
    return (results)



if __name__ == "__main__":
    #downloadscript()
    paragraph = paragraph()
    #print ("paragraph : ", paragraph)
    results = word_seg(paragraph)
    print ("results: ", results)
