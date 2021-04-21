#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:49:47 2021

@author: From Evans GH
"""

"""Module to web scrap data from stackoverflow pages and assigning to relevant types."""
from csv import reader
from bs4 import BeautifulSoup as bs
import requests

LINK = "https://stackoverflow.com/questions/26009095/"


def get_soup(link):
    """Get the parsed html file as a soup file"""
    page = requests.get(link)
    soup = bs(page.text, 'html.parser')
    return soup


def get_question(soup):
    """Returns the question codes and other text as a dictionary"""
    question = soup.find("div", class_='question')
    question_code = question.select("code")
    question_text = question.select("p")
    return {"question_code": question_code, "question_text": question_text}


def get_accepted_answer(soup):
    """Returns the answers code and rest of the text as a dictionary"""
    answer = soup.find("div", class_='answer accepted-answer')
    answer_code = answer.select("code")
    answer_text = answer.select("p")
    return {"answer_code": answer_code, "answer_text": answer_text}


def get_everything_else(soup):
    """Returns everything else (Codes and text) as a dictionary"""
    everything_else = soup.find_all('div', class_="answer")[1:]
    everything_else_code = everything_else.select("code")
    everything_else_text = everything_else.select("p")
    return {"everything_else_code":everything_else_code,
            "everything_else_text": everything_else_text}

#load data fxn
def load_data(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def main():
    """Main function"""
    relevant_ids = load_data('relevant_ids.csv')
    irrelevant_ids = load_data('irrelevant_ids.csv')
    for entry in irrelevant_ids:
        print(entry[1])
        LINK = 'https://stackoverflow.com/questions/%s' % (entry[1])
        print(LINK)
    soup = get_soup(LINK)
    ret = get_question(soup=soup)
    print(ret["question_code"])

if __name__ == "__main__":
    main()


#TODO
"""
    REMOVE THE TAGS FROM THE OUTPUT FROM THE INPUT 
    
"""