import csv
import pandas as pd
import re
import shutil
import spacy
from spacy.tokenizer import Tokenizer
from bs4 import BeautifulSoup
import os
import sys
import time

import numpy as np

# Load spacy eng core
nlp = spacy.load("en_core_web_sm")


def cleaner(overview):
  # Clean parenthese from text
  parenthese_pat = r"\([^)]+\)"
  # Clean any HTML tag if exist in the text
  soup = BeautifulSoup(overview, "lxml")
  souped = soup.get_text()
  try:
    bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
  except:
    bom_removed = souped
  # Get all the text as lower
  stripped = re.sub(parenthese_pat, "", bom_removed.lower())
  # remove repeated letters and char
  stripped = re.sub(r"(.)\1{2,}", r"\1", stripped)
  stripped = re.sub(r"\s+", r" ", stripped)
  stripped = re.sub("_", " ", stripped)

  tok = None

  stripped = stripped.replace('"', ' ')
  stripped = stripped.replace('.', ' ')
  stripped = stripped.replace('/', ' ')
  stripped = stripped.replace('-', ' ')
  tokens = nlp(stripped)
  # Use spacy model to remove the Stop word and punctuation from text
  results = [token for i, token in enumerate(tokens) if not token.is_stop and not token.is_punct]
  list_of_strings = [i.text for i in results]
  tok = (" ".join(list_of_strings)).strip().lower()
  # remove the new line and tab (SO description are one line)
  tok = tok.strip('\n')
  tok = tok.strip('\t')
  tok = tok.strip()
  return tok
