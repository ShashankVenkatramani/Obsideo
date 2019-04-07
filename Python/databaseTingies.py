import qrcode
import pyrebase

import os
import time

import pandas as pd
import numpy as np
import shutil

from Blockchain import Block, Chain

""" configuration """
config = {
  "apiKey": "AIzaSyBBcqE9246sRQ6sezlS5DDFzqzza0zpCSE",
  "authDomain": "com.cvenkatramani.Matador",
  "databaseURL": "https://blockchainbank-e4f33.firebaseio.com",
  "storageBucket": "blockchainbank-e4f33.appspot.com"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

""" loop itself"""

userResults = db.child("requests").get().val()
print(userResults)

if (len(userResults) not == 0):
    chain = Chain()
    while (len(userResults)  >= 3):
        """
        One blockchain block, every set of 3 transactions is the data
        Each child is a uid1, uid2, and amount, separated by semicolons.
        Mine the blockchain block, wait until first 4 digits are 0.
        """
        # Take the subset of user results from 0 to 2, then cut it out.
        currSegment = userResults[0:3]
        userResults = userResults[3:]
        # Parse each line of the user results
        # Put the three transaction arras into a data rray for the entire block
        # For each line, split by ';', then put it in a transaction array
        data = []
        for transaction in currSegment:
            data.append(transaction.split(";"))

        block = Block(chain._get_last_block(), data)
        chain.add_block(_find_nonce(block)) 
