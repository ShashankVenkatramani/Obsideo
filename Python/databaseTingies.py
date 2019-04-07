import pyrebase

import os
import time

import pandas as pd
import numpy as np
import shutil
import hashlib

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

if (not len(userResults) == 0):
    chain = Chain()
    first = True
    allHashes = []
    
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
        """
            blockchain array
                block 1
                    previous hash,
                    next hash,
                    nonce
                    data
                        transaction 1, transaction 2, transaction 3
            """
        #current hash is by hashing the entire block after combiniig it with like a ";"s
        nonce = 0
        sum = ""
        for transaction in currSegment:
            sum += str(transaction)
    
        print(sum + str(nonce))
        hash_object = hashlib.sha224((sum + str(nonce)).encode(encoding='UTF-8',errors='strict')).hexdigest()

        while (not str(hash_object)[:4] == "0000"):
            nonce += 1
            print(str(hash_object))
            hash_object = hashlib.sha256((sum + str(nonce)).encode(encoding='UTF-8',errors='strict')).hexdigest()
        
        block = []
        prev_hash = 0
        if (not first):
            prev_hash = allHashes[-1]
        
        allHashes.append(hash_object)


        block.append(prev_hash)
        block.append(hash_object)
        block.append(nonce)
        block.append(data)

        print(db.child("blockchain").get().val())
        
        yeet = db.child("blockchain").get().val()
        yeet.append(block)
        db.child("blockchain").set(yeet)
