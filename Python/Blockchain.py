import time
from hashlib import sha256

class Block:
    CHECKSUM_LEAD_ZEROS = 4
    NONCE_SYMBOL = 'Z'

    def __init__(self, prev_block, data):
        self._prev_block = prev_block
        self.data = data
        self.checksum = None
        self.nonce = 0
        self.timestamp = time.time()

    @property
    def is_valid(self):
        checksum = self.calculate_checksum()

        return (
            checksum[:self.CHECKSUM_LEAD_ZEROS] == '0' * self.CHECKSUM_LEAD_ZEROS
            and checksum == self.checksum
        )

    def calculate_checksum(self):
        data = '|'.join([
            str(self.timestamp),
            self.data,
            self._prev_block.checksum,
        ])
        data += self.NONCE_SYMBOL * self.nonce

        return sha256(bytes(data, 'utf-8')).hexdigest()

import json

class Chain:

    def __init__(self):
        self._chain = [
            self._get_genesis_block(),
        ]

    def is_valid(self):
        prev_block = self._chain[0]
        for block in self._chain[1:]:
            assert prev_block.checksum == self._prev_block.checksum
            assert block.is_valid()
            prev_block = block

    def add_block(self, data):
        block = Block(self._chain[-1], data)
        block = self._find_nonce(block)
        self._chain.append(block)

        return block

    @staticmethod
    def _get_genesis_block():
        genesis_block = Block(None, None)
        genesis_block.checksum = '00000453880b6f9179c0661bdf8ea06135f1575aa372e0e70a19b04de0d4cbc7'

        return genesis_block

    @staticmethod
    def _get_last_block():
        return self._chain[-1]

    @staticmethod
    def _find_nonce(block):
        beginning = '0' * Block.CHECKSUM_LEAD_ZEROS
        while True:
            checksum = block.calculate_checksum()
            if checksum[:Block.CHECKSUM_LEAD_ZEROS] == beginning:
                break
            block.nonce += 1

        block.checksum = checksum

        return block
