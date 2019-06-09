import unittest
import time
from dataframe import EncoderDataFrame

class TimedCase(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

class EncoderDataFrameTest(TimedCase):

    def test_init(self):
        df = EncoderDataFrame()
        df['test1'] = [0,2,3]
        df['test2'] = ['a','b', 'c']

    def test_scramble(self):
        df = 

if __name__ == '__main__':
    unittest.main()
