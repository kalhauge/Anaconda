"""
Name: AssociationMining
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk
Date : 7th August - 2012

AAssociationMining is a class which purpus is to find assosiations in a
bitvector.  Assoiations is found by compareing the vektors. The
implementaion uses a deque to extact the solutions one by one trying to 

For more information:
http://en.wikipedia.org/wiki/Association_rule_learning

This implementation uses the BitArray Libary which can be found under 
libs/bitarray-0.8.0 where the LICENSE also is located.

"""

from collections import deque
from bitarray import bitarray;
from bitarray import bitdiff;
from pprint import pprint as pp


import unittest
import logging

log = logging.getLogger(__name__)


  
class AAssociationMining (object): 

    """ 
    AssociationMining is the class that
    enable you to mine in data, the data must be supplied in binary vectors of
    the same size.

    It is possible to denote the starting position, but if it is not set the
    search for associations will continue through the entire tree.

    To minimize the search 
   
    """
   
    def __init__(self):
        self._minimumSupport = 0.2;
        self._minimumConfidence = 0.8;
        self._testQueue = deque();
        self._searchStart = [];
      
    def setVectors(self,vectors):
        log.debug("vectors set to {0} x {1} bitarrays"\
              .format(len(vectors),len(vectors[0])))
        self._vectors = [bitarray(vector) for vector in vectors];
      
    def setMinimumSupport(self,minimumSupport):
        log.debug("minimumSupport set to " + str(minimumSupport));
        self._minimumSupport = minimumSupport;
      
    def setMinimumConfidence(self,minimumConfidence):
        log.debug("minimumConfidence set to " + str(minimumConfidence));
        self._minimumConfidence = minimumConfidence;

    def addSearchStart(self,indices):
        """
        Add a Search Start position to the group to make the computations,
        more efficient. 
        """
        log.debug("searchStart added indices: " + str(indices));
        self._searchStart.append(indices);
   
    def _countAppearances(self,vector):
        """
        Counts the appearances of the vector in the vectors. For a
        vector appears if vector & test == vector
        """
        count = 0;
        for test in self._vectors:
         if vector & test == vector:
            count += 1;
        return count;
   
    def getSolutions(self):
        """
        Yields the solutions as they are found. solutions is yield like a
        tuble of (vector, rule, confidence, apperances);
        """
        if len(self._searchStart) == 0: 
            # Create an 0 vector to use as first iteration.
            empty = bitarray('0')*len(self._vectors[0]);
            # The Tuble represents (vector, count) where count is the
            # number of times the vector appers in the vectors.
            tuble = (empty,len(self._vectors));
      
            # Then create a test queue containing the tuble
            self._testQueue = deque([tuble]);
        else:
            for start in self._searchStart:
                s = bitarray('0')*len(self._vectors[0]);
            for index in start:
                s[index] = True;
            self._testQueue.append((s,self._countAppearances(s)));
        support = (self._minimumSupport * len(self._vectors));
      
        while len(self._testQueue) > 0:
            (vector, count) = self._testQueue.popleft();
         
            #Iterate over all the posible extensions.
            for x in range(len(self._vectors[0])):
            
                if vector[x] : continue;
            
                extension = bitarray(vector);
                extension[x] = True;
            
                appearances = self._countAppearances(extension);
            
                islarger = appearances >= support and appearances != 0;
            
                # if the apperances is higher that the minimumSupport,
                # else add the extension to the self._testQueue
                if islarger:
                    # Do only calculate on it if the vectors highest check
                    # is lower that the new check.
                    if not vector[x:-1].any() and len(vector[x:-1]) > 0:
                        self._testQueue.append((extension,appearances));
            
                    # If the confidence is higher than the minimumConfidence,
                    # then yield the vector extention pair as a solution.
                    confidence = float(appearances)/ count;
                    if confidence >= self._minimumConfidence:
                        yield (vector.tolist(),x,confidence,appearances);
            
            #If done make sure its not calculated again.
            

    def getTestQueueLength(self):
        return len(self._testQueue)

class AAssociationMapppingTester (unittest.TestCase):
   
    def setUp(self):
        self.mapper = AAssociationMining();
        self.test = [
         [True, True, False],
         [True,False,False]];
        self.mapper.setVectors(self.test);
      
    def testSetVectors(self):
        self.assertEqual(self.mapper._vectors,[bitarray('110'),bitarray('100')]);

    def testCountAppearances(self):
        count = self.mapper._countAppearances(bitarray('100'));
        self.assertEqual(2,count);
      
        count = self.mapper._countAppearances(bitarray('110'));
        self.assertEqual(1,count);

    def testGetSolutions(self):
        largeTest = [
         [0,0,1,1,0,1],
         [0,1,0,1,0,1],
         [1,0,0,1,0,1],
         [1,1,1,0,1,1],
         [0,1,1,0,0,0],
         [0,1,0,1,0,0]
        ];
        self.mapper.setVectors(largeTest);
        solutions = list(self.mapper.getSolutions());
        self.assertEqual(1,len(solutions));
      
    def testSearchStart(self):
        largeTest = [
         [0,0,1,1,0,1],
         [0,1,0,1,0,1],
         [1,0,0,1,0,1],
         [1,1,1,0,1,1],
         [0,1,1,0,0,0],
         [0,1,0,1,0,0]
        ];
        self.mapper.setVectors(largeTest);
        self.mapper.addSearchStart([1,2]);
        self.mapper.setMinimumSupport(0);
        self.mapper.setMinimumConfidence(0);
        solutions = list(self.mapper.getSolutions());
        pp(solutions)
        self.assertEqual(5,len(solutions));
   

    def testContainAllSolutions(self):
        largeTest = [[1,1,1,1,1,1]] * 100;
        self.mapper.setVectors(largeTest);
        
        solutions = list(self.mapper.getSolutions());
        def flatten(entry):
            entry[0][entry[1]] = True;
            return tuple(entry[0]);
        entries = set(map(flatten,solutions));  
        self.assertEqual(len(entries),2**len(largeTest[0]) - 1);
    def testContainDoubles(self):
        largeTest = [[1,1,1,1]] * 100;
        self.mapper.setVectors(largeTest);
        
        solutions = list(self.mapper.getSolutions());
        tuples = set([(tuple(m[0]),m[1],m[2],m[3]) for m in solutions])
        self.assertEqual(len(tuples),len(solutions)); 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO);
    unittest.main();
