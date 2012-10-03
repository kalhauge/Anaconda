from collections import deque
from bitarray import bitarray;
from bitarray import bitdiff;

import unittest
import logging

log = logging.getLogger(__name__)

"""
Name: AssociationMining
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk
Date : 7th August - 2012

AAssociationMining is a class which purpus is to find assosiations in a bitvector. 
Assoiations is found by compareing the vektors. The implementaion uses a deque to
extact the solutions one by one trying to 

For more information:
http://en.wikipedia.org/wiki/Association_rule_learning

This implementation uses the BitArray Libary which can be found under 
libs/bitarray-0.8.0 where the LICENSE also is located.

"""
 


class AAssociationMining (object):
   
   def __init__(self):
      self._minimumSupport = 0.2;
      self._minimumConfidence = 0.8;
      self._testQueue = deque();
      
   def setVectors(self,vectors):
      log.info("vectors set to {0} x {1} bitarrays".format(len(vectors),len(vectors[0])))
      self._vectors = [bitarray(vector) for vector in vectors];
      
   def setMinimumSupport(self,minimumSupport):
      log.info("minimumSupport set to " + str(minimumSupport));
      self._minimumSupport = minimumSupport;
      
   def setMinimumConfidence(self,minimumConfidence):
      log.info("minimumConfidence set to " + str(minimumConfidence));
      self._minimumConfidence = minimumConfidence;
   
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
      tuble of (vector, differnce, confidence, apperances);
      """
      # Create an 0 vector to use as first itteration.
      empty = bitarray('0')*len(self._vectors[0]);
      
      # The Tuble represents (vector, count) where count is the
      # number of times the vector appers in the vectors.
      tuble = (empty,len(self._vectors));
      
      # Then create a test queue containing the tuble
      self._testQueue = deque([tuble]);
      
      support = (self._minimumSupport * len(self._vectors));
      
      while len(self._testQueue) > 0:
         (vector, count) = self._testQueue.popleft();
         
         #Iterate over all the posible extensions.
         for x in range(len(self._vectors[0])):
            
            if vector[x] : continue;
            
            extension = bitarray(vector);
            extension[x] = True;
            
            appearances = self._countAppearances(extension);
            
            islarger = appearances >= support;
            
            # if the apperances is lower that the minimumSupport,
            # then continue, else add the extension to the self._testQueue
            if islarger:
               self._testQueue.append((extension,appearances));
            
            # If the confidence is higher than the minimumConfidence,
            # then yield the vector extention pair as a solution.
            confidence = float(appearances)/ count;
            if confidence >= self._minimumConfidence:
               yield (vector.tolist(),x,confidence,appearances);
            
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
      print "start";
      solutions = list(self.mapper.getSolutions());
      print "done";
      self.assertEqual(1,len(solutions));      
      
if __name__ == "__main__":
   logging.basicConfig(level=logging.INFO);
   suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
   unittest.TextTestRunner(verbosity=2).run(suite)
