"""
Name: Clustering
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk 
Date : 31th Oktober - 2012

This module has to purpurse to add Clustering functionality to Anaconda. 
The module should be used with numpy. 
"""

import numpy as np
import logging
import AMath

log = logging.getLogger(__name__)

import unittest

class Clustering():
   pass;


def assoiateDataWithCenters(data,centers,sqDistFunction):
   # Using bad slow solution:
   if(data.shape[-1] == centers.shape[-1]):
      output = dict();
      
      vector_size = data.shape[-1];
      d = data.reshape(-1,vector_size);
      for i in range(d.shape[0]):
         index = np.unravel_index(sqDistFunction(d[i],centers).argmin(),centers.shape[0:-1]);
         if(not index in output):
            output[index] = [];
         output[index].append(d[i]);
      return output;

def getKmeansClusters(data):
   pass;
   
   
   
   
class ClusteringTester(unittest.TestCase):
   
   def setUp(self):
      pass;
   
   def testAssoiateDataWithCenters(self):
      data = np.array([[1,0],[0,1]]);
      centers = np.array([[[0,0],[0,1]],[[1,0],[1,1]]])
      o = assoiateDataWithCenters(data,centers,AMath.euclideanSqDist);
      self.assertEqual(len(o),2);
      self.assertTrue((o[(1,0)] == data[0]).all());
      self.assertTrue((o[(0,1)] == data[1]).all());
      print o;
      
if __name__ == "__main__":
   logging.basicConfig(level=logging.DEBUG);
   unittest.main();