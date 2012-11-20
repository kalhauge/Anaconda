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


def associateDataWithCenters(data,centers,sqDistFunction=AMath.euclideanSqDist):
   # Using bad slow solution:
   if(data.shape[-1] == centers.shape[-1]):
      output = dict();
      
      vector_size = data.shape[-1];
      d = data.reshape(-1,vector_size);
      for i in range(d.shape[0]):
         index = np.unravel_index(sqDistFunction(d[i],centers).argmin(),centers.shape[0:-1]);
         if(not index in output):
            output[index] = [];
         output[index].append(i);
      return output;

class AgglomerativeClusterAlgorithm:
   """
   The highest class for doing AgglomertiveClustering.
   """
   class AgglomerativeClusterNode:
      def __init__(self,vectors,sqDistFunction,A=None,B=None):
          """
          Vectors is a matrix of vectores.
          """
          self.cluster = cluster;
          self.size = vector.shape[0];
          self.avg_dist = np.sum(np.sqrt(sqDistFunction(vector,vector))) / (self.size * self.size - self.size)
          self.centroid = np.sum(vector) / self.size;
          self.cent_dist = np.sum(np.sqrt(sqDistFunction(self.centroid,vector))) / self.size
          self.A = A;
          self.B = B;

   def __init__(self):
      setSqDistanceMethod(AMath.euclidianSqDist);

   def setClusterFitnessMethod(self,clusterFitFunction):
      """
      Sets the method to determine the distance of two clusters.
      the function needs to be of the form:
       lambda cluster1, cluster2, sqdistfunction: real.
      """
      self._clusterFitFunction = clusterFitFunction;
   
   def setSqDistanceMethod(self,sqDistFunction):
      """
      Sets the method for calculating the Squard Distance between two
      methods, per default is it AMath's euclidianSqDist.
      """
      self._sqDistFunction = sqDistFunction;

   def setStartingClusters(self,clusters):
      """
      Sets the begining clusters for the method. Clusters is an
      iterative array of vectors. The sqDistanceMethod, needs to be
      set before calling this method. 
      """
      self._startingClusters = [];
      for cluster in clusters:
          self._startingClusters.append(AgglomerativeClusterNode(vectors,self._sqDistFunction));

   def createDendrogram(self):
      """
      Runs the Agglomerative clustering to create a tree, which the
      user later can query. The original method uses a navive approach to
      clustering. Runningtime O(n^3)..
      """
      C = list(self._startingClusters);

      while len(C) >= 2:
      best_pair = (self._clusterFitFunction(C[0],C[1]),C[0],C[1])
      
      for L in C:
         for K in C:
             if L != K:
                
                 
         
   
   
   
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