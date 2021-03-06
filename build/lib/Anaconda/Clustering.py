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
import sys

log = logging.getLogger(__name__)

import unittest

class Clustering():
   pass;


def associateDataWithCenters(data,centers,sqDistFunction=AMath.euclideanSqDist):
   """
   Associates the data vectors with centers. Takes the arguments:
   
   [data] which is a list of vertex information of length N.

   [centers] which is a list of vectors of length N, whom sybolizes the centers of
   the clusters.
   
   The output is a list of tubles of the same length than the number of
   clusters. Each tuble contains a list of indicies, and the information of
   the indicies.
   """
   
   # Using bad slow solution:
   if(data.shape[-1] == centers.shape[-1]):

      output = [([],[]) for i in range(np.prod(centers.shape[0:2]))];
      
      vector_size = data.shape[-1];
      d = data.reshape(-1,vector_size);
      for i in range(d.shape[0]):
         index = sqDistFunction(d[i],centers).argmin();
         (id, data) = output[index];
         id.append(i);
         data.append((d[i,:]));
      return [(id,np.array(o)) for (id,o) in output];

class AgglomerativeClusterAlgorithm:
   """
   The highest class for doing AgglomertiveClustering.
   """
   class AgglomerativeClusterNode:

      def __init__(self,index,vectors,sqDistFunction,A=None,B=None):
          """
          Vectors is a matrix of vectores.
          """
          vectors = AMath.createMatrix(vectors);
          self.cluster = vectors;
          self.size = vectors.shape[0];
          self.index = index;
          if self.size == 1:
             self.avg_dist = 0;
             self.centroid = vectors[0];
             self.cent_dist = 0;
          else:
             self.avg_dist = np.sum(np.sqrt(sqDistFunction(vectors,vectors))) \
                             /(self.size * self.size - self.size);
             self.centroid = np.sum(vectors,0) / self.size;
             self.cent_dist = np.sum(np.sqrt(sqDistFunction(self.centroid,vectors)))\
                              / self.size;
          self.A = A;
          self.B = B;

   def __init__(self):
      self.setSqDistanceMethod(AMath.euclideanSqDist);

   def setClusterFitnessMethod(self,clusterFitFunction):
      """
      Sets the method to determine the distance of two clusters.
      the function needs to be of the form:
       lambda cluster1, cluster2, sqdistfunction: real.

      The more fit the cluster is the less the real number should be.

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
      index = 0;
      for cluster in clusters:
          self._startingClusters.append(\
                self.AgglomerativeClusterNode([index],cluster,self._sqDistFunction));
          index += 1;

   def createDendrogram(self):
      """
      Runs the Agglomerative clustering to create a tree, which the
      user later can query. The original method uses a navive approach to
      clustering. Runningtime O(n^3)..
      """
      C = list(self._startingClusters);

      while len(C) >= 2:
         best_pair = (self._clusterFitFunction(C[0],C[1],self._sqDistFunction),C[0],C[1])
      
         for L in C:
            for K in C:
               if L != K:
                  fitness = self._clusterFitFunction(L,K,self._sqDistFunction)
                  if best_pair[0] <= fitness:
                     best_pair = (fitness, L, K);
               else:
                  # Jump out of the loop because the other compinations will be tested.
                  break;

         a = self.AgglomerativeClusterNode(
            np.concatenate([best_pair[1].index,best_pair[2].index]),
            np.vstack((best_pair[1].cluster,best_pair[2].cluster)),
            self._sqDistFunction,
            best_pair[1],
            best_pair[2]
         );
         C.remove(best_pair[1]);
         C.remove(best_pair[2]);
         
         C.append(a);
      self._dendrogram = C[0];

   def itterateDendrogram(self,function,node=None, layer=0):
      """
      Enables the programmer to itterate over the dendrogram. The function receives
      layer, node.
      """
      if(node == None):
         node = self._dendrogram

      function(layer,node);
      if(node.A != None):
         self.itterateDendrogram(function,node.A, layer +1);
      if(node.B != None):
         self.itterateDendrogram(function,node.B,layer +1);

   def cutDendrogram(self,number):
       if len(self._dendrogram.index) <= number:
           return self._dendrogram.index
       nodes = [self._dendrogram];
       while len(nodes) < number:
         # Find worst node and split
         node = max([n for n in nodes if n.avg_dist != 0],
                  key=lambda x: self._clusterFitFunction(x.A,x.B,
                  self._sqDistFunction))
         nodes.remove(node);
       
         nodes.extend([node.A,node.B]);
         # spilt it
       return nodes;

   def getClusters(self,number):
       return [n.index for n in self.cutDendrogram(number)];
def printDendrogram(layer,node):
   layer_str = ''.join(['| ' for i in range(layer -1 )]) + '+-+ '+ str(layer);
   cluster_str = layer_str + '\n' + str(node.index);
   cluster_str = ''.join(['| ' for i in range(layer)]) + '\n' + \
                 cluster_str.replace('\n','\n' + ''.join(['| ' for i in range(layer+1)])) + '\n' ;
   
   sys.stdout.write(cluster_str);

   
   
class ClusteringTester(unittest.TestCase):
   
   def setUp(self):
      pass;
   
   def testAssociateDataWithCenters(self):
      data = np.array([[1,0],[0,1]]);
      centers = np.array([[[0,0],[0,1]],[[1,0],[1,1]]])
      o = associateDataWithCenters(data,centers,AMath.euclideanSqDist);
      self.assertEqual(len(o),4);
      np.testing.assert_equal(o[1][0],[1]);
      np.testing.assert_equal(o[2][0],[0]);

   def testAgglomerativeClusterAlgorithm(self):
      a = AgglomerativeClusterAlgorithm()
      a.setClusterFitnessMethod(lambda x, y, dist: dist(x.centroid,y.centroid));
      a.setStartingClusters([np.array([1,2,3]),np.array([1,3,4]),np.array([13,1,3]),np.array([-2,2,-4])])
      a.createDendrogram();
      a.itterateDendrogram(printDendrogram);

      label = dict();

      def addToLabel(layer,node):
         if not layer in label:
            label[layer] = list(); 
         label[layer].append(node.cluster);

      a.itterateDendrogram(addToLabel);

      print label;
      
if __name__ == "__main__":
   logging.basicConfig(level=logging.DEBUG);
   unittest.main();
