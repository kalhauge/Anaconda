"""
Name: AMath
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk 
Date : 14th September - 2012

Math is a collection of methods used in math.

"""

import math
import sys
import unittest
import numpy as np

def calculateStatistics(vectors):
   """
   returns number of values pr column, the mean of each colloum
   and the stdartd diviration of each column

   returns (values_pr_coloum,mean,stddiv);
   """
   values_pr_column = (vectors.shape[0] - np.isnan(vectors).sum(0));

   mean = np.nan_to_num(vectors).sum(0)
   mean[values_pr_column != 0] /= values_pr_column[values_pr_column != 0];
   mean[values_pr_column == 0] = np.nan;
   
   sqrdmean = (np.nan_to_num(vectors)**2).sum(0);
   sqrdmean[values_pr_column != 0] /= values_pr_column[values_pr_column != 0];
   sqrdmean[values_pr_column == 0] = np.nan;
   
   stddiv = np.sqrt(sqrdmean - mean**2)

   return (values_pr_column,mean,stddiv);


def normalizeData(grid,between=(0,1)):
   not_nans = ~np.isnan(grid);
   minimum = np.min(grid[not_nans]);
   maximum = np.max(grid[not_nans]);
   res = (grid - minimum) / ( maximum - minimum ) * (between[1] - between[0]) + between[0];
   return res;

   
def gammalog(n):
   return math.log(math.factorial(n-1));

def binomiallog(n,k):
   return gammalog(n+1) - (gammalog(k+ 1) + gammalog(n-k+1));

def binomial(n,k):
   if n < k: return 0;
   return round(math.exp(binomiallog(n,k)));

def hypergeometric(k,G,n,N):
   return math.exp(binomiallog(G,k) + binomiallog(N-G,n-k) - binomiallog(N,n));

def createMatrix(vectors):
   if vectors.ndim != 2:
      vectors = np.reshape(vectors,(-1,vectors.shape[-1]));
   return vectors;
   
def euclideanSqDist(data,grid):
   """
   Returns the distance from one np_array to all
   vectors in the grid, the input is any compination of array and matrix
   """
   
   d = createMatrix(data);
   g = createMatrix(grid);
   di,gi = np.mgrid[0:d.shape[0],0:g.shape[0]];
   ndir = d[di,:] - g[gi,:];
   number = np.sum(~np.isnan(ndir),-1);
   return np.sum(np.nan_to_num(ndir*ndir),-1) * d.shape[-1] / number;


   
def gaussian(pos,matrix,shift=None):
 #   pos = pos.reshape((pos.shape[0],-1));
 #  
 #   if(shift == None):
 #      shift = np.zeros(pos.shape[0]);
 #  
 #   x = np.mgrid[0:pos.shape[0]];
 #   print pos;
 # #  pos = pos[x] - shift[x];
 #   
 #   print shift;
 #   print pos;
 #   print matrix;
   
   
   
   if(len(matrix.shape) == 1):
      inner_dot = pos * matrix;
      outer_dot = inner_dot * pos;
      return 1.0/(np.sqrt(np.pi*matrix)) * np.exp(- outer_dot );
   else:
      print "Enter"
      shape = pos.shape;
      print "Shape", shape
      inner_dot = pos.transpose()
    #  gauss =np.exp(-np.dot(np.dot(pos.transpose(),matrix),pos));
      return

   return gauss;


class MathTester (unittest.TestCase):
   
   def testBinomial(self):
      self.assertEqual(binomial(2,3),0);
      self.assertEqual(binomial(3,2),3);
      self.assertEqual(binomial(15,4),1365);
      self.assertEqual(binomial(52,16),10363194502115);
      
   
   def testHypergeometric(self):
      self.assertAlmostEqual(hypergeometric(1,4,2,10),0.5333333333,10);
      self.assertAlmostEqual(hypergeometric(10,15,20,64),0.00125782191601423,10);
      self.assertAlmostEqual(hypergeometric(10,15,80,102),0.124300105660462,10);
   
   def testEuclideanSqDist(self):
      a = np.array([1,2,np.nan,4]);
      test = np.array([0,1,24,3]);
      self.assertAlmostEqual(euclideanSqDist(a,test),4.0,10);
      
      test2 = np.array([[0,1,24,3],[-1,0,24,2],[-2,-1,24,1]]);
      self.assertTrue((euclideanSqDist(a,test2) == np.array([4.0,16.0,36.0])).all());
      
      test3 = np.array([[[0,1,24,3],[-1,0,24,2],[-2,-1,24,1]],[[0,1,24,3],[-1,0,24,2],[-2,-1,24,1]],[[0,1,24,3],[-1,0,24,2],[-2,-1,24,1]]]);
      self.assertTrue((euclideanSqDist(a,test3) == np.array([[4.0,16.0,36.0,4.0,16.0,36.0,4.0,16.0,36.0]])).all());

   def testEuclideanSqDistAddvanced(self):
      a = np.array([[0,0,np.nan,0],[1,1,1,1]]);
      test = np.array([[0.5,np.nan,0.5,0.5],[1.5,1.5,1.5,1.5]]);

   def testCreateMatrix(self):
      self.assertEquals(createMatrix(np.array([1,2,3,4])).ndim,2);
      self.assertEquals(createMatrix(np.array([[1,2,3,4],[5,6,7,8]])).ndim,2);
      self.assertEquals(createMatrix(np.array([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]])).ndim,2);

   def testCalculateStatistics(self):
      data = np.array([[1,1,1],[np.nan,0,0],[-1,-1,-1]]);
      (vpc,mean,stddiv) = calculateStatistics(data);
      np.testing.assert_equal(vpc,[2,3,3]);
      np.testing.assert_almost_equal(mean,[0,0,0]);
      np.testing.assert_almost_equal(stddiv,[1,0.8164965,0.8164965]);

   def testNormalizeData(self):
      data = np.array([[1,1,1],[np.nan,0,0],[-1,-1,-1]]);
      normalizeData(data);
      ##TODO: do test later.
   
   def testGaussian(self):
      array =np.mgrid[-1:2,-1:2];
#      print array
#      matrix = np.array([1.0]);
      matrix = np.matrix([[1.0/2,0],[0,1.0/2]]);
     
#      print gaussian(array,matrix);
      

if __name__ == "__main__":
   unittest.main();