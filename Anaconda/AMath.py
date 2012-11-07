import math
import sys
import unittest
import numpy as np

"""
Name: AMath
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk 
Date : 14th September - 2012

Math is a collection of methods used in math.

"""

def gammalog(n):
   return math.log(math.factorial(n-1));

def binomiallog(n,k):
   return gammalog(n+1) - (gammalog(k+ 1) + gammalog(n-k+1));

def binomial(n,k):
   if n < k: return 0;
   return round(math.exp(binomiallog(n,k)));

def hypergeometric(k,G,n,N):
   return math.exp(binomiallog(G,k) + binomiallog(N-G,n-k) - binomiallog(N,n));

def euclideanSqDist(data,grid):
   not_nan = ~np.isnan(data);
   number = not_nan.sum();
   # sys.stderr.write("{}\n".format(type(not_nan)));
   # sys.stderr.write("{}\n".format(type(data)));
   # sys.stderr.write("{}\n".format(type(grid)));
   # 
   # sys.stderr.write("{}\n".format(data[not_nan]));
   # sys.stderr.write("{}\n".format(grid[...,not_nan]));
   
   ndir = data[not_nan] - grid[...,not_nan];
   return np.sum(ndir*ndir,-1) * data.shape[0] / number;

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
      self.assertTrue((euclideanSqDist(a,test3) == np.array([[4.0,16.0,36.0],[4.0,16.0,36.0],[4.0,16.0,36.0]])).all());
   
   def testGaussian(self):
      array =np.mgrid[-1:2,-1:2];
#      print array
#      matrix = np.array([1.0]);
      matrix = np.matrix([[1.0/2,0],[0,1.0/2]]);
     
#      print gaussian(array,matrix);
      

if __name__ == "__main__":
   unittest.main();