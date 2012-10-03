import time
import numpy as np
import unittest
import logging
import Image,ImageDraw

log = logging.getLogger(__name__)

"""
Name: SOM
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk 
Date : 13th August - 2012

An Implementation of Self Orgenising Maps (SOM).
"""


class SOM:
   def __init__(self):
      self.size = 1;
      self.isHexagon = False;
      self.grid = None;
      self.vector_size = 1;
      
   def createGrid(self,size,vector_size,isHexagon=False):
      log.debug("Create grid of size [{},{},{}], hexagon = {} ".format(size,size,vector_size,isHexagon))
      self.size = size;
      self.isHexagon = isHexagon;
      self.vector_size = vector_size;
      self.grid = np.random.rand(size,size,vector_size);
#      self.grid = np.zeros([size,size,vector_size]) + 0.5;
   
   def _createGaussian(self,node,learn_val):
      sigma_sqrt = learn_val *  self.size / 8.0;
      (x,y) = np.mgrid[0:self.size,0:self.size];
      if self.isHexagon:
         y[0:self.size:2] = y[0:self.size:2] - 0.5;
         x = x * 0.8660254038;
      x_d, y_d = node[0]-x, node[1]-y;
      gauss =np.exp(0 - (x_d*x_d + y_d*y_d / sigma_sqrt));
      return gauss;
      
   
   def getGrid(self):
      return self.grid;
   
   def _feed(self,data,learn_val):
      """
      feed(data) -> void,
      Feeds the data to the grid.
      """

#     Calculate the squared distance to each node.      
      ndir = data - self.grid ;
      dist = np.sum(ndir*ndir,2);

      (x,y) = np.unravel_index(dist.argmin(),dist.shape);
      gauss = self._createGaussian((x,y),learn_val);
      
      corrector = (gauss * learn_val).reshape(self.size,self.size,1);
      
      self.grid = self.grid + (corrector * ndir)
      
   def train(self,data,iterations):
      log.debug("Training with {} iterations".format(iterations));
      start = time.clock();
      for i in xrange(iterations):
         learn_val = float(iterations - i) / iterations;
         x = i % len(data);
         self._feed(data[x],learn_val);
      dur = time.clock() -start;
      
      log.debug("Done training took {} s, speed {} hz".format(dur,iterations/dur));
   
   def getGridAsImages(self,size):
      if self.isHexagon:
         log.debug("Outputing images with hexagons");
         scale = 32;
         imgs = [];
         (xs,ys) = np.mgrid[0:self.size,0:self.size];
         (p_xs,p_ys) = (xs*scale,ys*scale);
         p_ys[0:self.size:2] = np.float64(p_ys[0:self.size:2]) - ( 0.25* scale)
         p_ys[1:self.size:2] = np.float64(p_ys[1:self.size:2]) + ( 0.25* scale)
#         p_xs = p_xs * 0.8660254038;
         hexa_x = [0]*6
         hexa_y = [0]*6
         for r in xrange(6):
            hexa_x[r] = (p_xs + np.cos(r*np.pi/3)*scale/2) + scale;
            hexa_y[r] = (p_ys + np.sin(r*np.pi/3)*scale/2) + scale;
         
         for i in xrange(self.vector_size):
            img = Image.new("L",(self.size*scale+scale,self.size*scale+scale));
            d = ImageDraw.Draw(img);
            for (x,y) in np.nditer([xs,ys]):
               a = []
               for r in xrange(6):
                  a.append((hexa_x[r][x,y],hexa_y[r][x,y]));
               d.polygon(a,fill=self.grid[x,y,i]*255);
               
            imgs.append(img.resize(size,Image.BICUBIC));
               
      else:
         log.debug("Outputing images with squares");
         imgs = [];
         for i in xrange(self.vector_size):
            img = Image.fromarray(np.uint8(self.grid[:,:,i]*255));
            imgs.append(img.resize(size,Image.NEAREST));
      return imgs;
      
   
class SOMTester(unittest.TestCase):
   
   def setUp(self):
      self.som = SOM();
      pass;
   
   def testInitalization(self):
      self.assertIsNone(self.som.getGrid());
      self.som.createGrid(2,2,False);
      self.assertIsNotNone(0,self.som.getGrid());
      
   def testSimpleFeed(self):
      self.som.createGrid(2,2,False);
      self.som._feed([0.5,0.5],1);
   
   def testTrain(self):
      self.som.createGrid(2,2,False);
      self.som.train([[0,0],[1,0],[0,1],[1,1]],10);
      
   def testSimpleImages(self):
      self.som.createGrid(10,2,True);
      self.som.train([[0,0],[1,0],[0,1],[1,1]],1000);
      imgs = self.som.getGridAsImages((512,512));
      imgs[0].show();
      imgs[1].show();
      
if __name__ == "__main__":
   logging.basicConfig(level=logging.DEBUG);
   unittest.main();