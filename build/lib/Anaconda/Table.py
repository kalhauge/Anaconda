"""
.. module:: Table
   :platform: Unix
   :synopsis: A module to handle Tables

.. moduleauthor:: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk 

7th August - 2012

ATable is a class deticated to handeling the loading of tables and accces. If you'll need to 
change the data a AMutableTable class is available.

Currently the Class support different kinds of CSV format but still not the " " notation.

"""

import copy
import unittest
import numpy as np

from exceptions import Exception



__all__ = ["WrongSizeHeaderException","CSVFormat","ATable","AMutableTable","AFileTable"]

class WrongSizeHeaderException(Exception):
   """It point to that the row inserted in the tables isn't the same length as the header."""
   def __init__(self, table,row):
      self._table = table
      self._row = row;
   def __str__(self):
      return "Row {} not same size as {}".format(len(self._row),len(self._table._header))

class CSVFormat (object):
   """ This class is used to format, and interpret CSV strings. It contains 2 options,
   the field delimiter and the line delimiter, that descripes where to split the lines,
   and fields, and how to join them.
   
   """
   
   def __init__(self):
      self._fieldDelimiter='\t'
      self._lineDelimiter='\n'
      
   def __repr__(self):
      return repr((self._fieldDelimiter,self._lineDelimiter))
   
   def setFieldDelimiter(self,delimiter):
      """Sets the field delimiter"""
      self._fieldDelimiter = delimiter;
   
   def setLineDelimiter(self,delimiter):
      """Sets the line delimiter"""
      self._LineDelimiter = delimiter;
      
   def getFieldDelimiter(self):
      return self._fieldDelimiter;
   
   def getLineDelimiter(self):
      return self._lineDelimiter;
   
   def writeColoums(self,coloums):
      """Contatenates a array of fields to a row-string"""
      return self._fieldDelimiter.join([str(c) for c in coloums]);
      
   def writeRows(self,rows):
      """Contatenates a array of string representations of row"""
      return self._lineDelimiter.join(rows);
      
   def writeTable(self,table):
      """Writes a Table to a string using the field and line delimiter"""
      return self.writeRows([self.writeColoums(row) for row in table]);
   
   def readColoums(self,line):
      return line.split(self._fieldDelimiter);
   
   def readRows(self,string):
      return string.split(self._lineDelimiter);
      
   def readTable(self,content):
      """Reads a Table from a string using the field and line delimiter"""
      content = content.rstrip(self._lineDelimiter);
      return [self.readColoums(line) for line in self.readRows(content)];

class ATable (object):
   """ATable is the main class of this module. It is desinged to be immutable, which
      means that the ATable only functions to read data from the Table. If you want to
      change the data in the table, use the AMutableTable wich is designed for that.
   """
   
   def __init__(self,header):
      self._hasHeader = isinstance(header, list)
      if self._hasHeader:
          self._header = header;
      else:
         self._header = [""]*header;
      self._rows = [];
   
   def toCSV(self,format=CSVFormat()):
      """Converts the table into a table string, using the format."""
      if self._hasHeader:
         table = [self._header];
         table.extend(self._rows);
      else:
         table = self._rows;
      
      return format.writeTable(table);
   
   @staticmethod
   def fromArray(array,header):
      """A static method that creates a table out of an Numpy array, using an header"""
      table = ATable(header);
      table._rows = np.asarray(array);
      table._test();
      return table;
   
   @staticmethod
   def fromCSV(content,hasHeader=False,format=CSVFormat()):
      """A static method that creates a table from a CSV input"""
      rows = format.readTable(content);
      table = ATable(rows[0]);
      table._hasHeader = hasHeader;
      if table._hasHeader:
         table._rows = rows[1:];   
      else:
         table._rows = rows;
      table._test();
      return table;
   
   @staticmethod
   def loadCSV(f,hasHeader=False,format=CSVFormat()):
      content = f.read();
      return ATable.fromCSV(content,hasHeader,format);
      
   def saveCSV(self,f,format=CSVFormat()):
      f.write(self.toCSV(format));

   def setHasHeader(self,hasHeader):
      self._hasHeader = hasHeader;
   
   def _test(self):
      for row in self._rows:
         if len(row) != len(self._header):
            raise WrongSizeHeaderException(self,row);
   
   def get(self,r,c):
      return self._rows[r][c];
      
   def getHeader(self):
      return copy.deepcopy(self._header);
      
   def getHeaderForColumn(self,c):
      return self._header[c];
   
   def makeMutable(self):
      table = AMutableTable(self._header);
      table._hasHeader = self._hasHeader;
      table._rows = self._rows;
      return table

   def __iter__(self):
      return iter(self._rows);
   
   def getRows(self):
      for row in self._rows:
         yield copy.copy(row);
   
   def width(self):
      return len(self._header);
   
   def height(self):
      return len(self._rows);
   
   def xcolumn(self,index):
      for row in self._rows:
         yield row[index];

   def xrow(self,index):
      for row in self._rows[index]:
         yield row;
   
   def getFloatArray(self,columns,NaN='NA'):
      """ 
      Get Selected columns in numpy ndarray. There is a posibility to set a NaN string value
      this is by default set to 'NA'.
      """
      return np.asfarray([[np.nan if row[i] == NaN else row[i] for i in columns] for row in self._rows]);
   
   def getSubTable(self,columns):
      newtable = ATable([self._header[i] for i in columns]);
      newtable._rows = [[row[c] for c in columns] for row in self._rows];
      return newtable;
      
   def getSorted(self,key,reverse=False):
      newtable = ATable(copy.deepcopy(self._header));
      newtable._rows = sorted(self._rows,key=key,reverse=reverse);
      return newtable;
   
   def __str__(self):
      return self.toCSV();

class AMutableTable (ATable):
   
   def __init__(self,header):
      super(AMutableTable,self).__init__(header)
      
   @staticmethod
   def fromArray(array,header):
      return ATable.fromArray(array,header).makeMutable();
   
   @staticmethod
   def copy(table):
      self = AMutableTable(copy.deepcopy(table._header));
      self._hasHeader = table._hasHeader;
      self._rows = copy.deepcopy(table._rows);
      return self;
   
   @staticmethod
   def fromCSV(content,hasHeader=False,format=CSVFormat()):
      return ATable.fromCSV(content,hasHeader,format).makeMutable();
   
   @staticmethod
   def loadCSV(f,hasHeader=False,format=CSVFormat()):
      return AMutableTable.fromCSV(content,hasHeader,format);
   
   def set(self,r,c,val):
      self._rows[r][c] = val;
   
   def append(self,row):
      if len(row) == len(self._header):
         self._rows.append(row);

class AFileTable(AMutableTable):
   
   def __init__(self,header,dumpFile,format=CSVFormat()):
      super(AFileTable,self).__init__(header);
      self._dumpFile = dumpFile;
      self._format = format;
      self._dumpFile.write(format.writeColoums(header) + format.getLineDelimiter());
      
   def append(self,row):
      if len(row) == len(self._header):
         self._dumpFile.write(self._format.writeColoums(row) + self._format.getLineDelimiter());
         

      
class TableTester (unittest.TestCase):
   
   def setUp(self):
      self.table = ATable.loadCSV(open("../testfiles/table.csv","r"),True);
      
   def testLoadCSV(self):
      self.assertEqual(1,int(self.table.get(0,0)));
      self.assertEqual(12.0,float(self.table.get(1,2)));

   def testHeader(self):
      self.assertEqual("col2",self.table.getHeaderForColumn(1))

   def testCopy(self):
      mtable = AMutableTable.copy(self.table);
      mtable.set(0,0,21);
      
      self.assertEqual(21,int(mtable.get(0,0)));
      self.assertEqual(1,int(self.table.get(0,0)));
   
   def testMakeMutable(self):
      mtable = self.table.makeMutable();
      mtable.set(0,0,21);
      
      self.assertEqual(21,int(mtable.get(0,0)));
      self.assertEqual(21,int(self.table.get(0,0)));
      
   def testGetHeader(self):
      self.table.getHeader();
   
   def testNoHeaderToCSV(self):
      noHeadTable = ATable(10);
      noHeadTable.toCSV();
   
   def testNoHeaderLoadCSV(self):
      noHeadTable = ATable.loadCSV(open("../testfiles/table.csv","r"),False);
      self.assertEquals(noHeadTable.height(),4);
      
   def testCreateWithHeader(self):
      table = AMutableTable(["one","two","three"]);
      table.append([1,2,3]);
      table.append([1,2,3]);
      
   def testGetFloatArray(self):
      self.table.getFloatArray([0,2])*3;
   
   def testFromArray1(self):
      test = ATable.fromArray([[10,10],[20,20]],['a','b']);
   
   def testFromArray2(self):
      with self.assertRaises(WrongSizeHeaderException):
         test = ATable.fromArray([[10,10,20],[20,20]],['a','b']);

   def testGetRows(self):
      for row in self.table.getRows():
         row[0] = 10;
      self.assertEqual(1,int(self.table.get(0,0)));
      
   def testGetSubTable(self):
      table = self.table.getSubTable([0,2]);
      self.assertEqual(table.width(), 2);
      self.assertEqual(int(table.get(0,0)), 1);
      
   def testGetSorted(self):
      table = self.table.getSorted(lambda row: (float(row[2]),row[1]));
      self.assertEqual(float(table.get(0,2)), 1.2);

if __name__ == "__main__":
   unittest.main();
