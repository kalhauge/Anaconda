from distutils.core import setup,Extension

setup(name='Anaconda',
      version='0.1',
      packages = ["Anaconda"],
      py_modules=[
         'Anaconda.AssociationMining',
         'Anaconda.Table',
         'Anaconda.__init__',
         'Anaconda.SOM',
         'Anaconda.AMath'],
      )
      