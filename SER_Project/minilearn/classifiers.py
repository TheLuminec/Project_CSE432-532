from pathlib import Path
import sys

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from minilearn.models.regression import *
from minilearn.models.decisiontree import *
from minilearn.models.naivebayes import *
from minilearn.models.neighbors import *
from minilearn.models.svm import *
from minilearn.models.clustering import *
from minilearn.models.ann import *

