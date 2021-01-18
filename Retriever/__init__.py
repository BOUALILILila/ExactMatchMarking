import os 
os.environ['PATH']=os.environ['PATH']+':/logiciels/jdk-13.0.1/bin'

from .retriever import retrieve
from .get_from_run import get_content_from_run_ids

