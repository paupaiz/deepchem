import allel
import os
from typing import List, Optional, Tuple, Union, Dict, Iterator
import numpy as np
import time
import deepchem as dc
from deepchem.data import Dataset
from deepchem.utils.typing import OneOrMany
from deepchem.feat import UserDefinedFeaturizer, Featurizer
from deepchem.data import Dataset, DiskDataset
from deepchem.data.data_loader import DataLoader
from deepchem.feat.molecule_featurizers import OneHotFeaturizer
import warnings
from deepchem.molnet.load_function.molnet_loader import featurizers, _MolnetLoader, splitters, transformers, TransformerGenerator
warnings.filterwarnings('ignore')
import logging

"""
VCF and FASTQ dataset loader.
"""

logger = logging.getLogger(__name__)

# Put all tasks here. I am not sure this one is needed for the vcf and fastq loaders.
# TASKS = []
fields = None
n_samples = None
shard_size = 5500000

vcf_sample_link1 = "https://ftp.ncbi.nlm.nih.gov/toolbox/gbench/samples/vcf/GCA_000001215.4_current_ids.vcf.gz"
vcf_sample_link2 = "https://www.cottongen.org/cottongen_downloads/Gossypium_arboreum/CGP-BGI_G.arboreum_Agenome/Udall_A2ref_snp_vcf/A13.snp2.0.vcf.gz"
fastq_sample_link = "https://zenodo.org/record/3736457/files/1_control_18S_2019_minq7.fastq?download=1"

def load_files(input_files: List[str],                   
              shard_size: Optional[int] = 4096) -> Iterator:
  """Load data as Iterator.
  Parameters
  ----------
  input_files: List[str]
    List of fastq filenames.
  shard_size: int, default None
    Chunksize for reading fastq files.
  Returns
  -------
  Iterator
    Generator which yields the data which is the same shard size.
  """
  
  shard_num = 0
  for input_file in input_files:
    logger.info("About to start loading fastq from %s." % input_file)
    # Open index file
    with open(input_file, 'r') as f: 
      # create an empty list to store lines in files.
      df = []
      line_number = 0
      # iterate through each line in the input file
      for num, line in enumerate(f):
        if (num + 1) - line_number <= (shard_size * 4) :
          df.append(line)
        else:
          shard_num += 1
          logger.info("Loading shard %d of size %s." %
                  (shard_num, str(shard_size)))
          line_number = num
          yield df
          df = [line]
          
def load_vcf_files(input_files: List[str],
                    n_samples: Optional[int] = None,
                    fields : Union[str, List[str]] = None,                    
                    shard_size: Optional[int] = 70000) -> Iterator:
  """Load data as Iterator[dict].
  Parameters
  ----------
  input_files: List[str]
    List of vcf filenames.
  n_samples: Optional[int]
    Number of human samples to extract.
  fields: str or List[str]
    Fields/columns to extract e.g 'variants/CHROM', 'variants/POS' etc
  shard_size: int, default None
    Chunksize for reading json files.
  Returns
  -------
  Iterator
    Generator which yields the data which is the same shard size.
  Notes
  -----
  To load shards from a json file into a Pandas dataframe, the file
  must be originally saved with ``df.to_json('filename.json', orient='records', lines=True)``
  """
  if n_samples != None:
    n_samples = range(n_samples) 

  # shard_num = chunks = batch_size
  shard_num = 1
  for input_file in input_files:
    logger.info("About to start loading vcf from %s." % input_file)
    
    # 1 iterator per file.
    _, _, _, iterator = allel.iter_vcf_chunks(input_file,
                            fields = fields,
                            samples = n_samples,
                            chunk_length= shard_size)
    for df in iterator:
      logger.info("Loading shard %d of size %s." %
                  (shard_num, str(shard_size)))
      # iterator has chunk, chunk_length, chrom, pos but we only want chunk
      df = df[0]
      shard_num += 1
      yield df

class VCFLoader(DataLoader):
  """Handles loading of VCF files.
  VCF files are commonly used to hold very large sequence data. This
  class provides convenience files to load VCF data.
  """

  def __init__(self,
              featurizer = None,
               fields = None,
               ):
    """Initialize VCFLoader.
    Parameters
    ----------
    fields: list[str] (default None) as column names in the dataset.
      List of field names to be extracted from file. Each name can be prefixed with
      'variants'.
    
   """
    
    if fields is not None:
      assert 'calldata/GT' in fields, "Genotype data not included in the list of fields."
    # Set self.fields
    self.fields = fields
    

    if featurizer:
      warnings.warn(
          """
                    Featurizer must process data along axis >= 1 (and not axis 0)
                    for numpy variant data as human samples per variant are always along axis other 
                    than axis 0 (usually axis 1) .
                    """, )
      self.featurizer = featurizer
    else:
      self.featurizer = None

  def _get_shards(self, input_files: List[str],
                  n_samples: Optional[int] = None,
                  shard_size: Optional[int]= 70000) -> Iterator:
    """Defines a generator which returns data for each shard
    Parameters
    ----------
    input_files: List[str]
      List of file names to process
    n_samples:  int, optional
      The number of samples to extract from each variant in the data
    shard_size: int, optional
      The size of a shard of data to process at a time. Here, shard_size is equal to the
      number of variants to fetch. You can think of them as number of rows to get from the 
      full dataset.
    Returns
    -------
    Iterator
      Iterator over shards
    """
    
    return load_vcf_files(input_files, n_samples, self.fields, shard_size)
  
  def create_dataset(self,
                     input_files: OneOrMany[str],
                     data_dir: Optional[str] = None,
                     n_samples: Optional[int] = None,
                     shard_size: Optional[int] = None,
                     ) -> DiskDataset:
    """Creates a `Dataset` from input VCF files.
    Parameters
    ----------
    input_files: List[str]
      List of vcf files.
    n_samples: Optional[int]
      Number of human samples to extract from data.
    data_dir: str, optional (default None)
      Name of directory where featurized data is stored.
    shard_size: int, optional (default None)
      For now, this argument is ignored and each VCF file gets its
      own shard.
    Returns
    -------
    DiskDataset
      A `DiskDataset` object containing an array of string data
      from `input_files`.
    """
    if isinstance(input_files, str):
      input_files = [input_files]

    def shard_generator(): 

      for shard_num, shard in enumerate(self._get_shards(input_files, n_samples, shard_size)):
        time1 = time.time()
        # Read and process files in shards (batches) and store data in an array, return the variants and data_samples (X)
        # variants_features is the number of variant, X is the number of human samples per variant. 
        variants_features , X = _process(shard)
        
        if self.featurizer:
          X = self.featurizer(X)
        ids = np.arange(variants_features.shape[0])
        yield X, None, variants_features, ids


    def _process(batch: Dict) -> Tuple:
      """
      Create a numpy array of extracted vcf data. The processing of the data was simplified to 
      the most common use case (i.e use of the calldata Genotype (GT) data alongside the 
      variant fields ['CHROM', 'POS', 'REF', 'ALT'] ).
      This is done because most of the other calldata reflects depth, quality or probability of 
      the extracted Genotype Data so, they are seldom used in analysis. They also have different data shapes.
      batch: Dict[field_names[str] : values [np.ndarray]]

      Returns: tuple(variants_features[np.ndarray], data_array[np.ndarray])
              variants_features.shape == (n_rows, len(fields), 1)
              data_array.shape == (n_rows, len(human_samples), 2)
      """
      first_field = True
      # keys are chrom, pos, ref, alt, calldata/GT
      # bath is the dictionary
      for field_name in batch.keys():
        # Only attend to fields that start with variants
        # field names that start with variants are chrom, pos, reg, alt
        if field_name.startswith('variants'): 
          field = batch[field_name]
          if first_field:  
            # If each row has more than 1 dimension (column), remove the redundant element columns           
            if len(field.shape) > 1:
              for i in range(field.shape[1]-1,-1,-1):
                  if np.all(field[:,i]== ''):
                      field = field[:,:i]
            # if the field is empty after removin the redundant columns, move to the next field.
            if field.size == 0:
                continue
            elif len(field.shape) > 1:
                # If the field still has more than one element(columns), change shape as below:
                n_rows, n_columns = field.shape
                variants_features = field.reshape(n_rows, n_columns, 1)
                first_field = False
            else:
                # if there is just one element per row, then change shape and assign to the file_array variable.
                # Reshape == (number of rows, 1,1)
                variants_features = field.reshape(-1,1,1)
                first_field= False
            
          else: 
            if len(field.shape) > 1:
              for i in range(field.shape[1]-1,-1,-1):
                    if np.all(field[:,i]== ''):
                        field = field[:,:i]
            if field.size == 0:
              continue
            elif len(field.shape) > 1:   
              variants_features = np.concatenate((variants_features, np.expand_dims(field, axis=-1)), axis=1)
              
            else:         
              variants_features = np.concatenate((variants_features, field.reshape(-1,1,1)),axis=1)
               
        else: #If no variant then it's a sample calldata, we only want the calldata that contains the genotype
          if field_name == 'calldata/GT':
            # Extract genotype data only
            data_array = batch[field_name]
   
      return variants_features, data_array

  
    
    return DiskDataset.create_dataset(shard_generator(), data_dir)

class VCFDataLoader(_MolnetLoader):
  def __init__(self, featurizer: Union[dc.feat.Featurizer, str, None],
               splitter: Union[dc.splits.Splitter, str, None],
               transformer_generators: Optional[List[Union[TransformerGenerator, str]]],
               tasks: Union[List[str],None],
               data_dir: Optional[str],
               save_dir: Optional[str], **kwargs):
    """Construct an object for loading a dataset.
    Parameters
    ----------
    featurizer: Featurizer or str or None
      the featurizer to use for processing the data. Alternatively you can pass
      one of the names from dc.molnet.featurizers as a shortcut.
    splitter: Splitter or str
      the splitter to use for splitting the data into training, validation, and
      test sets.  Alternatively you can pass one of the names from
      dc.molnet.splitters as a shortcut.  If this is None, all the data
      will be included in a single dataset.
    transformer_generators: list of TransformerGenerators or strings or None
      The Transformers to apply to the data.  Each one is specified by a
      TransformerGenerator or, as a shortcut, one of the names from
      dc.molnet.transformers.
    tasks: List[str] or None
      the names of the tasks in the dataset
    data_dir: str
      a directory to save the raw data in
    save_dir: str
      a directory to save the dataset in
    """

    if 'split' in kwargs:
      splitter = kwargs['split']
      logger.warning("'split' is deprecated.  Use 'splitter' instead.")
    if featurizer is not None and isinstance(featurizer, str):
      featurizer = featurizers[featurizer.lower()]
    if isinstance(splitter, str):
      splitter = splitters[splitter.lower()]
    if data_dir is None:
      data_dir = dc.utils.data_utils.get_data_dir()
    if save_dir is None:
      save_dir = dc.utils.data_utils.get_data_dir()
    self.featurizer = featurizer
    self.splitter = splitter

    if transformer_generators is not None:
      self.transformers = [
          transformers[t.lower()] if isinstance(t, str) else t
          for t in transformer_generators
      ]
    else:
      self.transformers  = None

    if tasks != None:
      self.tasks = list(tasks)
    else:
      self.tasks = ['']

    self.data_dir = data_dir
    self.save_dir = save_dir
    self.args = kwargs

  def load_dataset(
      self, name: str, 
      reload: bool
      ) -> Tuple[List[str], Tuple[Dataset, ...], Optional[List[dc.trans.Transformer]]]:
    """Load the dataset.
    Parameters
    ----------
    name: str
      the name of the dataset, used to identify the directory on disk
    reload: bool
      if True, the first call for a particular featurizer and splitter will cache
      the datasets to disk, and subsequent calls will reload the cached datasets.
    """
    # Build the path to the dataset on disk.
    splitter_name = 'None' if self.splitter is None else str(self.splitter)

    if self.featurizer is not None: 
      featurizer_name = str(self.featurizer)
      save_folder = os.path.join(self.save_dir, name + "-featurized",
                               featurizer_name, splitter_name)
    else:
      save_folder = os.path.join(self.save_dir, name , splitter_name)

    if self.transformers is not None and len(self.transformers) > 0:
      transformer_name = '_'.join(
          t.get_directory_name() for t in self.transformers)
      save_folder = os.path.join(save_folder, transformer_name)

    # Try to reload cached datasets.
    if reload:
      if self.splitter is None:
        if os.path.exists(save_folder):
          if self.transformers is not None:
            transformers = dc.utils.data_utils.load_transformers(save_folder)
            return self.tasks, (DiskDataset(save_folder),), transformers
          else:
            return self.tasks, (DiskDataset(save_folder),)
      else:
        if self.transformers is not None:
          loaded, all_dataset, transformers = dc.utils.data_utils.load_dataset_from_disk(
              save_folder)
          if all_dataset is not None:
            return self.tasks, all_dataset, transformers
        else:
          train_dir = os.path.join(save_folder, "train_dir")
          valid_dir = os.path.join(save_folder, "valid_dir")
          test_dir = os.path.join(save_folder, "test_dir")
          if not os.path.exists(train_dir) or not os.path.exists(
              valid_dir) or not os.path.exists(test_dir):
            return False, None, list()

          loaded = True
          train = dc.data.DiskDataset(train_dir)
          valid = dc.data.DiskDataset(valid_dir)
          test = dc.data.DiskDataset(test_dir)
          train.memory_cache_size = 40 * (1 << 20)  # 40 MB
          all_dataset = (train, valid, test)

          if all_dataset is not None:
            return self.tasks, all_dataset

    # Create the dataset
    logger.info("About to create %s dataset." % name)
    dataset = self.create_dataset()
    

    # Split and transform the dataset.
    if self.splitter is None:
      if self.transformers  != None:
        transformer_dataset: Dataset = dataset
    else:
      logger.info("About to split dataset with {} splitter.".format(
          self.splitter.__class__.__name__))
      train, valid, test = self.splitter.train_valid_test_split(dataset)
      if self.transformers is not None:
        transformer_dataset = train

    if self.transformers is not None: 
      transformers = [
          t.create_transformer(transformer_dataset) for t in self.transformers
      ]

    logger.info("About to transform data.")

    # If there is no splitter
    if self.splitter is None:
      # If there are transformers
      if self.transformers is not None:
        # Transform dataset
        for transformer in transformers:
          dataset = transformer.transform(dataset)
        # If reload and instance of dataset is Diskdataset, cache data and transformers
        if reload and isinstance(dataset, DiskDataset):
          dataset.move(save_folder)
          dc.utils.data_utils.save_transformers(save_folder, transformers)
        #return 
        return self.tasks, (dataset,), transformers
      else:
        if reload and isinstance(dataset, DiskDataset):
          dataset.move(save_folder)
        return self.tasks, (dataset,)

    if self.transformers is not None:
      # If there are transformers, transform dataset
      for transformer in transformers:
        train = transformer.transform(train)
        valid = transformer.transform(valid)
        test = transformer.transform(test)

      # If reload and all instances are of the DiskDataset class
      if reload and isinstance(train, DiskDataset) and isinstance(
          valid, DiskDataset) and isinstance(test, DiskDataset):
        # save all to disk
        dc.utils.data_utils.save_dataset_to_disk(save_folder, train, valid, test,
                                                transformers)
      return self.tasks, (train, valid, test), transformers
    else:
      if reload and isinstance(train, DiskDataset) and isinstance(
          valid, DiskDataset) and isinstance(test, DiskDataset):
        train_dir = os.path.join(save_folder, "train_dir")
        valid_dir = os.path.join(save_folder, "valid_dir")
        test_dir = os.path.join(save_folder, "test_dir")
        train.move(train_dir)
        valid.move(valid_dir)
        test.move(test_dir)

      return self.tasks, (train, valid, test)
      
  
  def create_dataset(self,) -> Dataset:
    dataset_file = os.path.join(self.data_dir, "A13.snp2.0.vcf.gz")
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url= vcf_sample_link2, dest_dir=self.data_dir)
      # Uncomment the next 2 lines of code if the file is zipped tar file.
      # dc.utils.data_utils.untargz_file(  
      # os.path.join(self.data_dir, "A13.snp2.0.vcf.gz"), self.data_dir)

    # Contribute VCFLoader to MolNet
    # loader = dc.data.VCFLoader(fields=fields)

    loader = VCFLoader(featurizer=self.featurizer, fields=fields) #Strictly for testing code
    return loader.create_dataset(dataset_file,
                                n_samples=n_samples,
                                shard_size= shard_size)

if __name__ == "__main__":
  import time
  vcf_loader = VCFLoader()
  fastq_loader = FASTQLoader()
  print("Testing VCFLoader...")
  start = time.time()
  vcf_loader.create_dataset('../AD.vcf.gz', os.path.abspath(os.curdir), 10, shard_size=5000000)
  stop = time.time()
  print(f"Finished in {stop - start} second)s")
   
  # demo_loader = VCFDataLoader(None, None, None, None,
  #                   os.path.abspath(os.curdir),
  #                   os.path.join(os.path.abspath(os.curdir), 'saved_data'))

  # demo_loader.load_dataset('A13', False)
  # count = 0
  # for each in load_files(['s.fastq'], 2):

  
    