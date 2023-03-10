from dgl.data import DGLDataset
from dgl.data.citation_graph import CiteseerGraphDataset
import dgl
import dgl.data
import yaml
# TODO: read tutorial of chapter 4.5 (Loading OGB datasets using ogb package)
#  https://docs.dgl.ai/guide/data-loadogb.html

# template of the DGLDataset
class MyDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(MyDataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def __getitem__(self, idx):
        # get one example by index
        pass

    def __len__(self):
        # number of data examples
        pass

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass


if __name__ == "__main__":
    # use dataset
    dataset = CiteseerGraphDataset()
    graph = dataset[0]  # for node prediction dataset, there is usually only one graph
    print(graph)

    # build a dataset from CSV files
    ds = dgl.data.CSVDataset('/Users/zhanghao/My Drive/code/DGL_PlayGround/data/demo_graph')
    g = ds[0]
    print(g)