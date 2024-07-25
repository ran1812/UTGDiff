import json
import datasets
import os
from rdkit import Chem

class CHEBI(datasets.GeneratorBasedBuilder):

    def _info(self):
        features = datasets.Features(
            {
                "src": datasets.Value("string"),
                "trg": datasets.Value("string"),
                "cid": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        data_dir = './dataset/PCDes'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"despath": os.path.join(data_dir, "test_des.txt"),"molpath": os.path.join(data_dir, "test_smiles.txt")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"despath": os.path.join(data_dir, "val_des.txt"),"molpath": os.path.join(data_dir, "val_smiles.txt")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"despath": os.path.join(data_dir, "train_des.txt"),"molpath": os.path.join(data_dir, "train_smiles.txt")}
            ),
        ]

    def _generate_examples(self, despath, molpath):
        """This function returns the examples in the raw (text) form."""
        with open(despath, encoding="utf-8") as f:
            with open(molpath, encoding="utf-8") as g:
                for idx, line in enumerate(zip(f.readlines(),g.readlines())):
                    line,smiles = line
                    line = line[:-1]
                    mol =  Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    yield idx, {
                        "src": line,
                        "trg": smiles,
                        "cid": idx,
                    }