import json
import datasets
import os
from rdkit import Chem

class MOIretro(datasets.GeneratorBasedBuilder):

    def _info(self):
        features = datasets.Features(
            {
                "src": datasets.Value("string"),
                "src_graph": datasets.Value("string"),
                "trg": datasets.Value("string"),
                "cid": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        data_dir = './dataset/MOIretro'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "test.txt")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "val.txt")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "train.txt")}
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, encoding="utf-8") as f:
            tot = 0
            for idx, line in enumerate(f.readlines()):
                line = line.split('\t')
                line[3] = line[3][:-1]
                mol1 =  Chem.MolFromSmiles(line[2])
                mol2 =  Chem.MolFromSmiles(line[3])
                if mol1 is None or mol2 is None:
                    tot += 0
                    continue
                yield idx, {
                    "src": line[1],
                    "src_graph": line[2],
                    "trg": line[3],
                    "cid": line[0],
                }
            print(tot,'erw')