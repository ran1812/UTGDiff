import json
import datasets
import os
from rdkit import Chem

class MOIrp(datasets.GeneratorBasedBuilder):

    def _info(self):
        features = datasets.Features(
            {
                "src": datasets.Value("string"),
                "src_graph": datasets.Value("string"),
                "trg": datasets.Value("string"),
                "cid": datasets.Value("string"),
                "sep": datasets.Value("bool"),
            }
        )
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        data_dir = './dataset/MOIrp'
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
                line[4] = line[4][:-1]
                mol3 =  Chem.MolFromSmiles(line[4])
                if mol3 is None or mol3.GetNumAtoms() > 125:
                    tot += 1
                    continue
                tmp = line[2]+'.[H-].'+line[3]
                mol_tmp = Chem.MolFromSmiles(tmp)
                if mol_tmp is None or mol_tmp.GetNumAtoms() > 125:
                    tot += 1
                    continue
                yield idx, {
                    "src": line[1],
                    "src_graph": tmp,
                    "trg": line[4],
                    "cid": line[0],
                    "sep": True,
                }
            print(tot,'erw')