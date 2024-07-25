import json
import datasets
import os
from rdkit import Chem

atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I','P','S','Se','Si']

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
        data_dir = './dataset/CHEBI'
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
            for idx, line in enumerate(f.readlines()):
                line = line.split('\t')
                mol =  Chem.MolFromSmiles(line[1])
                '''
                if mol is None or mol.GetNumAtoms() > 125:
                    continue'''
                yield idx, {
                    "src": line[2],
                    "trg": line[1],
                    "cid": line[0],
                }