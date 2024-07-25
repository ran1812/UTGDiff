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
        data_dir = './dataset/fixed_CHEBI'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "test.txt"),"fixedpath": os.path.join(data_dir, "gen.txt")}
            ),
        ]

    def _generate_examples(self, filepath,fixedpath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, encoding="utf-8") as f:
            with open(fixedpath, encoding="utf-8") as g:
                tot = g.readlines()
                tmp_i = 0
                for idx, line in enumerate(f.readlines()):
                    line = line.split('\t')
                    mol =  Chem.MolFromSmiles(line[1])
                    if mol.GetNumAtoms() > 125:
                        continue
                    fixed = tot[tmp_i]
                    tmp_i += 1
                    mol =  Chem.MolFromSmiles(fixed,False)
                    if mol is None:
                        continue
                    yield idx, {
                        "src": line[2],
                        "trg": fixed,
                        "cid": line[0],
                    }