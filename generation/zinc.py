# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: File Description"""

import json
import gzip
import os

import datasets

# Using https://www.bibtex.com/c/doi-to-bibtex-converter/ (doi: 10.1093/nar/gkv951)
_CITATION = """\
@article{Irwin2020,
  doi = {10.1021/acs.jcim.0c00675},
  url = {https://doi.org/10.1021/acs.jcim.0c00675},
  year = {2020},
  month = oct,
  publisher = {American Chemical Society ({ACS})},
  volume = {60},
  number = {12},
  pages = {6065--6073},
  author = {John J. Irwin and Khanh G. Tang and Jennifer Young and Chinzorig Dandarchuluun and Benjamin R. Wong and Munkhzul Khurelbaatar and Yurii S. Moroz and John Mayfield and Roger A. Sayle},
  title = {{ZINC}20{\textemdash}A Free Ultralarge-Scale Chemical Database for Ligand Discovery},
  journal = {Journal of Chemical Information and Modeling}
}
"""

# You can copy an official description
_DESCRIPTION = """\
This dataset contains ~1B molecules from ZINC20, with their SMILES and SELFIES representations.
"""

_HOMEPAGE = "https://zinc20.docking.org/"

_LICENSE = "Open Data Commons Open Database License"

# this list of files has been preshuffled so we can use the same splits everytime
FILES = [
   f"https://huggingface.co/datasets/zpn/zinc20/resolve/main/zinc_processed/smiles_all_{i:02d}_clean.jsonl.gz" for i in range(5)
]

#FILES = ["https://huggingface.co/datasets/zpn/zinc20/resolve/main/zinc_processed/smiles_all_80_clean.jsonl.gz"]

class PubchemSelfies(datasets.GeneratorBasedBuilder):
    """A dataset of ZINC20 molecules represented as SELFIES."""

    VERSION = datasets.Version("1.1.0")

    # You will be able to load one or the other configurations in the following list with
    BUILDER_CONFIG = datasets.BuilderConfig(
        version=VERSION, description="A dataset of PubChem molecules represented as SELFIES."
    )

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "selfies": datasets.Value("string"),
                    "smiles": datasets.Value("string"),
                    "id": datasets.Value("string"),
                }
            ),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download(FILES)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "subdirs": downloaded_files,
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, subdirs, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        for filepath in subdirs:
            with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for row_idx, row in enumerate(f):
                    data = json.loads(row)
                    key = f"{os.path.basename(filepath)}_{row_idx}"
                    yield key, {
                        "smiles": data["smiles"],
                        "selfies": data["selfies"],
                        "id": data["id"],
                    }