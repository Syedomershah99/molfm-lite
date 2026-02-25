"""Data preprocessing for MolFM-Lite"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Lazy imports for optional dependencies
_rdkit_available = None
_selfies_available = None


def _check_rdkit():
    global _rdkit_available
    if _rdkit_available is None:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors
            _rdkit_available = True
        except ImportError:
            _rdkit_available = False
    return _rdkit_available


def _check_selfies():
    global _selfies_available
    if _selfies_available is None:
        try:
            import selfies
            _selfies_available = True
        except ImportError:
            _selfies_available = False
    return _selfies_available


@dataclass
class MolecularFeatures:
    """Container for molecular features across modalities"""
    smiles: str
    selfies: Optional[str] = None
    # 1D features
    token_ids: Optional[np.ndarray] = None
    # 2D features
    atom_features: Optional[np.ndarray] = None
    bond_features: Optional[np.ndarray] = None
    edge_index: Optional[np.ndarray] = None
    # 3D features
    conformers: Optional[List[np.ndarray]] = None  # List of (N, 3) coordinates
    conformer_energies: Optional[np.ndarray] = None
    # Context (optional)
    context: Optional[Dict[str, Any]] = None


class MoleculePreprocessor:
    """Preprocessor for molecular data"""

    # Atom features
    ATOM_FEATURES = {
        "atom_type": ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Na", "K", "other"],
        "degree": [0, 1, 2, 3, 4, 5],
        "formal_charge": [-2, -1, 0, 1, 2],
        "num_hs": [0, 1, 2, 3, 4],
        "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "other"],
    }

    # Bond features
    BOND_FEATURES = {
        "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
        "stereo": ["NONE", "E", "Z", "CIS", "TRANS"],
        "is_conjugated": [False, True],
        "is_in_ring": [False, True],
    }

    # SELFIES vocabulary (simplified)
    SELFIES_VOCAB = {
        "[C]": 1, "[N]": 2, "[O]": 3, "[S]": 4, "[F]": 5, "[Cl]": 6, "[Br]": 7,
        "[I]": 8, "[P]": 9, "[=C]": 10, "[=N]": 11, "[=O]": 12, "[=S]": 13,
        "[#C]": 14, "[#N]": 15, "[Ring1]": 16, "[Ring2]": 17, "[Branch1]": 18,
        "[Branch2]": 19, "[/C]": 20, "[\\C]": 21, "[C@@H1]": 22, "[C@H1]": 23,
        "[nop]": 24, "[epsilon]": 25, "[PAD]": 0, "[UNK]": 26, "[CLS]": 27, "[SEP]": 28,
    }

    def __init__(self, max_seq_len: int = 256, vocab_size: int = 128):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self._build_vocab()

    def _build_vocab(self):
        """Build SELFIES vocabulary dynamically"""
        self.token_to_id = dict(self.SELFIES_VOCAB)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.next_id = max(self.token_to_id.values()) + 1

    def _get_token_id(self, token: str) -> int:
        """Get or create token ID"""
        if token not in self.token_to_id:
            if self.next_id < self.vocab_size - 1:
                self.token_to_id[token] = self.next_id
                self.id_to_token[self.next_id] = token
                self.next_id += 1
            else:
                return self.token_to_id["[UNK]"]
        return self.token_to_id[token]

    def smiles_to_selfies(self, smiles: str) -> Optional[str]:
        """Convert SMILES to SELFIES"""
        if not _check_selfies():
            return None
        import selfies as sf
        try:
            return sf.encoder(smiles)
        except Exception:
            return None

    def tokenize_selfies(self, selfies_str: str) -> np.ndarray:
        """Tokenize SELFIES string"""
        if not _check_selfies():
            return np.zeros(self.max_seq_len, dtype=np.int64)

        import selfies as sf
        try:
            tokens = list(sf.split_selfies(selfies_str))
        except Exception:
            tokens = []

        # Add special tokens
        token_ids = [self.token_to_id["[CLS]"]]
        for token in tokens[: self.max_seq_len - 2]:
            token_ids.append(self._get_token_id(token))
        token_ids.append(self.token_to_id["[SEP]"])

        # Pad
        while len(token_ids) < self.max_seq_len:
            token_ids.append(self.token_to_id["[PAD]"])

        return np.array(token_ids[: self.max_seq_len], dtype=np.int64)

    def _one_hot(self, value: Any, choices: List[Any]) -> List[int]:
        """Create one-hot encoding"""
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        else:
            encoding[-1] = 1  # "other" category
        return encoding

    def get_atom_features(self, atom) -> np.ndarray:
        """Get feature vector for an atom"""
        if not _check_rdkit():
            raise ImportError("RDKit is required for atom feature extraction")

        from rdkit import Chem

        features = []
        # Atom type
        features.extend(self._one_hot(atom.GetSymbol(), self.ATOM_FEATURES["atom_type"]))
        # Degree
        features.extend(self._one_hot(atom.GetDegree(), self.ATOM_FEATURES["degree"]))
        # Formal charge
        features.extend(
            self._one_hot(atom.GetFormalCharge(), self.ATOM_FEATURES["formal_charge"])
        )
        # Num Hs
        features.extend(
            self._one_hot(atom.GetTotalNumHs(), self.ATOM_FEATURES["num_hs"])
        )
        # Hybridization
        features.extend(
            self._one_hot(
                str(atom.GetHybridization()), self.ATOM_FEATURES["hybridization"]
            )
        )
        # Additional features
        features.append(int(atom.GetIsAromatic()))
        features.append(int(atom.IsInRing()))

        return np.array(features, dtype=np.float32)

    def get_bond_features(self, bond) -> np.ndarray:
        """Get feature vector for a bond"""
        if not _check_rdkit():
            raise ImportError("RDKit is required for bond feature extraction")

        features = []
        # Bond type
        features.extend(
            self._one_hot(str(bond.GetBondType()), self.BOND_FEATURES["bond_type"])
        )
        # Stereo
        features.extend(self._one_hot(str(bond.GetStereo()), self.BOND_FEATURES["stereo"]))
        # Conjugated
        features.append(int(bond.GetIsConjugated()))
        # In ring
        features.append(int(bond.IsInRing()))

        return np.array(features, dtype=np.float32)

    def smiles_to_graph(
        self, smiles: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert SMILES to graph representation"""
        if not _check_rdkit():
            raise ImportError("RDKit is required for graph conversion")

        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        atom_features = np.array(atom_features, dtype=np.float32)

        # Bond features and edge index
        edge_index = []
        bond_features = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])  # Undirected
            bf = self.get_bond_features(bond)
            bond_features.extend([bf, bf])

        if len(edge_index) == 0:
            # Single atom molecule
            edge_index = np.zeros((2, 0), dtype=np.int64)
            bond_features = np.zeros((0, 11), dtype=np.float32)
        else:
            edge_index = np.array(edge_index, dtype=np.int64).T
            bond_features = np.array(bond_features, dtype=np.float32)

        return atom_features, edge_index, bond_features

    def process_molecule(
        self, smiles: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[MolecularFeatures]:
        """Process a molecule into all modality features"""
        if not _check_rdkit():
            raise ImportError("RDKit is required for molecule processing")

        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # 1D: SELFIES tokens
        selfies_str = self.smiles_to_selfies(smiles)
        token_ids = self.tokenize_selfies(selfies_str) if selfies_str else None

        # 2D: Graph features
        atom_features, edge_index, bond_features = self.smiles_to_graph(smiles)

        return MolecularFeatures(
            smiles=smiles,
            selfies=selfies_str,
            token_ids=token_ids,
            atom_features=atom_features,
            bond_features=bond_features,
            edge_index=edge_index,
            conformers=None,  # Generated separately
            conformer_energies=None,
            context=context,
        )


class ConformerGenerator:
    """Generator for 3D conformer ensembles"""

    def __init__(
        self,
        num_conformers: int = 5,
        max_attempts: int = 100,
        optimize: bool = True,
        random_seed: int = 42,
    ):
        self.num_conformers = num_conformers
        self.max_attempts = max_attempts
        self.optimize = optimize
        self.random_seed = random_seed

    def generate_conformers(
        self, smiles: str
    ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """Generate conformer ensemble for a molecule"""
        if not _check_rdkit():
            return None, None

        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate conformers using ETKDG
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = self.random_seed
            params.numThreads = 0  # Use all available threads
            # maxAttempts may not be available in all RDKit versions
            try:
                params.maxAttempts = self.max_attempts
            except AttributeError:
                pass

            conf_ids = AllChem.EmbedMultipleConfs(
                mol, numConfs=self.num_conformers * 2, params=params
            )
        except Exception:
            # Fallback without params
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, numConfs=self.num_conformers * 2, randomSeed=self.random_seed
            )

        if len(conf_ids) == 0:
            # Fallback to random coordinates
            AllChem.EmbedMolecule(mol, randomSeed=self.random_seed)
            if mol.GetNumConformers() == 0:
                return None, None
            conf_ids = [0]

        # Optimize conformers with MMFF
        energies = []
        if self.optimize:
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
            for conf_id, (converged, energy) in zip(conf_ids, results):
                energies.append(energy)
        else:
            # Calculate energies without optimization
            for conf_id in conf_ids:
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id
                )
                if ff is not None:
                    energies.append(ff.CalcEnergy())
                else:
                    energies.append(float("inf"))

        # Sort by energy and keep top conformers
        sorted_indices = np.argsort(energies)[: self.num_conformers]
        conf_ids_list = list(conf_ids)  # Convert to list for indexing
        conf_ids = [conf_ids_list[int(i)] for i in sorted_indices]
        energies = [energies[int(i)] for i in sorted_indices]

        # Extract coordinates (without hydrogens for efficiency)
        mol_no_h = Chem.RemoveHs(mol)
        conformers = []
        for conf_id in conf_ids:
            conf = mol.GetConformer(conf_id)
            # Map coordinates back to non-H atoms
            coords = []
            h_idx = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 1:  # Not hydrogen
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coords.append([pos.x, pos.y, pos.z])
            conformers.append(np.array(coords, dtype=np.float32))

        return conformers, np.array(energies, dtype=np.float32)

    def process_batch(
        self, smiles_list: List[str], n_jobs: int = -1
    ) -> List[Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]]:
        """Process multiple molecules in parallel"""
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.generate_conformers)(smiles) for smiles in smiles_list
            )
        except ImportError:
            # Fallback to sequential processing
            results = [self.generate_conformers(smiles) for smiles in smiles_list]
        return results


def compute_boltzmann_weights(
    energies: np.ndarray, temperature: float = 300.0
) -> np.ndarray:
    """Compute Boltzmann weights from conformer energies"""
    # Convert kcal/mol to kT units
    kT = 0.001987 * temperature  # R in kcal/(mol·K)
    relative_energies = energies - np.min(energies)
    weights = np.exp(-relative_energies / kT)
    return weights / np.sum(weights)
