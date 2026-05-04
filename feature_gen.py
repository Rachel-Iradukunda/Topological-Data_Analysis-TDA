import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GIOTTO-TDA IMPORTS
# ============================================================================
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude, BettiCurve
    GTDATDA_AVAILABLE = True
except ImportError:
    GTDATDA_AVAILABLE = False
    print("Warning: Giotto-TDA not available. Some features will be disabled.")

# ============================================================================
# PERSISTENT HOMOLOGY FEATURE EXTRACTOR
# ============================================================================
class PersistentHomologyExtractor:
    """Extract topological features using giotto-tda"""
    
    def __init__(self, homology_dimensions=(0, 1), n_bins=50, n_jobs=1):
        self.homology_dimensions = homology_dimensions
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        
        if GTDATDA_AVAILABLE:
            self._initialize_gtda()
        else:
            self.persistence = None
            self.entropy = None
            self.amplitudes = {}
            self.betti = None
    
    def _initialize_gtda(self):
        """Initialize giotto-tda components"""
        # Persistence diagrams
        self.persistence = VietorisRipsPersistence(
            metric='precomputed',
            homology_dimensions=self.homology_dimensions,
            collapse_edges=True,
            n_jobs=self.n_jobs
        )
        
        # Persistence entropy
        self.entropy = PersistenceEntropy(n_jobs=self.n_jobs)
        
        # Amplitude metrics
        self.amplitudes = {
            'wasserstein': Amplitude(
                metric='wasserstein',
                metric_params={'p': 1},
                n_jobs=self.n_jobs
            ),
            'bottleneck': Amplitude(
                metric='bottleneck',
                n_jobs=self.n_jobs
            ),
            'landscape': Amplitude(
                metric='landscape',
                metric_params={'n_layers': 1, 'n_bins': self.n_bins},
                n_jobs=self.n_jobs
            ),
            'heat': Amplitude(
                metric='heat',
                metric_params={'sigma': 0.1, 'n_bins': self.n_bins},
                n_jobs=self.n_jobs
            ),
            'betti': Amplitude(
                metric='betti',
                metric_params={'n_bins': self.n_bins},
                n_jobs=self.n_jobs
            )
        }
        
        # Betti curves (this includes H0 and H1 curves)
        self.betti = BettiCurve(n_bins=self.n_bins, n_jobs=self.n_jobs)
    
    def adjacency_to_distance(self, adjacency_matrix):
        """Convert adjacency matrix to distance matrix"""
        distance = adjacency_matrix.copy()
        np.fill_diagonal(distance, 0)
        
        # Convert weights to distances (inverse relationship)
        with np.errstate(divide='ignore', invalid='ignore'):
            distance = 1.0 / (distance + 1e-10)
            distance[distance > 1e6] = 1e6  # Cap large distances
        
        # Ensure symmetry
        distance = np.maximum(distance, distance.T)
        np.fill_diagonal(distance, 0)
        
        return distance
    
    def extract_features(self, adjacency_matrices):
        """
        Extract all persistent homology features
        
        Returns:
        --------
        dict: Dictionary with features
        """
        if not GTDATDA_AVAILABLE:
            return self._get_empty_features(adjacency_matrices)
        
        features = {}
        
        # 1. Convert to distance matrices
        distance_matrices = np.array([
            self.adjacency_to_distance(adj) for adj in adjacency_matrices
        ])
        
        # 2. Persistence diagrams
        persistence_diagrams = self.persistence.fit_transform(distance_matrices)
        features['persistence_diagrams'] = persistence_diagrams
        
        # 3. Persistence entropy
        entropy_features = self.entropy.fit_transform(persistence_diagrams)
        features['persistent_entropy'] = entropy_features
        
        # 4. Amplitudes (all metrics)
        amplitude_features = {}
        for metric_name, amplitude_calc in self.amplitudes.items():
            try:
                amp = amplitude_calc.fit_transform(persistence_diagrams)
                amplitude_features[metric_name] = amp
            except:
                n_samples = len(adjacency_matrices)
                n_dims = len(self.homology_dimensions)
                amplitude_features[metric_name] = np.zeros((n_samples, n_dims))
        features['amplitudes'] = amplitude_features
        
        # 5. Betti curves
        betti_curves = self.betti.fit_transform(persistence_diagrams)
        features['betti_curves'] = betti_curves
        
        return features
    
    def _get_empty_features(self, adjacency_matrices):
        """Return empty features when giotto-tda is not available"""
        n_samples = len(adjacency_matrices)
        n_dims = len(self.homology_dimensions)
        
        return {
            'persistence_diagrams': [np.array([])] * n_samples,
            'persistent_entropy': np.zeros((n_samples, n_dims)),
            'amplitudes': {
                'wasserstein': np.zeros((n_samples, n_dims)),
                'bottleneck': np.zeros((n_samples, n_dims)),
                'landscape': np.zeros((n_samples, n_dims)),
                'heat': np.zeros((n_samples, n_dims)),
                'betti': np.zeros((n_samples, n_dims))
            },
            'betti_curves': np.zeros((n_samples, n_dims * self.n_bins))
        }


# ============================================================================
# SPECTRAL FEATURE EXTRACTOR
# ============================================================================
class SpectralFeatureExtractor:
    """Extract spectral features from graph Laplacians"""
    
    def __init__(self, n_eigenvalues=5):
        self.n_eigenvalues = n_eigenvalues
    
    def compute_graph_laplacian(self, adjacency):
        """
        Compute the weighted graph Laplacian (0-Laplacian)
        L = D - A, where D is the weighted degree matrix
        """
        # Get only the non-zero part
        nonzero_mask = np.any(adjacency != 0, axis=1)
        adj_reduced = adjacency[nonzero_mask][:, nonzero_mask]
        
        if adj_reduced.shape[0] == 0:
            return np.zeros((1, 1))
        
        # Compute degree matrix
        degrees = np.sum(adj_reduced, axis=1)
        D = np.diag(degrees)
        
        # Laplacian = D - A
        L = D - adj_reduced
        
        return L
    
    def compute_edge_laplacian(self, adjacency):
        """
        Compute the edge Laplacian (1-Laplacian)
        This operates on the edges of the graph
        """
        # Build NetworkX graph
        G = nx.from_numpy_array(adjacency)
        
        # Remove self-loops and zero-weight edges
        G.remove_edges_from(nx.selfloop_edges(G))
        edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w == 0 or w is None]
        G.remove_edges_from(edges_to_remove)
        
        if G.number_of_edges() == 0:
            return np.zeros((1, 1))
        
        # Compute edge Laplacian using incidence matrix
        # For a graph with n nodes and m edges:
        # B is the incidence matrix (n x m)
        # Edge Laplacian L1 = B^T B (m x m matrix)
        
        edges = list(G.edges())
        n_nodes = G.number_of_nodes()
        n_edges = len(edges)
        
        # Build incidence matrix
        B = np.zeros((n_nodes, n_edges))
        for idx, (u, v) in enumerate(edges):
            B[u, idx] = 1
            B[v, idx] = -1
        
        # Edge Laplacian
        L1 = B.T @ B
        
        return L1
    
    def extract_eigenvalues(self, laplacian, k=None):
        """
        Extract smallest k eigenvalues from Laplacian
        """
        if k is None:
            k = self.n_eigenvalues
        
        n = laplacian.shape[0]
        if n == 0:
            return np.zeros(k)
        
        k = min(k, n - 1)
        if k <= 0:
            return np.zeros(self.n_eigenvalues)
        
        try:
            # Use sparse eigensolver for efficiency
            if n > 10:
                eigenvalues, _ = eigsh(laplacian, k=k, which='SM')
            else:
                # For small matrices, use dense solver
                eigenvalues = eigh(laplacian, eigvals_only=True)
                eigenvalues = np.sort(eigenvalues)[:k]
            
            # Pad if necessary
            if len(eigenvalues) < self.n_eigenvalues:
                eigenvalues = np.pad(eigenvalues, 
                                    (0, self.n_eigenvalues - len(eigenvalues)), 
                                    mode='constant')
            
            return eigenvalues[:self.n_eigenvalues]
        except:
            return np.zeros(self.n_eigenvalues)
    
    def extract_features(self, adjacency):
        """
        Extract all spectral features for a single graph
        """
        features = []
        
        # 1. Graph Laplacian (0-Laplacian) eigenvalues
        L0 = self.compute_graph_laplacian(adjacency)
        eigenvalues_0 = self.extract_eigenvalues(L0)
        features.extend(eigenvalues_0)
        
        # 2. Edge Laplacian (1-Laplacian) eigenvalues
        L1 = self.compute_edge_laplacian(adjacency)
        eigenvalues_1 = self.extract_eigenvalues(L1)
        features.extend(eigenvalues_1)
        
        # 3. Algebraic connectivity (second smallest eigenvalue of L0)
        if len(eigenvalues_0) >= 2:
            algebraic_connectivity = eigenvalues_0[1]
        else:
            algebraic_connectivity = 0
        features.append(algebraic_connectivity)
        
        # 4. Spectral range (max - min eigenvalue)
        spectral_range_0 = np.max(eigenvalues_0) - np.min(eigenvalues_0)
        spectral_range_1 = np.max(eigenvalues_1) - np.min(eigenvalues_1)
        features.append(spectral_range_0)
        features.append(spectral_range_1)
        
        return np.array(features)


# ============================================================================
# CARLSSON COORDINATES EXTRACTOR
# ============================================================================
class CarlssonCoordinatesExtractor:
    """
    Extract Carlsson coordinates based on graph eccentricity
    These provide a geometric embedding of the graph structure
    """
    
    def __init__(self, n_coordinates=5):
        self.n_coordinates = n_coordinates
    
    def extract_features(self, adjacency):
        """
        Extract Carlsson coordinates
        
        Carlsson coordinates are based on eccentricity:
        - Eccentricity of a node = max distance to any other node
        - Coordinates are statistical summaries of eccentricities
        """
        # Build graph
        G = nx.from_numpy_array(adjacency)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if G.number_of_nodes() == 0:
            return np.zeros(self.n_coordinates * 2)
        
        features = []
        
        try:
            # Compute eccentricities if graph is connected
            if nx.is_connected(G):
                eccentricities = nx.eccentricity(G)
                ecc_values = list(eccentricities.values())
            else:
                # For disconnected graphs, compute per component
                ecc_values = []
                for component in nx.connected_components(G):
                    subG = G.subgraph(component)
                    if len(component) > 1:
                        ecc = nx.eccentricity(subG)
                        ecc_values.extend(ecc.values())
            
            if len(ecc_values) == 0:
                ecc_values = [0]
            
            ecc_array = np.array(ecc_values)
            
            # Statistical features from eccentricities
            features.append(np.mean(ecc_array))
            features.append(np.std(ecc_array))
            features.append(np.min(ecc_array))
            features.append(np.max(ecc_array))
            features.append(np.median(ecc_array))
            
            # Percentiles
            if len(ecc_array) > 1:
                features.append(np.percentile(ecc_array, 25))
                features.append(np.percentile(ecc_array, 75))
            else:
                features.append(ecc_array[0])
                features.append(ecc_array[0])
            
        except:
            features = [0] * 7
        
        # Degree-based coordinates (complementary geometric info)
        try:
            degrees = [d for n, d in G.degree()]
            if len(degrees) > 0:
                deg_array = np.array(degrees)
                features.append(np.mean(deg_array))
                features.append(np.std(deg_array))
                features.append(np.max(deg_array))
            else:
                features.extend([0, 0, 0])
        except:
            features.extend([0, 0, 0])
        
        return np.array(features)


# ============================================================================
# GRAPH STATISTICS EXTRACTOR
# ============================================================================
class GraphStatisticsExtractor:
    """Extract classical graph-theoretic features"""
    
    def extract_features(self, adjacency):
        """Extract graph statistics"""
        # Build graph
        G = nx.from_numpy_array(adjacency)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        features = []
        
        # 1. Transitivity (global clustering coefficient)
        try:
            transitivity = nx.transitivity(G)
        except:
            transitivity = 0
        features.append(transitivity)
        
        # 2. Average clustering coefficient
        try:
            avg_clustering = nx.average_clustering(G, weight='weight')
        except:
            avg_clustering = 0
        features.append(avg_clustering)
        
        # 3. Number of triangles
        try:
            triangles = sum(nx.triangles(G).values()) // 3
        except:
            triangles = 0
        features.append(triangles)
        
        # 4. Degree statistics
        degrees = [d for n, d in G.degree()]
        if len(degrees) > 0:
            features.append(np.mean(degrees))  # Mean degree
            features.append(np.std(degrees))   # Std degree
            features.append(np.max(degrees))   # Max degree
            features.append(np.min(degrees))   # Min degree
        else:
            features.extend([0, 0, 0, 0])
        
        # 5. Diameter and average path length (for largest component)
        try:
            if nx.is_connected(G):
                diameter = nx.diameter(G)
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                subG = G.subgraph(largest_cc)
                if len(largest_cc) > 1:
                    diameter = nx.diameter(subG)
                    avg_path_length = nx.average_shortest_path_length(subG)
                else:
                    diameter = 0
                    avg_path_length = 0
        except:
            diameter = 0
            avg_path_length = 0
        features.append(diameter)
        features.append(avg_path_length)
        
        # 6. Number of connected components
        n_components = nx.number_connected_components(G)
        features.append(n_components)
        
        # 7. Graph density
        try:
            density = nx.density(G)
        except:
            density = 0
        features.append(density)
        
        # 8. Assortativity (degree correlation)
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
        except:
            assortativity = 0
        features.append(assortativity)
        
        # 9. Cycle basis length statistics
        try:
            cycle_basis = nx.cycle_basis(G)
            if len(cycle_basis) > 0:
                cycle_lengths = [len(c) for c in cycle_basis]
                features.append(len(cycle_basis))  # Number of cycles
                features.append(np.mean(cycle_lengths))  # Mean cycle length
                features.append(np.max(cycle_lengths))   # Max cycle length
            else:
                features.extend([0, 0, 0])
        except:
            features.extend([0, 0, 0])
        
        # 10. Edge weight statistics
        weights = [d['weight'] for u, v, d in G.edges(data=True) if 'weight' in d]
        if len(weights) > 0:
            features.append(np.mean(weights))
            features.append(np.std(weights))
            features.append(np.max(weights))
        else:
            features.extend([0, 0, 0])
        
        # 11. Node count and edge count
        features.append(G.number_of_nodes())
        features.append(G.number_of_edges())
        
        return np.array(features)


# ============================================================================
# MAIN FEATURE GENERATOR
# ============================================================================
class DiscriminativeFeatureGenerator:
    """
    Enhanced feature generator with TDA + Spectral + Carlsson + Graph Statistics
    """
    
    def __init__(self, homology_dimensions=(0, 1), n_bins=50, 
                 n_eigenvalues=5, n_carlsson=10, n_jobs=1):
        self.homology_dimensions = homology_dimensions
        self.n_bins = n_bins
        self.n_eigenvalues = n_eigenvalues
        
        # Initialize components
        self.scaler = StandardScaler()
        self.ph_extractor = PersistentHomologyExtractor(
            homology_dimensions=homology_dimensions,
            n_bins=n_bins,
            n_jobs=n_jobs
        )
        self.spectral_extractor = SpectralFeatureExtractor(n_eigenvalues=n_eigenvalues)
        self.carlsson_extractor = CarlssonCoordinatesExtractor(n_coordinates=n_carlsson)
        self.graph_stats_extractor = GraphStatisticsExtractor()
        
        # Feature tracking
        self.feature_dim = None
        self.feature_names = []
    
    def generate_features(self, adjacency_matrices, labels):
        """Generate discriminative features"""
        print("Generating enhanced discriminative features...")
        X = []
        y = []
        feature_info = {
            'feature_shapes': [],
            'feature_names': []
        }
        
        # First pass: determine feature dimension
        if self.feature_dim is None:
            print("Determining feature dimension...")
            test_feature = self._graph_to_discriminative_features(adjacency_matrices[0])
            self.feature_dim = len(test_feature)
            feature_info['feature_names'] = self._get_feature_names()
            print(f"Feature dimension: {self.feature_dim}")
            print(f"Giotto-TDA available: {GTDATDA_AVAILABLE}")
        
        for i, adjacency in enumerate(adjacency_matrices):
            if i % 500 == 0:
                print(f"Processing graph {i+1}/{len(adjacency_matrices)}")
            
            feature_vector = self._graph_to_discriminative_features(adjacency)
            
            # Ensure consistent feature length
            if len(feature_vector) != self.feature_dim:
                if len(feature_vector) > self.feature_dim:
                    feature_vector = feature_vector[:self.feature_dim]
                else:
                    feature_vector = np.pad(feature_vector, (0, self.feature_dim - len(feature_vector)), 
                                          mode='constant')
            
            X.append(feature_vector)
            y.append(labels[i])
            feature_info['feature_shapes'].append(feature_vector.shape)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Total features: {X.shape[1]}")
        print(f"Feature extraction completed for {len(adjacency_matrices)} molecules")
        
        return X, y, feature_info
    
    def _graph_to_discriminative_features(self, adjacency):
        """Convert graph to discriminative features"""
        # Extract persistent homology features for this graph
        ph_features_single = self.ph_extractor.extract_features([adjacency])
        
        features = []
        
        # ===== TDA FEATURES =====
        # 1. Persistent entropy
        if 'persistent_entropy' in ph_features_single:
            features.extend(ph_features_single['persistent_entropy'][0])
        
        # 2. Amplitudes (all metrics)
        if 'amplitudes' in ph_features_single:
            for metric in ['wasserstein', 'bottleneck', 'landscape', 'heat', 'betti']:
                if metric in ph_features_single['amplitudes']:
                    features.extend(ph_features_single['amplitudes'][metric][0])
        
        # 3. Betti curves (flattened)
        if 'betti_curves' in ph_features_single:
            betti_flat = ph_features_single['betti_curves'][0].flatten()
            features.extend(betti_flat)
        
        # ===== SPECTRAL FEATURES =====
        spectral_features = self.spectral_extractor.extract_features(adjacency)
        features.extend(spectral_features)
        
        # ===== CARLSSON COORDINATES =====
        carlsson_features = self.carlsson_extractor.extract_features(adjacency)
        features.extend(carlsson_features)
        
        # ===== GRAPH STATISTICS =====
        graph_stats = self.graph_stats_extractor.extract_features(adjacency)
        features.extend(graph_stats)
        
        return np.array(features)
    
    def _get_feature_names(self):
        """Get descriptive feature names"""
        names = []
        
        # === TDA Features ===
        # Persistent entropy
        for dim in self.homology_dimensions:
            names.append(f'persistent_entropy_H{dim}')
        
        # Amplitudes
        for metric in ['wasserstein', 'bottleneck', 'landscape', 'heat', 'betti']:
            for dim in self.homology_dimensions:
                names.append(f'amplitude_{metric}_H{dim}')
        
        # Betti curves
        for dim in self.homology_dimensions:
            for bin_idx in range(self.n_bins):
                names.append(f'betti_H{dim}_bin_{bin_idx}')
        
        # === Spectral Features ===
        # 0-Laplacian eigenvalues
        for i in range(self.n_eigenvalues):
            names.append(f'laplacian_0_eigenvalue_{i}')
        
        # 1-Laplacian eigenvalues
        for i in range(self.n_eigenvalues):
            names.append(f'laplacian_1_eigenvalue_{i}')
        
        # Other spectral features
        names.append('algebraic_connectivity')
        names.append('spectral_range_L0')
        names.append('spectral_range_L1')
        
        # === Carlsson Coordinates ===
        names.extend([
            'eccentricity_mean',
            'eccentricity_std',
            'eccentricity_min',
            'eccentricity_max',
            'eccentricity_median',
            'eccentricity_q25',
            'eccentricity_q75',
            'degree_mean_carlsson',
            'degree_std_carlsson',
            'degree_max_carlsson'
        ])
        
        # === Graph Statistics ===
        names.extend([
            'transitivity',
            'avg_clustering',
            'num_triangles',
            'degree_mean',
            'degree_std',
            'degree_max',
            'degree_min',
            'diameter',
            'avg_path_length',
            'num_components',
            'density',
            'assortativity',
            'num_cycles',
            'mean_cycle_length',
            'max_cycle_length',
            'edge_weight_mean',
            'edge_weight_std',
            'edge_weight_max',
            'num_nodes',
            'num_edges'
        ])
        
        return names
    
    def fit_transform(self, adjacency_matrices, labels):
        """Generate features and fit scaler"""
        X, y, feature_info = self.generate_features(adjacency_matrices, labels)
        if len(X) == 0:
            return None, None, None
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y, feature_info
    
    def transform(self, adjacency_matrices):
        """Transform new data"""
        X = []
        for adjacency in adjacency_matrices:
            feature_vector = self._graph_to_discriminative_features(adjacency)
            if len(feature_vector) != self.feature_dim:
                if len(feature_vector) > self.feature_dim:
                    feature_vector = feature_vector[:self.feature_dim]
                else:
                    feature_vector = np.pad(feature_vector, (0, self.feature_dim - len(feature_vector)), 
                                          mode='constant')
            X.append(feature_vector)
        
        X = np.array(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X


# ============================================================================
# FEATURE COUNT CALCULATOR
# ============================================================================
def calculate_feature_counts(homology_dimensions=(0, 1), n_bins=50, n_eigenvalues=5):
    """Calculate total number of features"""
    n_dims = len(homology_dimensions)
    
    counts = {
        # TDA features
        'Persistent_entropy': n_dims,
        'Amplitudes': 5 * n_dims,
        'Betti_curves': n_bins * n_dims,
        
        # Spectral features
        'Laplacian_0_eigenvalues': n_eigenvalues,
        'Laplacian_1_eigenvalues': n_eigenvalues,
        'Spectral_summary': 3,  # algebraic connectivity + 2 spectral ranges
        
        # Carlsson coordinates
        'Carlsson_coordinates': 10,
        
        # Graph statistics
        'Graph_statistics': 20
    }
    
    total = sum(counts.values())
    
    print("FEATURES")
    print("=" * 50)
    print("\nTDA FEATURES:")
    print(f"  Persistent entropy    : {counts['Persistent_entropy']:3d}")
    print(f"  Amplitudes            : {counts['Amplitudes']:3d}")
    print(f"  Betti curves          : {counts['Betti_curves']:3d}")
    print(f"  Subtotal              : {counts['Persistent_entropy'] + counts['Amplitudes'] + counts['Betti_curves']:3d}")
    
    print("\nSPECTRAL FEATURES:")
    print(f"  0-Laplacian eigenvals : {counts['Laplacian_0_eigenvalues']:3d}")
    print(f"  1-Laplacian eigenvals : {counts['Laplacian_1_eigenvalues']:3d}")
    print(f"  Spectral summary      : {counts['Spectral_summary']:3d}")
    print(f"  Subtotal              : {counts['Laplacian_0_eigenvalues'] + counts['Laplacian_1_eigenvalues'] + counts['Spectral_summary']:3d}")
    
    print("\nGEOMETRIC FEATURES:")
    print(f"  Carlsson coordinates  : {counts['Carlsson_coordinates']:3d}")
    
    print("\nGRAPH STATISTICS:")
    print(f"  Graph statistics      : {counts['Graph_statistics']:3d}")
    
    print("\n" + "=" * 50)
    print(f"TOTAL FEATURES          : {total:3d}")
    print(f"Budget (max 120)        : 120")
    print(f"Remaining capacity      : {120 - total:3d}")
    print("=" * 50)
    
    return total, counts


# Test the feature counter
if __name__ == "__main__":
    print("\n" + "="*70)
    print("FEATURE COUNT TEST")
    print("="*70)
    total, counts = calculate_feature_counts(homology_dimensions=(0, 1), n_bins=20, n_eigenvalues=5)
