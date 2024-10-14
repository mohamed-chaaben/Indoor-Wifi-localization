import time
import mlflow
import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import homogeneity_completeness_v_measure, silhouette_score, silhouette_samples
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class Data:
    def __init__(self, features:list, subgroup=None,  scaler='std', path='all_dataframeV2.csv', drop_duplicates=False):
        self.path = path
        self.scaler = scaler
        self.df = pd.read_csv(path, index_col=0)
        self.features = features + ['device_label']
        if isinstance(subgroup, list):
            self.df = self.df[self.df['device_label'].isin(subgroup)]
        self.df = self.df[self.features]
        if drop_duplicates:
            self.df = self.df.drop_duplicates()
            self.df = self.df.reset_index(drop=True)
        self.X = self.df[features].fillna(0)
        self.labels = self.df['device_label']
        if self.scaler == 'std':
            self.X = StandardScaler().fit_transform(self.X)
        elif self.scaler == 'minmax':
            self.X = MinMaxScaler().fit_transform(self.X)
            
    @property
    def length(self):
        return len(self.X)
    
def hamming_distance(numpy_array: np.ndarray) -> np.ndarray:
    """
    Generate a pairwise distance matrix by assigning 1 to non-equal features and 0 to equal features for each pair, and then summing the values across features.
    """
    return np.sum(numpy_array[:, np.newaxis, :] != numpy_array[np.newaxis, :, :], axis=2)


def hex_to_dec(x: any) -> float:
    """Convert hexadecimal to decimal"""
    try:
        return eval(x)
    except Exception:
        return 0


def dictionary_encoding(dictionary: dict) -> float:
    """
    Convert a packet's information from a dictionary format to a numerical encoding.
    """
    total_sum = 0
    for key, value in dictionary.items():
        if key != "ID" and key != "len":
            if isinstance(value, list) and all(
                isinstance(x, (int, float)) for x in value
            ):
                total_sum += sum(value)
            elif isinstance(value, bytes):
                total_sum += sum(
                    list(
                        map(
                            hex_to_dec,
                            "".join(
                                char
                                for char in repr(value)
                                .replace("\\x", "\\0x")
                                .strip("'")
                                if char.isalnum() or char == "\\"
                            ).split("\\"),
                        )
                    )
                )
            elif isinstance(value, str):
                ascii_sum = sum(ord(char) for char in value)
                total_sum += ascii_sum
            elif isinstance(value, (int, float)):
                total_sum += value
    return total_sum

class EpsEstimation:
    def __init__(self, X, min_points:int, show_plot:bool=False, metric:str='precomputed'):
        self.X = X
        self.min_points = min_points
        self.show_plot = show_plot
        self.distances_desc = None
        self.metric = metric
        self.pairwise_matrix_path = 'pairwise_matrix.npz'
        self.input = X
        if self.metric == 'precomputed':
            import os
            if os.path.exists(self.pairwise_matrix_path):
                self.input = np.load(self.pairwise_matrix_path)['arr_0']
            else:
                import time
                t0 = time.time()
                self.input = hamming_distance(self.X)
                print("Pairwise matrix loaded in ", time.time()-t0)
                np.savez_compressed(self.pairwise_matrix_path, self.input)
        self.knee = None
        self.knees = None
    

    def fit(self):
        distances, indices = NearestNeighbors(n_neighbors=self.min_points, n_jobs=-1, metric=self.metric).fit(self.input).kneighbors(self.input)
        self.distances_desc = sorted(distances[:, - 1], reverse=True)
        return self.find_knee()

        

    def plot(self):
        import plotly.express as px
        px.line(x=list(range(1, len(self.distances_desc) + 1)), y=self.distances_desc).show()
        

    def find_knee(self, s=1):
        kneedle = KneeLocator(range(1, len(self.distances_desc)+1), self.distances_desc, S=s, curve='convex', direction='decreasing')
        self.knee = kneedle.knee
        self.knees = kneedle.all_knees
        return self.distances_desc[kneedle.knee]


def perform_pca(standardized_data, target_cumulative_variance=0.85) -> pd.DataFrame:
    """
    It takes the standardized data of type pandas dataframe or ndarray, and returns a pandas dataframe
    """
    pca = PCA()
    pca.fit(standardized_data)
    cumulative_variance = 0
    num_components = 0

    for explained_variance in pca.explained_variance_ratio_:
        cumulative_variance += explained_variance
        num_components += 1

        if cumulative_variance >= target_cumulative_variance:
            break

    # Use the determined number of components
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(standardized_data)

    # Create a new DataFrame with the principal components
    columns = [f'PC{i+1}' for i in range(num_components)]
    pca_dataframe = pd.DataFrame(data=pca_result, columns=columns)

    return pca_dataframe
    
class Clustering():
    def __init__(self, dataframe:np.ndarray, labels, *args, clusterer_name:str='dbscan', metric:str='euclidean'):
        self.df = dataframe
        self.labels = labels
        self.clusterer_name = clusterer_name
        
        self.algorithm = None
        self.eps = args[0] if self.clusterer_name == 'dbscan' else None
        self.min_points = args[1] if self.clusterer_name == 'dbscan' else None
        
        if self.clusterer_name == 'dbscan':
            self.algorithm = DBSCAN
        elif self.clusterer_name == 'optics':
            self.algorithm = 'OPTICS'
        
        self.metric = metric
        self.pairwise_matrix_path = 'pairwise_matrix.npz'
        self.input = dataframe
        if self.metric == 'precomputed':
            import os
            if os.path.exists(self.pairwise_matrix_path):
                self.input = np.load(self.pairwise_matrix_path)['arr_0']
            else:
                import time
                t0 = time.time()
                self.input = hamming_distance(self.df)
                print("Pairwise matrix created in ", time.time()-t0)
                np.savez_compressed(self.pairwise_matrix_path, self.input)
            
        # Metrics
        self.number_of_clusters = None
        self.number_of_outliers = None
        self.homogeneity = None
        self.completeness = None
        self.v_measure = None
        self.avg_silhouette = None
        self.outliers = None
        self.silhouette_values = None

        
    def fit(self):
        self.cls = self.algorithm(eps=self.eps, min_samples=self.min_points, n_jobs=-1, metric=self.metric).fit(self.input)
        self.homogeneity, self.completeness, self.v_measure = homogeneity_completeness_v_measure(self.labels, self.cls.labels_)
        self.avg_silhouette = silhouette_score(self.input,  self.cls.labels_, metric='precomputed')
        self.silhouette_values = silhouette_samples(self.input, self.cls.labels_, metric='precomputed') 
        self.number_of_clusters = len(set(self.cls.labels_))    
        outliers_indices = np.where(self.cls.labels_ == -1)[0]
        self.number_of_outliers = len(outliers_indices)
        self.outliers = self.df[outliers_indices]

    
    def plot(self):
        import matplotlib.cm as cm 
        num_clusters = len(np.unique(self.cls.labels_))
        y_ax_lower, y_ax_upper = 0, 0
        y_ticks = []
        fig, ax = plt.subplots(figsize=(3, 5))
        
        for cls in np.unique(self.cls.labels_):
            cls_silhouette_vals = self.silhouette_values[self.cls.labels_ == cls]
            cluster_size = len(cls_silhouette_vals)
            cls_silhouette_vals.sort()

            y_ax_upper += cluster_size
            cmap = cm.get_cmap('Spectral')
            rgba = list(cmap(cls / num_clusters))


            ax.barh(
                y=range(y_ax_lower, y_ax_upper),
                width=cls_silhouette_vals,
                height=1,
                edgecolor="none",
                color=rgba
            )

            y_ticks.append((y_ax_lower + y_ax_upper) / 2.0)
            y_ax_lower += cluster_size + 150
            y_ax_upper += 150


        silhouette_avg = np.mean(self.silhouette_values)
        ax.axvline(silhouette_avg, color="orangered", linestyle="--")
        ax.text(silhouette_avg + 0.025, 0.5 * (y_ax_lower + y_ax_upper), f'Average: {silhouette_avg:.2f}',
                color="red", verticalalignment='center')
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster Id")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(np.unique(self.cls.labels_) + 1)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_fname = f"silhouette_plot_{timestamp}.pdf"
        mlflow.log_figure(fig, plot_fname)
        plt.close(fig) 

    @property
    def results(self):
        return {'Number of predicted clusters: ': self.number_of_clusters,
        'Homogeneity: ': self.homogeneity,
        'Completeness: ': self.completeness,
        'V_measure ': self.v_measure,
        'Avg silhouete: ': self.avg_silhouette}
        