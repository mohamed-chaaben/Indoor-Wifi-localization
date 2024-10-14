import mlflow
from utils import Data, EpsEstimation, Clustering
features = ['length', 'IE 221', 'IE 127', 'IE 45', 'IE 221*', 'IE 127*', 'IE 45*']


subgroups = (
    ['D', 'C', 'N', 'E', 'M', 'R', 'U', 'B', 'A', 'L', 'V', 'I'],
    ['G', 'H', 'Q',   'W', 'J', 'S', 'X', 'T', 'K', 'O', 'N', 'E', 'M', 'R', 'U', 'B', 'A', 'L', 'V', 'I'],
    ['W', 'J', 'S', 'X', 'T', 'K', 'O', 'N', 'E', 'M', 'R', 'U', 'B', 'A', 'L', 'V', 'I'],
    ['W', 'J', 'S', 'T', 'K', 'O', 'N', 'E',  'R', 'B', 'A', 'L', 'V', 'I'],
    ['W', 'J', 'S', 'X', 'T', 'K', 'O', 'N', 'E', 'M', 'R', 'U', 'B', 'A', 'L', 'V', 'I', 'G', 'H', 'Q', 'D', 'C' ])

mlflow.set_experiment("Clustering Experiment")

for subgroup in subgroups:
    with mlflow.start_run():
        data = Data(path='all_dataframeV2.csv', features=features, subgroup=subgroup, drop_duplicates=False)
        min_points = 6
        eps_estimation = EpsEstimation(data.X, min_points)
        eps = eps_estimation.fit()

        mlflow.log_param('subgroup', subgroup)
        mlflow.log_param('min_points', min_points)
        mlflow.log_param('eps', eps)
        cls = Clustering(data.X, data.labels, eps, min_points, metric='precomputed')
        cls.fit()

        mlflow.log_metric('homogeneity', cls.homogeneity)
        mlflow.log_metric('completeness', cls.completeness)
        mlflow.log_metric('v_measure', cls.v_measure)
        mlflow.log_metric('avg_silhouette', cls.avg_silhouette)
        plot_file = cls.plot()
        mlflow.log_artifact(plot_file)

