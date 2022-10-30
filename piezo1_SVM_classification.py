import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from importlib import reload
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from pathlib import Path
from scipy import stats
from sklearn import datasets, decomposition, metrics
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import power_transform, PowerTransformer, StandardScaler

def predict_SPT_class(train_data_path, pred_data_path):
    """Computes predicted class for SPT data where
        1:Mobile, 2:Intermediate, 3:Trapped

    Args:
        train_data_path (str): complete path to training features data file in .csv format, ex. 'C:/data/tdTomato_37Degree_CytoD_training_feats.csv'
                               should be a .csv file with features columns, an 'Experiment' column identifying the unique experiment ('tdTomato_37Degree'),
                               a 'TrackID' column with unique numerical IDs for each track, and an 'Elected_Label' column derived from human voting.
        pred_data_path (str): complete path to features that need predictions in .csv format, ex. 'C:/data/newconditions/gsmtx4_feature_data.csv'
                               should be a .csv file with features columns, an 'Experiment' column identifying the unique experiment ('Cytochalasin_D'),
                               and a 'TrackID' column with unique numerical IDs for each track.
    
    Output:
        .csv file of dataframe of prediction_file features with added SVMPredictedClass column output to pred_data_path parent folder
    """
    def prepare_box_cox_data(data):
        data = data.copy()
        for col in data.columns:
            minVal = data[col].min()
            if minVal <= 0:
                data[col] += (np.abs(minVal) + 1e-15)
        return data
    train_feats = pd.read_csv(Path(train_data_path), sep=',')
    train_feats = train_feats.loc[train_feats['Experiment'] == 'tdTomato_37Degree']
    train_feats = train_feats[['Experiment', 'TrackID', 'NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension', 'Elected_Label']]
    train_feats = train_feats.replace({"Elected_Label":  {"mobile":1,"confined":2, "trapped":3}})
    X = train_feats.iloc[:, 2:-1]
    y = train_feats.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    X_train_, X_test_ = prepare_box_cox_data(X_train), prepare_box_cox_data(X_test)
    X_train_, X_test_ = pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(X_train_), columns=X.columns), pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(X_test_), columns=X.columns)
    for col_name in X_train.columns:
        X_train.eval(f'{col_name} = @X_train_.{col_name}')
        X_test.eval(f'{col_name} = @X_test_.{col_name}')
    pipeline_steps = [("pca", decomposition.PCA()), ("scaler", StandardScaler()), ("SVC", SVC(kernel="rbf"))]
    check_params = {
        "pca__n_components" : [3],
        "SVC__C" : [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
        "SVC__gamma" : [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 1., 5., 10., 50.0],
    }
    pipeline = Pipeline(pipeline_steps)
    create_search_grid = GridSearchCV(pipeline, param_grid=check_params, cv=10)
    create_search_grid.fit(X_train, y_train)
    pipeline.set_params(**create_search_grid.best_params_)
    pipeline.fit(X_train, y_train)
    X = pd.read_csv(Path(pred_data_path), sep=',')
    X = X[['Experiment', 'TrackID', 'NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension']]
    X_label = X['Experiment'].iloc[0]
    X_feats = X.iloc[:, 2:]
    X_feats_ = pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(prepare_box_cox_data(X_feats)), columns=X_feats.columns)
    for col_name in X_feats.columns:
        X_feats.eval(f'{col_name} = @X_feats_.{col_name}')
    X_pred = pipeline.predict(X_feats)
    X['SVMPredictedClass'] = X_pred.astype('int')
    X_outpath = Path(pred_data_path).parents[0] / f'{Path(pred_data_path).stem}_SVMPredicted.csv'
    X.to_csv(X_outpath, sep=',', index=False)

def predict_SPT_class_plots(train_data_path, plot_output_folder_path):
    """produces multiple plots and outputs them to the plot_output_folder_path directory
       includes PC Loadings, PC Scree Plot, PC vs PC scatterplots, SVM confusion matrices

    Args:
        train_data_path (str): complete path to training features data file in .csv format, ex. 'C:/data/tdTomato_37Degree_CytoD_training_feats.csv'
        plot_output_folder_path (str): output folder path for plots
    
    Outputs:
        plots in the plot_output_folder_path
    """
    # plt.rcParams["font.family"] = "Arial"

    plot_colors = { "mobile":"#4daf4a",
                    "intermediate": "#377eb8",
                    "trapped": "#e41a1c"}
    
    def box_cox_features(data_array, pos_delta=1e-15):
        box_coxed_feats = pd.DataFrame(columns=data_array.columns)
        for colidx in list(range(0, len(data_array.columns))):
            if data_array.iloc[:, colidx].min() < 0:
                data_array.iloc[:, colidx] += ((data_array.iloc[:, colidx].min() * -1) + pos_delta)
            fitted_values, fitted_lambda = stats.boxcox(data_array.iloc[:, colidx])
            box_coxed_feats.iloc[:, colidx] = fitted_values
        return box_coxed_feats

    def calc_PCA5_features(data_df):
        feat_name_list = list(data_df.columns)
        # Box cox the features   
        fitteddata = box_cox_features(data_df)
        # Standard scale
        scaler = StandardScaler()
        scaler.fit(fitteddata)
        X=scaler.transform(fitteddata)
        # Compute PCAs
        pca = PCA(n_components=5)
        mypca = pca.fit_transform(X)
        cols = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        mypca_df = pd.DataFrame(data=mypca, columns=cols)
        pca_components = pd.DataFrame(pca.components_.T, columns=cols, index=feat_name_list)
        return mypca_df, pca_components, pca

    def PCA_loadings_plotter(pca_components, save_path):
        fig, axs = plt.subplots(5, 1, figsize=(12,20))
        for pc_axis, ax in zip(pca_components.columns, axs):
            pca_components[pc_axis].plot(   kind='bar', 
                                            color=(pca_components[pc_axis] > 0).map({True: 'g', False: 'r'}), 
                                            rot=0, 
                                            ax=ax, 
                                            title=f"{pc_axis} Loadings")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()

    def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor=facecolor, **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def PCA_versus_PCA_triplet_plotter(data_pca_df, pca_obj, scatter=True, confidence=True, features=True, scaling_factor=1, save_path=None):
        colors = ["#4daf4a", "#377eb8", "#e41a1c"]
        sns.set_palette(sns.color_palette(colors))
        pca_df = data_pca_df.copy()
        electedLabels_to_int = {"mobile":1, "confined":2, "trapped":3}
        if type(pca_df['Elected_Label'].iloc[0]) != np.int32:
            pca_df['Elected_Label'] = pca_df['Elected_Label'].map(electedLabels_to_int).astype('int')
        x_list = ["PC1", "PC1", "PC2"]
        y_list = ["PC2", "PC3", "PC3"]
        with sns.plotting_context("notebook", font_scale=2), sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5)) #, constrained_layout=True)
            i = 0
            for x_pc, y_pc in zip(x_list, y_list):
                if scatter:
                    sns.scatterplot( x=pca_df[x_pc], y=pca_df[y_pc], alpha=1.0, 
                                    hue=pca_df['Elected_Label'], palette=sns.color_palette(),
                                    zorder=1000, ax=axs[i], s=10)
                if confidence:
                    confidence_ellipse(pca_df.loc[  pca_df['Elected_Label'] == 1][x_pc], 
                                                    pca_df.loc[pca_df['Elected_Label'] == 1][y_pc], 
                                                    axs[i], edgecolor='darkgreen', linewidth=3, zorder=1000)
                    confidence_ellipse(pca_df.loc[  pca_df['Elected_Label'] == 2][x_pc], 
                                                    pca_df.loc[pca_df['Elected_Label'] == 2][y_pc], 
                                                    axs[i], edgecolor='darkblue', linewidth=3, zorder=1000)
                    confidence_ellipse(pca_df.loc[  pca_df['Elected_Label'] == 3][x_pc], 
                                                    pca_df.loc[pca_df['Elected_Label'] == 3][y_pc], 
                                                    axs[i], edgecolor='darkred', linewidth=3, zorder=1000)
                axs[i].set_xlabel(f'{x_pc}')
                axs[i].set_ylabel(f'{y_pc}')
                axs[i].set_yticks(np.arange(-8.0, 9.0, 4.0))
                axs[i].set_xticks(np.arange(-8.0, 9.0, 4.0))
                axs[i].set_xlim([-11.0, 11.0])
                axs[i].set_ylim([-11.0, 11.0])
                axs[i].axvline(0, c='k', linewidth=1, zorder=1, alpha=0.5)
                axs[i].axhline(0, c='k', linewidth=1, zorder=1, alpha=0.5)
                axs[i].grid(alpha=0.9, zorder=0)
                axs[i].set_aspect('equal')
                i += 1
            if features:
                annotation_text_size = 16
                pca_comps = pca_obj.components_[:3].T
                # PC1 vs PC2
                # Net Displacement
                axs[0].arrow(0, 0 , pca_comps[0, 0]*scaling_factor, pca_comps[0, 1]*scaling_factor, width=0.2, head_width=0.75, head_length=1, fc='r', ec='k', zorder=2000)
                axs[0].text(pca_comps[0, 0]*10+0.5, pca_comps[0, 1]*10-1.3, "NetDisp", size=annotation_text_size, fontweight="bold", zorder=3000)
                # Straight 
                axs[0].arrow(0, 0 , pca_comps[1, 0]*scaling_factor, pca_comps[1, 1]*scaling_factor, width=0.2, head_width=0.75, head_length=1, fc='r', ec='k', zorder=2000)
                axs[0].text(pca_comps[1, 0]*10-2.0, pca_comps[1, 1]*10+1.5, "Straight", size=annotation_text_size, fontweight="bold", zorder=3000)
                # Asymmetry 
                axs[0].arrow(0, 0 , pca_comps[2, 0]*scaling_factor, pca_comps[2, 1]*scaling_factor, width=0.2, head_width=0.75, head_length=1, fc='r', ec='k', zorder=2000)
                axs[0].text(pca_comps[2, 0]*10+1.25, pca_comps[2, 1]*10-1.5, "Asymm", size=annotation_text_size, fontweight="bold", zorder=3000)
                # RadiusGyration 
                axs[0].arrow(0, 0 , pca_comps[3, 0]*scaling_factor, pca_comps[3, 1]*scaling_factor, width=0.2, head_width=0.75, head_length=1, fc='r', ec='k', zorder=2000)
                axs[0].text(pca_comps[3, 0]*10+0.45, pca_comps[3, 1]*10+0.50, "RadGyr", size=annotation_text_size, fontweight="bold", zorder=3000)
                # Kurtosis 
                axs[0].arrow(0, 0 , pca_comps[4, 0]*scaling_factor, pca_comps[4, 1]*scaling_factor, width=0.2, head_width=0.75, head_length=1, fc='r', ec='k', zorder=2000)
                axs[0].text(pca_comps[4, 0]*5+0.5, pca_comps[4, 1]*5-3, "Kurtosis", size=annotation_text_size, fontweight="bold", zorder=3000)
                # FracDim
                axs[0].arrow(0, 0 , pca_comps[5, 0]*scaling_factor, pca_comps[5, 1]*scaling_factor, width=0.2, head_width=0.75, head_length=1, fc='r', ec='k', zorder=2000)
                axs[0].text(pca_comps[5, 0]*12-3.5, pca_comps[5, 1]*10+1.5, "FracDim", size=annotation_text_size, fontweight="bold", zorder=3000)
                # PC1 vs PC3
                # Net Displacement
                axs[1].arrow(0, 0 , pca_comps[0, 0]*scaling_factor, pca_comps[0, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[1].text(pca_comps[0, 0]*10, pca_comps[0, 2]*10-2, "NetDisp", size=annotation_text_size, fontweight="bold", zorder=3000)
                # Straight 
                axs[1].arrow(0, 0 , pca_comps[1, 0]*scaling_factor, pca_comps[1, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[1].text(pca_comps[1, 0]*10+0.75, pca_comps[1, 2]*10+0, "Straight", size=annotation_text_size, fontweight="bold", zorder=3000)
                # Asymmetry 
                axs[1].arrow(0, 0 , pca_comps[2, 0]*scaling_factor, pca_comps[2, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[1].text(pca_comps[2, 0]*5+1.2, pca_comps[2, 2]*5-2, "Asymm", size=annotation_text_size, fontweight="bold", zorder=3000)
                # RadiusGyration 
                axs[1].arrow(0, 0 , pca_comps[3, 0]*scaling_factor, pca_comps[3, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[1].text(pca_comps[3, 0]*10, pca_comps[3, 2]*10+0.5, "RadGyr", size=annotation_text_size, fontweight="bold", zorder=3000)
                # Kurtosis 
                axs[1].arrow(0, 0 , pca_comps[4, 0]*scaling_factor, pca_comps[4, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[1].text(pca_comps[4, 0]*20-4, pca_comps[4, 2]*20, "Kurtosis", size=annotation_text_size, fontweight="bold", zorder=3000)
                # FracDim
                axs[1].arrow(0, 0 , pca_comps[5, 0]*scaling_factor, pca_comps[5, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[1].text(pca_comps[5, 0]*10-5.5, pca_comps[5, 2]*10-1.25, "FracDim", size=annotation_text_size, fontweight="bold", zorder=3000)
                # PC2 vs PC3
                # Net Displacement
                axs[2].arrow(0, 0 , pca_comps[0, 1]*scaling_factor, pca_comps[0, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[2].text(pca_comps[0, 1]*40-0.5, pca_comps[0, 2]*40-0.2, "NetDisp", size=annotation_text_size, fontweight="bold", zorder=3000)
                # Straight 
                axs[2].arrow(0, 0 , pca_comps[1, 1]*scaling_factor, pca_comps[1, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[2].text(pca_comps[1, 1]*10-1, pca_comps[1, 2]*10+1.5, "Straight", size=annotation_text_size, fontweight="bold", zorder=3000)
                # Asymmetry 
                axs[2].arrow(0, 0 , pca_comps[2, 1]*scaling_factor, pca_comps[2, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[2].text(pca_comps[2, 1]*5+0.5, pca_comps[2, 2]*5-2, "Asymm", size=annotation_text_size, fontweight="bold", zorder=3000)
                # RadiusGyration 
                axs[2].arrow(0, 0 , pca_comps[3, 1]*scaling_factor, pca_comps[3, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[2].text(pca_comps[3, 1]*20-0, pca_comps[3, 2]*20-0.8, "RadGyr", size=annotation_text_size, fontweight="bold", zorder=3000)
                # Kurtosis 
                axs[2].arrow(0, 0 , pca_comps[4, 1]*scaling_factor, pca_comps[4, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[2].text(pca_comps[4, 1]*5-4, pca_comps[4, 2]*5+1, "Kurtosis", size=annotation_text_size, fontweight="bold", zorder=3000)
                # FracDim
                axs[2].arrow(0, 0 , pca_comps[5, 1]*scaling_factor, pca_comps[5, 2]*scaling_factor, width=0.2, head_width=0.5, head_length=1, fc='r', ec='k', zorder=2000)
                axs[2].text(pca_comps[5, 1]*20-0, pca_comps[5, 2]*20-1.2, "FracDim", size=annotation_text_size, fontweight="bold", zorder=3000)
            handles, _ = axs[1].get_legend_handles_labels()
            labels = ["Mobile", "Intermediate", "Trapped"]
            fig.legend(handles, labels, ncol=3, bbox_to_anchor=[0.50, 1.1], loc='upper center')
            for ax in axs.flatten():
                ax.get_legend().remove()
                for spine in ['top', 'right', 'bottom', 'left']:
                    ax.spines[spine].set_linewidth(4)
                    ax.spines[spine].set_color("k")
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

    save_dir = Path(plot_output_folder_path)
    train_feats = pd.read_csv(Path(train_data_path), sep=',')
    train_feats = train_feats.loc[train_feats['Experiment'] == 'tdTomato_37Degree']
    train_feats = train_feats[['Experiment', 'TrackID', 'NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension', 'Elected_Label']]
    electedLabels_to_int = {"mobile":1, "confined":2, "trapped":3}
    train_feats['Elected_Label'] = train_feats['Elected_Label'].map(electedLabels_to_int).astype('int')
    train_feats_only = train_feats.iloc[:, 2:8]
    train_pca_df, train_pca_components, pca_object = calc_PCA5_features(train_feats_only)
    # Combine PCA Features with Experiment, TrackID, & Elected Labels
    train_pca_df['Experiment'] = train_feats['Experiment']
    train_pca_df['TrackID'] = train_feats['TrackID']
    train_pca_df['Elected_Label'] = train_feats['Elected_Label']
    train_pca_df['Elected_Label'] = train_pca_df['Elected_Label'].astype('int')
    # PC Loadings
    PCA_loadings_save_path = save_dir / 'train_PCA_loadings.png'
    PCA_loadings_plotter(train_pca_components, PCA_loadings_save_path)
    # PCA Scree plot
    PC_values = np.arange(pca_object.n_components_) + 1
    with sns.plotting_context("notebook", font_scale=1.5), sns.axes_style("ticks"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        g= sns.lineplot(x=PC_values, y=pca_object.explained_variance_ratio_.cumsum(), color='r', marker="o", markersize=10, linewidth=2, ax=ax)
        # ax.set_title('Scree Plot', y=1.05)
        ax.set_xlabel('Principal Component Number')
        ax.set_ylabel('Cumulative Proportion of Variance Explained')
        ax.set_xticks([1, 2, 3, 4, 5])
        plt.yticks(np.arange(0.6, 1.01, 0.05))
        plt.ylim(0.58, 1.02)
        g.axhline(0.9, c='k', linestyle="--", label=r"90% Explained")
        plt.grid()
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_linewidth(2)
            ax.spines[spine].set_color("k")
        # plt.legend()
        plt.tight_layout()
        plt.savefig((save_dir / 'train_PCA_scree_plot.png'), dpi=100)
        plt.close()
    # PC vs PC scatterplots
    PCA_vs_PCA_triplet_save_path = save_dir / 'train_PCA_vs_PCA_triplet.png'
    PCA_versus_PCA_triplet_plotter(train_pca_df, pca_object, scatter=True, confidence=True, features=True, scaling_factor=10, save_path=PCA_vs_PCA_triplet_save_path)
    # confusion matrices
    train_pca = train_pca_df.drop(['PC4', 'PC5'], axis=1)
    train_pca_ids = train_pca[['Experiment', 'TrackID']]
    train_pca_y = train_pca['Elected_Label']
    train_pca_X_labeled = train_pca.copy()
    train_pca_X = train_pca.drop(['Experiment', 'TrackID', 'Elected_Label'], axis=1)
    train_pca_X_labeled["class_cat"] = pd.cut(train_pca_X_labeled["Elected_Label"], bins=[-1, 0, 1, np.inf], labels=['Mobile', 'Confined', 'Trapped'])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(train_pca_X_labeled, train_pca_X_labeled["class_cat"]):
        train_pca_strat_train_set = train_pca_X_labeled.iloc[train_index].drop("class_cat", axis=1).copy()
        train_pca_strat_test_set = train_pca_X_labeled.iloc[test_index].drop("class_cat", axis=1).copy()
    train_pca_strat_train_set_X = train_pca_strat_train_set.drop("Elected_Label", axis=1)
    train_pca_strat_train_set_y = train_pca_strat_train_set["Elected_Label"].copy()
    train_pca_strat_test_set_X = train_pca_strat_test_set.drop("Elected_Label", axis=1)
    train_pca_strat_test_set_y = train_pca_strat_test_set["Elected_Label"].copy()
    train_pca_strat_train_set_X_Expt_TrackIDs = train_pca_strat_train_set_X[['Experiment', 'TrackID']]
    train_pca_strat_train_set_X = train_pca_strat_train_set_X.drop(['Experiment', 'TrackID'], axis=1)
    train_pca_strat_test_set_X_Expt_TrackIDs = train_pca_strat_test_set_X[['Experiment', 'TrackID']]
    train_pca_strat_test_set_X = train_pca_strat_test_set_X.drop(['Experiment', 'TrackID'], axis=1)
    svm_params = np.array([40.0, 0.03])
    svm_clf = SVC(kernel='rbf', C=svm_params[0], gamma=svm_params[1])
    svm_clf.fit(train_pca_strat_train_set_X, train_pca_strat_train_set_y)
    with sns.plotting_context("notebook", font_scale=1.3), sns.axes_style("white"):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        metrics.ConfusionMatrixDisplay.from_estimator(svm_clf, train_pca_strat_train_set_X, train_pca_strat_train_set_y, normalize='true', values_format='.1f', cmap='Blues', ax=axs[0])
        axs[0].set_title("Train")
        for im in axs[0].get_images():
            im.set_clim(vmin=0,vmax=1)
        metrics.ConfusionMatrixDisplay.from_estimator(svm_clf, train_pca_strat_test_set_X, train_pca_strat_test_set_y, normalize='true', values_format='.1f', cmap='Reds', ax=axs[1])
        axs[1].set_title("Test")
        for im in axs[1].get_images():
            im.set_clim(vmin=0,vmax=1)
        for ax in axs.flatten():
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_linewidth(2)
                ax.spines[spine].set_color("k")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_xticklabels(["Mobile", "Intermediate", "Trapped"], rotation=0, ha='center', rotation_mode='anchor')
            ax.set_yticklabels(["Mobile", "Intermediate", "Trapped"])
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 10
        plt.tight_layout()
        confusion_matrices_save_path = save_dir / 'confusion_matrices.png'
        plt.savefig(confusion_matrices_save_path, dpi=100, bbox_inches='tight')
        plt.close()
