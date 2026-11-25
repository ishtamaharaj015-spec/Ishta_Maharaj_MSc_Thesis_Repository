import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, silhouette_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, TweedieRegressor
import matplotlib.cm as cm
from matplotlib.patches import Patch
import scipy.stats as stats
from scipy.stats import binned_statistic, mannwhitneyu, kstest, norm
from shapely.geometry import box
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial import cKDTree
from shapely.ops import unary_union
import seaborn as sns
from astropy.stats import kuiper, circstats
import statistics
from sklearn.impute import SimpleImputer
from libpysal.weights import KNN, lag_spatial
import xgboost as xgb
from esda.moran import Moran
import shap
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from sklearn.pipeline import Pipeline
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.neighbors import KNeighborsRegressor as KNR, NearestNeighbors
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Rectangle
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import re

FILENAME = "Lineament data.xlsx"            #Name of file with quantified lineament data
SECTOR = "study area"                     #Name of Sector that will be analysed from .xlsx file 
df = pd.read_excel(FILENAME, sheet_name=SECTOR, decimal=',') #Decimal seperator in Excel is a , not a .

bh = pd.read_excel("Borehole data.xlsx", decimal=',') #Borehole data


DF_PRIM = df[df["Primary/Secondary"] == "Primary"]  #Groups data into Primary/Secondary lineaments
DF_SEC = df[df["Primary/Secondary"] == "Secondary"]

def orientation_info(): #Descriptive statistics for the lineament types

    #Descriptive statistics on the length of the primary lineaments
    primary_orientation = DF_PRIM['Orientation (degrees)'] #Selects only orientations for primary layers
    primary_length = DF_PRIM['Length(m)']
    print(f"Number of primary lineaments: {DF_PRIM["Length(m)"].count()}") 
    print(f"Average length of primary layers in {SECTOR}: {primary_length.mean():.2f} m")
    print(f"Standard dev of lengths of primary lineaments in {SECTOR}: {primary_length.std():.1f}")
    print(f"Range in length of prim lineaments: {primary_length.min():.0f} to {primary_length.max():.0f}")
    print(f"Median length for primary: {statistics.median(primary_length)}")
    print(f"Modal length for primary: {statistics.mode(primary_length)}")

    #Distribution of length data for primar lineaments as a histogram (See chapter 5.1 in thesis)
    mu, sigma = primary_length.mean(), primary_length.std()
    x = np.linspace(primary_length.min(), primary_length.max(), 10000)
    sns.displot(data=DF_PRIM, x='Length(m)', bins=50, facecolor='cornflowerblue', edgecolor='black', kde=True, color='red')
    plt.xlabel("Length (m)", fontsize=8)
    plt.ylabel("Frequency", fontsize=8)
    plt.title("Histogram of length data for primary lineaments", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()

    #Descriptive statistics on orientation of primary lineaments
    prim_rad = np.deg2rad(primary_orientation * 2)
    mean_xp = np.mean(np.cos(prim_rad))
    mean_yp = np.mean(np.sin(prim_rad))
    mean_rad_doubled = 0.5 * np.arctan2(mean_yp, mean_xp)
    mean_angle = np.rad2deg(mean_rad_doubled)
    if mean_angle < 0:
        mean_angle += 180
    print(f"Mean of primary orientations: {mean_angle:.0f}")
    prim_rad_norm = prim_rad / 180
    sum_cos = np.sum(np.cos(prim_rad))
    sum_sin = np.sum(np.sin(prim_rad))
    R = np.sqrt(sum_cos**2 + sum_sin**2) / len(prim_rad)
    circ_std = np.sqrt(-2 * np.log(R)) / 2
    print(f"Consistency (R) in primary orientation: {R:.2f}")
    print(f"Circular std dev (deg) of primary orientation: {np.rad2deg(circ_std):.2f}")

    #Descriptive statistics on the length of the secondary lineaments
    secondary_average_orientation = pd.concat([DF_SEC['Orientation (degrees)'], DF_PRIM["Orientation of fault (degrees)"]]) #Combines orientations of faults and other secondary lineaments to calculate avg. orientation
    secondary_average_length = pd.concat([DF_SEC['Length(m)'], DF_PRIM["Length of fault (m)"]])
    print(f"Number of secondary lineaments: {secondary_average_orientation.count()}")
    print(f"Standard dev of lengths of secondary lineaments in {SECTOR}: {secondary_average_length.std():.1f}")
    print(f"Average length of secondary structures in {SECTOR}: {secondary_average_length.mean():.2f} m")
    print(f"Range in length of sec lineaments: {secondary_average_length.min():.0f} to {secondary_average_length.max():.0f}")
    print(f"Median length for secondary: {secondary_average_length.median()}")
    print(f"Modal length for secondary: {secondary_average_length.mode()}")

    #Distribution of length data for secondary lineaments as a histogram (See chapter 5.1 in thesis)
    mu2, sigma2 = secondary_average_length.mean(), secondary_average_length.std()
    y = np.linspace(secondary_average_length.min(), secondary_average_length.max(), 1000)
    sns.displot(data=DF_SEC, x=pd.concat([DF_SEC['Length(m)'], DF_PRIM["Length of fault (m)"]]), bins=50, facecolor='cornflowerblue', edgecolor='black', kde=True, color='red')
    plt.xlabel("Length (m)", fontsize=8)
    plt.ylabel("Frequency", fontsize=8)
    plt.title("Histogram of length data for secondary lineaments", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()

    #Descriptive statistics on orientation of secondary lineaments
    sec_rad = np.deg2rad(secondary_average_orientation * 2)
    mean_xs = np.mean(np.cos(sec_rad))
    mean_ys = np.mean(np.sin(sec_rad))
    mean_rad_doubled_s = 0.5 * np.arctan2(mean_ys, mean_xs)
    mean_angle_s = np.rad2deg(mean_rad_doubled_s)
    if mean_angle_s < 0:
        mean_angle_s += 180
    print(f"Mean of secondary orientations: {mean_angle_s:.0f}")
    sec_rad_norm = sec_rad / 180
    sum_cos_s = np.sum(np.cos(sec_rad))
    sum_sin_s = np.sum(np.sin(sec_rad))
    R = np.sqrt(sum_cos_s**2 + sum_sin_s**2) / len(sec_rad)
    circ_std = np.sqrt(-2 * np.log(R)) / 2
    print(f"Consistency (R) in secondary orientation: {R:.2f}")
    print(f"Circular std dev (deg) of secondary orientation: {np.rad2deg(circ_std):.2f}")

    avg_offset = DF_PRIM["Offset length (m)"].mean() #Calculates average displacement produced by faults
    print(f"Average offset in {SECTOR}: {avg_offset:.2f} m")
    

def correlations(): #Finding the correlation between the fault length and displacement and fault length and orientation

    corr_offset_length_p = DF_PRIM['Offset length (m)'].corr(DF_PRIM['Length of fault (m)'])
    corr_length_orient_p = DF_PRIM['Length of fault (m)'].corr(DF_PRIM['Orientation of fault (degrees)'])
    print(f"Correlation between offset length and fault length in {SECTOR}: {corr_offset_length_p:.2f}")
    print(f"Correlation between fault length and fault orientation in {SECTOR}: {corr_length_orient_p:.2f}")
    return corr_offset_length_p

def orientation_length_graph(corr): #Scatterplot figure with rose diagrams
    
    graph = df.groupby(["Primary/Secondary", "Length(m)"])     
    orientation = graph["Orientation (degrees)"].mean() #Organises the lengths and structure type based on orientation
    primary = orientation.loc["Primary"]    #Selects only orientations and lengths for primary structures
    secondary = orientation.loc["Secondary"]    #Selects only orientations and lengths for secondary structures
    faults = DF_PRIM.groupby("Offset (Y/N)")
    fault_y = faults.get_group("Y")
    fault_y["Offset direction (R/L)"] = fault_y["Offset direction (R/L)"].replace({"R": "L", "L": "R"})


    #Adjusting grid specs for the scatterplots and rose diagrams
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    fig = plt.figure(figsize=(10, 12))
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.18)
    gs_left_outer = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[0], hspace=0.4, height_ratios=[4, 3])
    gs_left_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_left_outer[0], hspace=1.9)
    gs_left_bottom = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_left_outer[1], hspace=0.7)
    gs_right_outer = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1], hspace=0.4, height_ratios=[4, 3])
    gs_right_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_right_outer[0], hspace=1.8)
    gs_right_bottom = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_right_outer[1], hspace=0.4)
    ax_s1 = fig.add_subplot(gs_left_top[0])
    ax_s2 = fig.add_subplot(gs_left_top[1])
    ax_s3 = fig.add_subplot(gs_left_bottom[0])
    ax_s4 = fig.add_subplot(gs_left_bottom[1])
    ax_r1_anchor = fig.add_subplot(gs_right_top[0])
    ax_r1_anchor.axis('off')
    ax_r2_anchor = fig.add_subplot(gs_right_top[1])
    ax_r2_anchor.axis('off')
    ax_r1 = inset_axes(ax_r1_anchor, width='250%', height='250%', loc='center', axes_class=PolarAxes)
    ax_r2 = inset_axes(ax_r2_anchor, width='250%', height='250%', loc='center', axes_class=PolarAxes)
    ax_r3 = fig.add_subplot(gs_right_bottom[0])
    ax_r4 = fig.add_subplot(gs_right_bottom[1])

    ax = ax_s1 #Scatterplot of primary lineaments
    gp_fault_o = df.loc[(df["Offset (Y/N)"] == "Y")]
    point_size = 3
    ax.scatter(DF_PRIM["Length(m)"], DF_PRIM["Orientation (degrees)"], s=point_size, color='blue', label="Primary lineament")
    #ax.set_axis([0, 1100, 0, 180])
    ax.set_yticks(np.arange(0, 181, 30))
    ax.set_xlabel("Length (m)", fontsize=8)
    ax.set_ylabel("Orientation (\N{DEGREE SIGN})", fontsize=8)
    ax.set_title("Length vs orientation of primary lineaments", fontsize=8)

    rose_diag_prim(ax_r1) #Rose diagram (Primary lineaments): refer to function 'def rose_diags_prim()'

    ax = ax_s2 #Scatter plot of secondary lineaments: refer to function 'def secondary_clusters()' for cluster information
    secondary_clusters(ax=ax)
    ax.set_yticks(np.arange(0, 181, 30))
    ax.set_xlabel("Length (m)", fontsize=8)
    ax.set_ylabel("Orientation (\N{DEGREE SIGN})", fontsize=8)
    ax.set_title("Length vs orientation of secondary lineaments", fontsize=8)

    rose_diag_sec(ax_r2) #Rose diagram (Secondary lineaments): refer to function 'def rose_diags_sec()'

    ax = ax_s3 #Scatterplot of fault data
    sinistral = fault_y[fault_y["Offset direction (R/L)"] == "L"]
    dextral = fault_y[fault_y["Offset direction (R/L)"] == "R"]
    ax.scatter(dextral["Length of fault (m)"], dextral["Orientation of fault (degrees)"], color="red", s=point_size, label="Dextral fault")
    ax.scatter(sinistral["Length of fault (m)"], sinistral["Orientation of fault (degrees)"], color="orange", s=point_size, label="Sinistral fault")
    ax.set_yticks(np.arange(0, 181, 30))
    ax.set_xlabel("Length (m)", fontsize=8)
    ax.set_ylabel("Orientation (\N{DEGREE SIGN})", fontsize=8)
    ax.set_title("Length vs orientation of faults", fontsize=8)

    ax = ax_r3 #Scatterplot of correlation between fault length and displacement
    ax.scatter(dextral["Length of fault (m)"], dextral["Offset length (m)"], color='red', s=point_size)
    ax.scatter(sinistral["Length of fault (m)"], sinistral["Offset length (m)"], color='orange', s=point_size)
    sns.regplot(x="Length of fault (m)", y="Offset length (m)", data=DF_PRIM, ax=ax, scatter=False, color='black', line_kws={"linestyle":"--", "linewidth":0.5}, ci=None)
    ax.set_xlabel("Length (m)", fontsize=8)
    ax.set_ylabel("Displacement (m)", fontsize=8)
    ax.set_yticks(np.arange(0, 451, 90))
    ax.set_title("Length of fault vs displacement", fontsize=8)
    ax.text(0.9, 0.6, f"$r$ = {corr:.2f}", transform=ax.transAxes, fontsize=8, ha="right", va="top")

    ax = ax_s4 #Scatterplot of rotated primary lineaments during lateral drag
    if len(df.loc[df["Fault drag (Y/N)"] == "Y"]) > 0:
        ax.scatter(DF_PRIM["Rotated by"], DF_PRIM["Drag orientation (degrees)"], color='darkorchid', s=point_size, label="Rotated primary lineament")
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks(np.arange(0, 181, 20))
        yticks = ax.get_yticks()
        ax.set_yticks([t for t in yticks if t != 0 and t !=180])
        ax.set_xlim(-40, 40)
        ax.set_title("Lateral drag faults that produce rotation", fontsize=8)
        ax.set_xlabel("Amount of rotation (\N{DEGREE SIGN})", fontsize=8)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        ax.set_ylabel("Fault orientation (\N{DEGREE SIGN})", fontsize=8)      #inputs degree sign into graph axis label
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor='black', linewidth=1, clip_on=False))
    else:
        ax.axis('off')

    ax = ax_r4 #Legend fr the scatterplots
    handles, labels = [], []
    for a in [ax_s1, ax_s2, ax_s3, ax_s4, ax_r1, ax_r2, ax_r3, ax_r4]:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = {}
    for handle, label in zip(handles, labels):
        if label not in by_label:
            by_label[label] = handle
    order = ["Primary lineament", "Set 1", "Set 2", "Set 3", "Dextral fault", "Sinistral fault", "Rotated primary lineament", "Fault"] #"Set 4", 
    ordered_handles = [by_label[label] for label in order if label in by_label]
    ordered_labels = [label for label in order if label in by_label]
    ax.legend(ordered_handles, ordered_labels, loc='upper left', fontsize=8, title="Key", ncol=2, title_fontsize=8)
    ax.axis('off')
    for ax in [ax_s3, ax_s4]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 1.2])
    for ax in [ax_s1, ax_s2]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.04, pos.width, pos.height * 2.1])
    for ax in [ax_r3]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 1.2])

    fig.savefig("Orientation vs length of structures in "+SECTOR+"NEW_ROSE.png", dpi=400)
    return fig, ax_s1, ax_s2, ax_s3, ax_s4, ax_r1, ax_r2, ax_r3, ax_r4

#Fault data: Average fault orientation, modal trend
def faults():
    fault_conditions = [
        df["Orientation of fault (degrees)"] == 0,
        (df["Orientation of fault (degrees)"] > 0) & (df["Orientation of fault (degrees)"] < 45),
        df["Orientation of fault (degrees)"] == 45,
        (df["Orientation of fault (degrees)"] > 45) & (df["Orientation of fault (degrees)"] < 90),
        df["Orientation of fault (degrees)"] == 90,
        (df["Orientation of fault (degrees)"] > 90) & (df["Orientation of fault (degrees)"] < 135),
        df["Orientation of fault (degrees)"] == 135,
        (df["Orientation of fault (degrees)"] > 135) & (df["Orientation of fault (degrees)"] < 180),
        df["Orientation of fault (degrees)"] == 180
    ]

    fault_choices = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S"]
    df['Fault trend'] = np.select(fault_conditions, fault_choices, default="Unkown")
    trend_order = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S"]
    trend_number = {g: i for i, g in enumerate(trend_order)}
    faults = df.groupby("Offset (Y/N)")
    fault_orientation = faults.get_group("Y")
    fault_orientation_y = fault_orientation["Orientation of fault (degrees)"]
    fault_length = DF_PRIM["Length of fault (m)"]
    print(f"Average length of faults in {SECTOR}: {fault_length.mean():.2f} m")
    print("Mode fault trend:", fault_orientation["Fault trend"].mode())

    fault_rad = np.deg2rad(fault_orientation_y * 2)
    mean_xp = np.mean(np.cos(fault_rad))
    mean_yp = np.mean(np.sin(fault_rad))
    mean_rad_doubled = 0.5 * np.arctan2(mean_yp, mean_xp)
    mean_angle = np.rad2deg(mean_rad_doubled)
    if mean_angle < 0:
        mean_angle += 180
    print(f"Mean of fault orientations: {mean_angle:.0f}")
    fault_rad_norm = fault_rad / 180
    sum_cos = np.sum(np.cos(fault_rad))
    sum_sin = np.sum(np.sin(fault_rad))
    R = np.sqrt(sum_cos**2 + sum_sin**2) / len(fault_rad)
    circ_std = np.sqrt(-2 * np.log(R)) / 2
    print(f"Consistency (R) in fault orientation: {R:.2f}")
    print(f"Circular std dev (deg) of fault orientation: {np.rad2deg(circ_std):.2f}")

#Lateral drag data
def offset_drag():
    rotated_layers_only = df.groupby("Fault drag (Y/N)")
    rotated_layers = rotated_layers_only.get_group("Y")
    print(f"Average orientation of clockwise rotated layers: {rotated_layers.loc[rotated_layers['Drag direction (C/A)'] == 'C', 'Orientation (degrees)'].mean():.2f}\N{DEGREE SIGN}")
    print(f"Average orientation of anticlockwise rotated layers: {rotated_layers.loc[rotated_layers['Drag direction (C/A)'] == 'A', 'Orientation (degrees)'].mean():.2f}\N{DEGREE SIGN}")
    print(f"Avg length of faults that cause drag: {rotated_layers['Drag length(m)'].mean():.2f} m")
    print(f"Avg orientation of faults that cause drag: {rotated_layers['Drag orientation (degrees)'].mean():.2f}\N{DEGREE SIGN}")

#Number of sinistral and dextral faults and average orientation of each
def fault_groups():
    faults = DF_PRIM.groupby("Offset (Y/N)")
    fault_y = faults.get_group("Y")
    fault_y["Offset direction (R/L)"] = fault_y["Offset direction (R/L)"].replace({"R": "L", "L": "R"})
    fault_unique = fault_y.drop_duplicates(subset="Orientation of fault (degrees)")
    print(f"Number of sinistral faults: {(fault_unique["Offset direction (R/L)"] == "L").sum()}\nNumber of dextral faults: {(fault_unique['Offset direction (R/L)'] == "R").sum()}")
    mean_r = fault_y[fault_y["Offset direction (R/L)"] == "R"]["Orientation of fault (degrees)"].mean()
    mean_l = fault_y[fault_y["Offset direction (R/L)"] == "L"]["Orientation of fault (degrees)"].mean()
    print("Avg O of dextral faults:", mean_r)
    print("Avg O of sinistral faults", mean_l)

def fault_clusters():
    faults = DF_PRIM.groupby("Offset (Y/N)")
    fault_y = faults.get_group("Y")
    faults_rad = np.deg2rad(fault_y["Orientation of fault (degrees)"].to_numpy() * 2)
    fault_length =  fault_y["Length of fault (m)"].to_numpy()
    fault_displacement = fault_y["Offset length (m)"].to_numpy()

    X = np.column_stack((np.cos(faults_rad), np.sin(faults_rad)))
    inertia = []
    K = range(1,10)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(X)
        inertia.append(kmeans.inertia_)

    kmeans = KMeans(n_clusters=3).fit(X)
    labels = kmeans.labels_
    for k in range(3):
        cluster_angles = faults_rad[labels == k]
        cluster_lengths = fault_length[labels == k]
        cluster_displacement = fault_displacement[labels == k]
        C = np.mean(np.cos(cluster_angles))
        S = np.mean(np.sin(cluster_angles))
        mean_angle = np.rad2deg(np.arctan2(S, C)) / 2
        if mean_angle < 0:
            mean_angle += 180
        R = np.sqrt(C**2 + S**2)
        circ_std = np.degrees(np.sqrt(-2*np.log(R)))
        xc = np.corrcoef(cluster_lengths, np.cos(cluster_angles))[0,1]
        xs = np.corrcoef(cluster_lengths, np.sin(cluster_angles))[0,1]
        cs = np.corrcoef(np.cos(cluster_angles), np.sin(cluster_angles))[0,1]
        rho = np.sqrt((xc**2 + xs**2 - 2*xc*xs*cs) / (1 - cs**2))
        corr_l_d = np.corrcoef(cluster_lengths, cluster_displacement)[0,1]

        #Orientation information for each fault set
        print(f"Fault data:\n")
        print(f"Mean orientation: {mean_angle:.2f}")
        print(f"Relative mean length: {R:.2f}")
        print(f"Circular standard deviation: {circ_std:.2f}")
        print(f"Corr btw length and orientation: {rho:.2f}")
        print(f"Corr btw length and displacement: {corr_l_d:.2f}\n")


def secondary_clusters(ax = None):

    COLOURS = ["limegreen", "teal", "darkgoldenrod"] 
    COLOURS4 = ["limegreen", "teal", "darkgoldenrod", "cornflowerblue"]
    if ax is None:
        fig, ax = plt.subplots()
    secondary_rad = np.deg2rad(DF_SEC["Orientation (degrees)"].to_numpy() * 2)
    secondary_length =  DF_SEC["Length(m)"].to_numpy()
    X = np.column_stack((np.cos(secondary_rad), np.sin(secondary_rad)))
    inertia = []
    K = range(1,10)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(X)
        inertia.append(kmeans.inertia_)

    print("Secondary lineaments:\n")
    kmeans = KMeans(n_clusters=3).fit(X)
    labels = kmeans.labels_
    cluster_means = []
    for k in range(3):
        cluster_angles = secondary_rad[labels == k]
        cluster_lengths = secondary_length[labels == k]
        C = np.mean(np.cos(cluster_angles))
        S = np.mean(np.sin(cluster_angles))
        mean_angle = np.rad2deg(np.arctan2(S, C)) / 2
        if mean_angle < 0:
            mean_angle += 180
        cluster_means.append((k, mean_angle))        

    cluster_means_sorted = sorted(cluster_means, key=lambda x: x[1])
    cluster_map ={old_label: new_label for new_label, (old_label, _) in enumerate(cluster_means_sorted)}
    for k in range(3):
        cluster_angles = secondary_rad[labels == k]
        cluster_lengths = secondary_length[labels == k]
        C = np.mean(np.cos(cluster_angles))
        S = np.mean(np.sin(cluster_angles))
        mean_angle = np.rad2deg(np.arctan2(S, C)) / 2
        if mean_angle < 0:
            mean_angle += 180
        new_k = cluster_map[k]
        R = np.sqrt(C**2 + S**2)
        circ_std = np.degrees(np.sqrt(-2*np.log(R)))
        xc = np.corrcoef(cluster_lengths, np.cos(cluster_angles))[0,1]
        xs = np.corrcoef(cluster_lengths, np.sin(cluster_angles))[0,1]
        cs = np.corrcoef(np.cos(cluster_angles), np.sin(cluster_angles))[0,1]
        rho = np.sqrt((xc**2 + xs**2 - 2*xc*xs*cs) / (1 - cs**2))

        #Orientation information for each set of secondary lineaments
        print(f"Mean orientation: {mean_angle:.2f}")
        print(f"Relative mean length: {R:.2f}")
        print(f"Circular standard deviation: {circ_std:.2f}")
        print(f"Corr btw length and orientation: {rho:.2f}\n")
        ax.scatter(cluster_lengths, (np.rad2deg(cluster_angles) / 2), s=5, color=COLOURS[new_k], label=f"Set {new_k+1}")

def rose_diag_sec(ax):
    secondary_average_orientation = pd.concat([DF_SEC['Orientation (degrees)'], DF_PRIM["Orientation of fault (degrees)"]])
    angles_doubled = np.concatenate([secondary_average_orientation, secondary_average_orientation + 180])
    faults_doubled = np.concatenate([DF_PRIM["Orientation of fault (degrees)"], DF_PRIM["Orientation of fault (degrees)"] + 180])
    angles_rad = np.deg2rad(angles_doubled)
    fault_rad = np.deg2rad(faults_doubled)
    num_bins = 40
    bins = np.linspace(0, 2*np.pi, num_bins + 1)

    counts1, _ = np.histogram(angles_rad, bins=bins)
    counts2, _ = np.histogram(fault_rad, bins=bins)

    width = (2*np.pi) / num_bins
    bars1 = ax.bar(bins[:-1], counts1, width=width, align='edge')
    bars2 = ax.bar(bins[:-1], counts2, width=width, align='edge', bottom=counts1)
    for bar in bars1:
        bar.set_facecolor('limegreen')
        bar.set_edgecolor('black')
    for bar in bars2:
        bar.set_facecolor('green')
        bar.set_edgecolor('black')

    legend = [Patch(facecolor='limegreen', label=f"Lineaments (n = {secondary_average_orientation.count()})"), Patch(facecolor='green', label=f"Faults (n = {DF_PRIM['Orientation of fault (degrees)'].nunique()})")]
    ax.legend(handles=legend, loc='center left', bbox_to_anchor=(0.75, -0.01), borderaxespad=0, fontsize=8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(alpha=0.2)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels([])
    ax.set_rlabel_position(20)
    ticks = ax.get_yticks()
    labels = []
    for i, t in enumerate(ticks):
        if i in [1, 3, 2]:
            labels.append(str(int(t)))
        else:
            labels.append('')
    ax.set_yticklabels(labels)
    ax.tick_params(axis='y', labelsize=8, pad=-10)
    for label in ax.get_yticklabels():
        label.set_verticalalignment('center')
    ax.set_title(f"Rose diagram of secondary lineaments in {SECTOR}", fontsize=8)


def rose_diag_prim(ax):
    
    angles_doubled = np.concatenate([DF_PRIM["Orientation (degrees)"], DF_PRIM["Orientation (degrees)"] + 180])
    angles_rad = np.deg2rad(angles_doubled)
    
    num_bins = 40
    bins = np.linspace(0, 2*np.pi, num_bins + 1)

    counts, _ = np.histogram(angles_rad, bins=bins)

    width = (2*np.pi) / num_bins
    bars = ax.bar(bins[:-1], counts, width=width, align='edge')
    for bar in bars:
        bar.set_facecolor('cornflowerblue')
        bar.set_edgecolor('black')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(alpha=0.2)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels([])
    ax.set_rlabel_position(90)
    ticks = ax.get_yticks()
    labels = []
    for i, t in enumerate(ticks):
        if i in [0, 2, 4]:
            labels.append(str(int(t)))
        else:
            labels.append('')
    ax.set_yticklabels(labels)
    ax.tick_params(axis='y', labelsize=8, pad=-10)
    for label in ax.get_yticklabels():
        label.set_verticalalignment('center')
    ax.set_title(f"Rose diagram of primary lineaments in {SECTOR}", fontsize=8)

#Distribution of borehole data: Stone distribution, parameter distribution
def bh_data():
    print(f"Min depth of bh: {bh['Tot_Pen'].min():.3f} m")
    print(f"Max depth of bh: {bh['Tot_Pen'].max():.3f} m")
    print(f"Avg depth of bh: {bh['Tot_Pen'].mean():.3f} m")
    print()
    fw = bh.groupby("FW_Reached")
    fw_y = fw.get_group("Y")
    fw_n = fw.get_group("N")
    print(f"FW reached: {fw_y["FW_Reached"].count()} bhs which is {((fw_y["FW_Reached"].count())/(bh["FW_Reached"].count())) * 100:.2f} %")
    print(f"FW not reached: {fw_n["FW_Reached"].count()} bhs which is {((fw_n["FW_Reached"].count())/(bh["FW_Reached"].count())) * 100:.2f} %")
    print()
    fw_type = bh["FW_Type"].value_counts().to_frame('count')
    fw_type["Percentage"] = bh["FW_Type"].value_counts(normalize=True) * 100
    print(fw_type)
    print()
    fw_desc = bh["FW_Desc"].value_counts().to_frame('count')
    fw_desc["Percentage"] = bh["FW_Desc"].value_counts(normalize=True) * 100
    print(fw_desc)
    print()
    epoch = bh["Epoch"].value_counts().to_frame('count')
    epoch["Percentage"] = bh["Epoch"].value_counts(normalize=True) * 100
    print(epoch)
    print()
    print(f"Min number of g units: {bh['G_Units'].min()}")
    print(f"Max number of g units: {bh['G_Units'].max()}")
    g_units = bh["G_Units"].value_counts().to_frame('count')
    g_units["Percentage"] = bh["G_Units"].value_counts(normalize=True) * 100
    print(g_units)
    print()
    print(f"Min g thickness: {bh['G_Thickns'].min():.3f} m")
    print(f"Max g thickness: {bh['G_Thickns'].max():.3f} m")
    print(f"Avg g thickness: {bh['G_Thickns'].mean():.3f} m")
    print()
    print(f"Total stones: {bh["Stones"].sum()}")
    colour = (bh['Colour'].dropna().str.split(',').explode().str.strip())
    colour_codes = r'WHT|YLW|BRN|OWH|GRY|PNK|L/BRN|FRO|ORN|BLU|BLK|GRN|CLR|L/YLW|PPL'
    #print(colour_counts)
    clean_colours = colour.str.findall(colour_codes).explode()
    clean_colours = clean_colours.dropna()
    colour_counts = clean_colours.value_counts()
    print(colour_counts)

    colour_names = {
        "WHT": "White",
        "YLW": "Yellow",
        "BRN": "Brown",
        "OWH": "Off white",
        "GRY": "Grey",
        "PNK": "Pink",
        "ORN": "Orange",
        "L/BRN": "Light Brown",
        "FRO": "Frost",
        "BLU": "Blue",
        "GRN": "Green",
        "L/YLW": "Light Yellow",
        "CLR": "Clear",
        "PPL": "Purple"
    }

    bh["Stone_colour_full"] = bh["Colour"].map(colour_names)
    names_total = bh["Stone_colour_full"].value_counts()
    small_colour = names_total[names_total < 32]
    large_colour = names_total[names_total > 32]
    if len(small_colour) > 0:
        large_colour['Other'] = small_colour.sum()
    plt.figure()
    plt.pie(large_colour, labels=large_colour.index, autopct='%1.0f%%', textprops={"fontsize": 12})

    total_carats =  bh["Act_Carats"].sum()
    print(f"Total carats retreived: {total_carats}")
    clarity = (bh['Clarity'].dropna().str.split(',').explode().str.strip().value_counts())
    print(clarity)

    rock_type_map = {
        "BAS": "Basalt",
        "BISCH": "Biotite schist",
        "BRC": "Breccia",
        "CASCH": "Calc-shist",
        "CNG": "Conglomerate",
        "COQ": "Coquina",
        "CY": "Clay",
        "CYS": "Clayey sand",
        "CYST": "Claystone",
        "DIA": "Diamctite",
        "G": "Gravel",
        "GNS": "Gneiss",
        "GRAN": "Granite",
        "GYWK": "Greywacke",
        "INTR": "Intrusive",
        "LST": "Limestone",
        "MBL": "Marble",
        "MCY": "Marine clay",
        "MISCH": "Mica schist",
        "MLST": "Meta limestone",
        "MSST": "Marine sandstone",
        "MST": "Mudstone",
        "NA": "N/A",
        "QZ": "Quartz",
        "QZMISCH": "Quartz-mica schist",
        "QZSCH": "Quartz schist",
        "QZT": "Quartzite", 
        "PHY": "Phyllite",
        "RBL": "Rubble",
        "S": "Sand",
        "SAP": "Saprolite",
        "SCH": "Schist",
        "SCY": "Sandy clay",
        "SHCNG": "Shelly conglomerate",
        "SHSST": "Shelly sandstone",
        "SLCT": "Silcrete",
        "SLTST": "Siltstone",
        "SST": "Sandstone",
        "SLT": "Slate",
        "SSTRBL": "Sandstone rubble",
        "TCY": "Terrigenous clay",
        "UNK": "Unknown",
        "VOLC": "Volcanics",
        "VQ": "Vein quartz",
        "DOL": "Dolomite"
    }

    bh["FW_Type_full"] = bh["FW_Type"].map(rock_type_map)
    fig, axes = plt.subplots(2, 2, figsize=(10,7), constrained_layout=True)
    ax = axes[0][0]
    fw_counts = bh['FW_Type_full'].value_counts()
    small = fw_counts[fw_counts < 597]
    large = fw_counts[fw_counts > 597]
    if len(small) > 0:
        large['Other'] = small.sum()
    ax.pie(large, labels=large.index, autopct='%1.0f%%')
    ax.set_title("Footwall lithology")
    ax = axes[0][1]
    fw_d_counts = bh['FW_Desc'].value_counts()
    small = fw_d_counts[fw_d_counts < 323]
    large = fw_d_counts[fw_d_counts > 323]
    if len(small) > 0:
        large['Other'] = small.sum()
    ax.pie(large, labels=large.index.str.title(), autopct='%1.0f%%')
    ax.set_title("Footwall texture")
    ax = axes[1][0]
    epoch_counts = bh['Epoch'].value_counts()
    small = epoch_counts[epoch_counts < 739]
    large = epoch_counts[epoch_counts > 739]
    if len(small) > 0:
        large['Other'] = small.sum()
    ax.pie(large, labels=large.index, autopct='%1.0f%%')
    ax.set_title("Epoch")
    ax = axes[1][1]
    ax.bar(bh["G_Units"], bh["G_Thickns"])
    ax.set_xlabel("Number of gravel units")
    ax.set_ylabel("Gravel thickness (m)")
    ax.set_title("Gravel distribution throughout the study area")
    ax.set_xlim([0.5, 7.5])
    plt.savefig("Borehole parameter distribution.jpeg", dpi=400)

#Plots for borehole data
def normalised():
    faults = DF_PRIM[["Fault start E", "Fault start S", "Fault end E", "Fault end S", "Length of fault (m)", "Orientation of fault (degrees)", "Sector"]].copy()
    faults.columns = ["E start", "S start", "E end", "S end", "L", "O", "Sector"]
    faults["Type"] = "Fault"
    secondary = DF_SEC[["Start E", "Start S", "End E", "End S", "Length(m)", "Orientation (degrees)", "Sector"]].copy()
    secondary.columns = ["E start", "S start", "E end", "S end", "L", "O", "Sector"]
    secondary["Type"] = "Secondary"
    primary = DF_PRIM[["Start E", "Start S", "End E", "End S", "Length(m)", "Orientation (degrees)", "Sector"]].copy()
    primary.columns = ["E start", "S start", "E end", "S end", "L", "O", "Sector"]
    primary["Type"] = "Primary"
    structures_clean = pd.concat([faults, secondary, primary], ignore_index=True)
    structures_clean2 = pd.concat([faults, secondary], ignore_index=True) 
    structures_clean["Structure ID"] = structures_clean.index
    structures_clean2["Structure ID"] = structures_clean2.index

    bh_gdf = gpd.GeoDataFrame(
        bh,
        geometry=gpd.points_from_xy(bh["E_actual"], bh["N_actual"]),
        crs="EPSG:32733"
    )
    structures_gdf = gpd.GeoDataFrame(
        structures_clean,
        geometry=structures_clean.apply(
            lambda row: LineString([
                (row["E start"], row["S start"]), 
                (row["E end"], row["S end"])
                ]),
            axis=1
        ),
        crs="EPSG:32733"
    )
    structures_gdf2 = gpd.GeoDataFrame(
        structures_clean2,
        geometry=structures_clean2.apply(
            lambda row: LineString([
                (row["E start"], row["S start"]), 
                (row["E end"], row["S end"])
                ]),
            axis=1
        ),
        crs="EPSG:32733"
    )

    result = gpd.sjoin_nearest(
        bh_gdf,
        structures_gdf,
        how="left",
        distance_col="distance_m",
        #max_distance=50
    )
    result2 = gpd.sjoin_nearest(
        bh_gdf,
        structures_gdf2,
        how="left",
        distance_col="distance_m",
        #max_distance=50
    )
    pd.options.display.max_rows = 10000

    result = result.dropna(subset=["Stones", "Tot_Pen", "FW_Pen", "SedT_Act", "Sed_Tot", "FW_Reached", "FW_Type", "FW_Desc", "G_Units", "G_Thickns", "Potential"])
    #print(result[["Sample_ID", "Stones", "Epoch", "L", "O", "Sector", "Tot_Pen", "FW_Pen", "SedT_Act", "Sed_Tot", "FW_Reached", "FW_Type", "FW_Desc", "G_Units", "G_Thickns", "Potential"]].sort_values(by="Stones"))
    result["Epoch"] = result["Epoch"].replace({"PLeistocene": "Pleistocene", "PLEistocene": "Pleistocene"})
    result["FW_Desc"] = result["FW_Desc"].replace({"Friable": "FRIABLE"})
    result = result.reset_index(drop=True)

    result2 = result2.dropna(subset=["Stones", "Tot_Pen", "FW_Pen", "SedT_Act", "Sed_Tot", "FW_Reached", "FW_Type", "FW_Desc", "G_Units", "G_Thickns", "Potential"])
    #print(result[["Sample_ID", "Stones", "Epoch", "L", "O", "Sector", "Tot_Pen", "FW_Pen", "SedT_Act", "Sed_Tot", "FW_Reached", "FW_Type", "FW_Desc", "G_Units", "G_Thickns", "Potential"]].sort_values(by="Stones"))
    result2["Epoch"] = result2["Epoch"].replace({"PLeistocene": "Pleistocene", "PLEistocene": "Pleistocene"})
    result2["FW_Desc"] = result2["FW_Desc"].replace({"Friable": "FRIABLE"})

    #Descriptive statistics on diamond distribution
    bh_per_sector = result["Sector"].value_counts()
    print("Boreholes per sector:")
    print(bh_per_sector)
    diamonds_per_sector = result.groupby("Sector")["Stones"].sum()
    print("Diamonds per sector:")
    print(diamonds_per_sector)
    sums, edges, _ = binned_statistic(result["O"], result["Stones"], statistic="sum", bins=90)
    #plt.scatter(result["O"], result["Stones"], color="blue", s=6)
    bins = np.arange(0, 181, 2)
    result["O_bin"] = pd.cut(result["O"], bins=bins, right=False, include_lowest=True)
    diamonds_o = (result.groupby("O_bin")["Stones"].sum().reset_index())
    print("Diamonds per orientation group: ")
    print(diamonds_o)
    length_by_o = ( result.groupby("O_bin")["L"].sum().reset_index())
    diamonds_o = diamonds_o.merge(length_by_o, on="O_bin")
    diamonds_o["Diamonds per m"] = (diamonds_o["Stones"] / diamonds_o["L"])
    bh_counts = (result.groupby("Sample_ID").size().reset_index(name="bh_count"))
    result = result.merge(bh_counts, on="Sample_ID", how="left")
    result["bh_count"] = result["bh_count"].fillna(0)
    bh_by_bin = (result.groupby("O_bin")["bh_count"].sum().reset_index())
    diamonds_o = diamonds_o.merge(bh_by_bin, on="O_bin")
    diamonds_o["Diamonds per bh"] = (diamonds_o["Stones"] / diamonds_o["bh_count"])
    result["Stones per bh"] = diamonds_o["Diamonds per bh"]

    def binned_rate(df, value_col='Stones', orient_col='O', bins=90, orient_range=(0,180)):
        bin_edges = np.linspace(*orient_range, bins + 1)
        full_index = pd.IntervalIndex.from_breaks(bin_edges, closed='right')
        df = df.copy()
        df['O_bin'] = pd.cut(df[orient_col], bins=bin_edges, include_lowest=True)
        grouped = df.groupby("O_bin", observed=True)[value_col].agg(['mean', 'count', 'sum'])
        grouped = grouped.reindex(full_index)
        out = grouped.reset_index().rename(columns={'index': 'O_bin'})
        out['bin_center'] = full_index.mid.astype(float)
        return out
    rate_primary   = binned_rate(result[result["Type"] == "Primary"])
    rate_secondary = binned_rate(result[result["Type"] == "Secondary"])
    rate_fault     = binned_rate(result[result["Type"] == "Fault"])

    #Orientation vs diamond distribution
    fig, ax1 = plt.subplots()
    ax1.plot(rate_primary["bin_center"],   rate_primary["mean"],   color="blue",  label="Primary lineament")
    ax1.plot(rate_secondary["bin_center"], rate_secondary["mean"], color="limegreen",   label="Secondary lineament")
    ax1.plot(rate_fault["bin_center"],     rate_fault["mean"],     color="green", label="Fault")
    ax1.set_xlabel("Orientation (\N{DEGREE SIGN})", fontsize=10)
    ax1.set_ylabel("Mean stones per borehole", fontsize=10)
    ax1.legend(loc='upper center', fontsize=9)
    ax1.tick_params(axis='both', labelsize=10)
    ax2 = ax1.twinx()
    ax2.bar(rate_primary["bin_center"],   rate_primary["count"],   alpha=0.2, color="blue",  width=2)
    ax2.bar(rate_secondary["bin_center"], rate_secondary["count"], alpha=0.2, color="limegreen",   width=2)
    ax2.bar(rate_fault["bin_center"],     rate_fault["count"],     alpha=0.2, color="green", width=2)
    ax2.set_ylabel("Boreholes per bin", fontsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    plt.title("Stone distribution vs lineament orientation", fontsize=10)
    plt.savefig("Stones vs orientation.png", dpi=400)

    def permutation_null(result_sub, structures_sub, n_iter=2000, bins=90, orient_range=(0, 180), value_col="Stones", id_col="Structure ID", orient_col="O"):
        bin_edges = np.linspace(*orient_range, bins + 1)
        full_index = pd.IntervalIndex.from_breaks(bin_edges, closed='right')
        null_matrix = np.full((n_iter, bins), np.nan)

        o_lookup = structures_sub.set_index(id_col)[orient_col]
        struct_ids = o_lookup.index.values
        o_values = o_lookup.values
        borehole_struct_ids = result_sub[id_col].values
        stones = result_sub[value_col].values

        for i in range(n_iter):
            shuffled_o = np.random.permutation(o_values)
            shuffle_map = pd.Series(shuffled_o, index=struct_ids)
            temp_O = shuffle_map.loc[borehole_struct_ids].values
            temp_bin = pd.cut(temp_O, bins=bin_edges, include_lowest=True)
            null_matrix[i] = (
                pd.Series(stones)
                .groupby(temp_bin, observed=True)
                .mean()
                .reindex(full_index)
                .values
            )
        return null_matrix
    results_by_type = {}
    for lineament_type in ['Primary', 'Secondary', 'Fault']:
        structures_sub = structures_clean[structures_clean['Type'] == lineament_type]
        result_sub = result[result['Type'] == lineament_type]
        null_matrix = permutation_null(result_sub, structures_sub, n_iter=2000)
        results_by_type[lineament_type] = null_matrix
    bin_edges = np.linspace(0, 180, 91)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    rate_lookup = {'Primary': rate_primary, 'Secondary': rate_secondary, 'Fault': rate_fault}
    colors = {'Primary': 'blue', 'Secondary': 'green', 'Fault': 'darkgreen'}
    for lineament_type in ['Primary', 'Secondary', 'Fault']:
        null_matrix = results_by_type[lineament_type]
        lower = np.nanpercentile(null_matrix, 2.5, axis=0)
        upper = np.nanpercentile(null_matrix, 97.5, axis=0)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(bin_centers, lower, upper, color="grey", alpha=0.3, label="95% null envelope")
        ax.plot(bin_centers, rate_lookup[lineament_type]["mean"], color=colors[lineament_type],
                label=f"Observed ({lineament_type})")
        ax.set_title(f"{lineament_type}: observed vs null")
        ax.set_xlabel("Orientation (°)")
        ax.set_ylabel("Mean stones per borehole")
        ax.legend()
        plt.tight_layout()

    rock_type_map = {
        "BAS": "Basalt",
        "BISCH": "Biotite schist",
        "BRC": "Breccia",
        "CASCH": "Calc-shist",
        "CNG": "Conglomerate",
        "COQ": "Coquina",
        "CY": "Clay",
        "CYS": "Clayey sand",
        "CYST": "Claystone",
        "DIA": "Diamctite",
        "G": "Gravel",
        "GNS": "Gneiss",
        "GRAN": "Granite",
        "GYWK": "Greywacke",
        "INTR": "Intrusive",
        "LST": "Limestone",
        "MBL": "Marble",
        "MCY": "Marine clay",
        "MISCH": "Mica schist",
        "MLST": "Meta limestone",
        "MSST": "Marine sandstone",
        "MST": "Mudstone",
        "NA": "N/A",
        "QZ": "Quartz",
        "QZMISCH": "Quartz-mica schist",
        "QZSCH": "Quartz schist",
        "QZT": "Quartzite", 
        "PHY": "Phyllite",
        "RBL": "Rubble",
        "S": "Sand",
        "SAP": "Saprolite",
        "SCH": "Schist",
        "SCY": "Sandy clay",
        "SHCNG": "Shelly conglomerate",
        "SHSST": "Shelly sandstone",
        "SLCT": "Silcrete",
        "SLTST": "Siltstone",
        "SST": "Sandstone",
        "SLT": "Slate",
        "SSTRBL": "Sandstone rubble",
        "TCY": "Terrigenous clay",
        "UNK": "Unknown",
        "VOLC": "Volcanics",
        "VQ": "Vein quartz",
        "DOL": "Dolomite"
    }
    result["FW_Type_full"] = result["FW_Type"].map(rock_type_map)
    plt.figure()
    result = result.reset_index(drop=True)
    lithology_counts = result.groupby('FW_Type_full')['Stones'].sum()
    lithology_counts = lithology_counts[lithology_counts > 50]
    sns.barplot(x=lithology_counts.index, y=lithology_counts.values, errorbar=None)
    plt.title("Stone distribution per footwall lithology", fontsize=10)
    plt.xlabel("Lithology", fontsize=10)
    plt.ylabel("Stones", fontsize=10)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("Footwall litho vs stones.png", dpi=400)

    plt.figure()
    sns.barplot(x="G_Units", y="Stones", data=result, estimator=sum, errorbar=None)
    plt.title("Stone distribution vs gravel distribution", fontsize=10)
    plt.xlabel("Number of gravel layers", fontsize=10)
    plt.ylabel("Stones", fontsize=10)
    plt.xticks(rotation=90)
    plt.xlim(0, 6)
    plt.tight_layout()
    plt.savefig("gravel distrib vs stones.jpeg", dpi=400)

    plt.figure()
    x = result["FW_Desc"].str.title()
    competency_counts = result.groupby('FW_Desc')['Stones'].sum()
    competency_counts = competency_counts[competency_counts > 50]
    sns.barplot(x=competency_counts.index.str.title(), y=competency_counts.values, errorbar=None)
    plt.title("Stone distribution per footwall competency", fontsize=10)
    plt.xlabel("Footwall competency", fontsize=10)
    plt.ylabel("Stones", fontsize=10)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("Footwall competency vs stones.jpeg", dpi=400)

    order = ["Precambrian", "Cretaceous", "Tertiary", "Miocene", "Pleistocene", "Quaternary", "Holocene"]
    result_new = result.reset_index(drop=True)
    plt.figure()
    sns.barplot(x="Epoch", y="Stones", data=result_new, estimator=sum, order=order, errorbar=None)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Stone count", fontsize=10)
    plt.title("Total stone count per Epoch", fontsize=10)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("Epoch vs stones.jpeg", dpi=400)

    plt.figure()
    bins= np.arange(0, 10.5, 0.5)
    result["binned_thickns"] = pd.cut(result["G_Thickns"], bins=bins, right=False)
    grouped = result.groupby("binned_thickns")["Stones"].sum()
    x = bins[:-1]
    plt.bar(x, grouped.values, width=0.5, align='edge', edgecolor="white")
    #sns.barplot(x="Tot_Pen", y="Stones", data=result, estimator=sum, errorbar=None)
    plt.title("Stone distribution vs gravel thickness", fontsize=10)
    #plt.xticks(range(21))
    plt.xlim(0, 5)
    plt.xlabel("Gravel thickness (m)", fontsize=10)
    plt.ylabel("Stones", fontsize=10)
    plt.xticks(np.arange(0, 6, 1))
    plt.tight_layout()
    plt.savefig("gravel thickness vs stones.jpeg", dpi=400)

    plt.figure()
    bins= np.arange(0, 10.5, 0.5)
    result["binned_depth"] = pd.cut(result["Tot_Pen"], bins=bins, right=False)
    grouped = result.groupby("binned_depth")["Stones"].sum()
    x = bins[:-1]
    plt.bar(x, grouped.values, width=0.5, align='edge', edgecolor="white")
    #sns.barplot(x="Tot_Pen", y="Stones", data=result, estimator=sum, errorbar=None)
    plt.title("Stone distribution vs borehole depth", fontsize=10)
    #plt.xticks(range(21))
    plt.xlim(0, 10)
    plt.xlabel("Depth (m)", fontsize=10)
    plt.ylabel("Stones", fontsize=10)
    plt.xticks(np.arange(0, 11, 1))
    plt.tight_layout()
    plt.savefig("bhole depth vs stones.jpeg", dpi=400)

def clean_feature_name(name): #Renaming the columns in the boreholoe data
    special_cases = {
        'spatial_lag_y':    'Spatial lag (m)',
        'dist_to_high_grade':   'Distance to high grade (m)',
        'local_grade_pct':      'Local grade %',
        'SedT_Act':         'Actual sediment thickness (m)',
        'Tot_Pen':          'Total penetration (m)',
        'FW_Pen':           'Penetration into footwall (m)',
        'FW_Reached_Y':     'Footwall reached (Yes)',
        'G_Thickns':        'Gravel thickness (m)',
        'G_Units':          'Gravel units',
        'distance_m':       'Distance to nearest structure (m)',
        'E_actual':         'Easting (m)',
        'N_actual':         'Northing (m)',
        'Sed_Tot':          'Total sediment thickness (m)',
        'FW_Type_SCH':      'Footwall type: Schist',
        'FW_Type_SAP':      'Footwall type: Saprolite',
        'FW_Type_RBL':      'Footwall type: Rubble',
        'FW_Type_BRC':      'Footwall type: Breccia',
        'FW_Type_CY':       'Footwall type: Clay',
        'FW_Type_S':        'Footwall type: Sand',
        'FW_Type_SST':      'Footwall type: Sandstone',
        'FW_Type_MISCH':    'Footwall type: Misc Hard',
        'FW_Desc_FIRM':     'Footwall description: Firm',
        'FW_Desc_STIFF':    'Footwall description: Stiff',
        'FW_Desc_HARD':     'Footwall description: Hard',
        'Epoch_Tertiary':   'Epoch: Tertiary',
        'Epoch_Pleistocene':'Epoch: Pleistocene',
        'Epoch_Precambrian':'Epoch: Precambrian',
        'Type_Primary':     'Lineament: Primary',
        'Type_Secondary':   'Lineament: Secondary',
        'FW_Type_CY':       'Footwall type: Clay',
        'FW_Desc_FRIABLE':  'Footwall description: Friable',
        'S start':          'Lineament northing start (m)',
        'S end':            'Lineament northing end (m)',
        'E start':          'Lineament easting start (m)',
        'E end':            'Lineament easting end (m)',
        'L':                'Lineament length (m)',
        'O':                'Lineament orientation (degrees)',
    }
    if name in special_cases:
        return special_cases[name]
    label = name
    label = label.replace('_', ' ')
    return label

def models(): #Building the machine learning algorithms
    faults = DF_PRIM[["Fault start E", "Fault start S", "Fault end E", "Fault end S", "Length of fault (m)", "Orientation of fault (degrees)", "Sector"]].copy()
    faults.columns = ["E start", "S start", "E end", "S end", "L", "O", "Sector"]
    faults["Type"] = "Fault"
    secondary = DF_SEC[["Start E", "Start S", "End E", "End S", "Length(m)", "Orientation (degrees)", "Sector"]].copy()
    secondary.columns = ["E start", "S start", "E end", "S end", "L", "O", "Sector"]
    secondary["Type"] = "Secondary"
    primary = DF_PRIM[["Start E", "Start S", "End E", "End S", "Length(m)", "Orientation (degrees)", "Sector"]].copy()
    primary.columns = ["E start", "S start", "E end", "S end", "L", "O", "Sector"]
    primary["Type"] = "Primary"
    structures_clean = pd.concat([faults, secondary, primary], ignore_index=True)
    structures_clean["Structure ID"] = structures_clean.index
    #Combined geodataframe of borehole and lineament data
    bh_gdf = gpd.GeoDataFrame(
        bh,
        geometry=gpd.points_from_xy(bh["E_actual"], bh["N_actual"]),
        crs="EPSG:32733"
    )

    structures_gdf = gpd.GeoDataFrame(
        structures_clean,
        geometry=structures_clean.apply(
            lambda row: LineString([
                (row["E start"], row["S start"]), 
                (row["E end"], row["S end"])
                ]),
            axis=1
        ),
        crs="EPSG:32733"
    )

    result = gpd.sjoin_nearest(
        bh_gdf,
        structures_gdf,
        how="left",
        distance_col="distance_m",
    )

    #excel_name = "Combined bh and structural data NEW2.xlsx"
    #result.to_excel(excel_name, index=False, sheet_name="Mapped area NEW") #Produces .xlsx file of combined geodataframe
    result = result.dropna(subset=["Structure ID"])
    result = result.drop(columns=['Act_Carats', 'Est_Carats', 'Minutes', 'Start_Time', 'End_Time', 'Dec_Hours', 'Colour', 'Clarity', 'Stone_colour_full', 'FW_Type_full']) #Removed to prevent data leakage
    result = result.dropna(subset=["Stones", "Tot_Pen", "FW_Pen", "SedT_Act", "Sed_Tot", "FW_Reached", "FW_Type", "FW_Desc", "G_Units", "G_Thickns", "Potential"]) #Removes incompelete/missing data
    #print(result[["Sample_ID", "Stones", "Epoch", "L", "O", "Sector", "Tot_Pen", "FW_Pen", "SedT_Act", "Sed_Tot", "FW_Reached", "FW_Type", "FW_Desc", "G_Units", "G_Thickns", "Potential"]].sort_values(by="Stones"))
    result["Epoch"] = result["Epoch"].replace({"PLeistocene": "Pleistocene", "PLEistocene": "Pleistocene"})
    result["FW_Desc"] = result["FW_Desc"].replace({"Friable": "FRIABLE"})

    target = "Stones"
    print(f"Total boreholes: {len(result)}")
    print(f"Zero recovery bh: {(result[target] == 0).sum()}")
    print(f"Zero % : {(result[target] == 0 ).mean() * 100:.1f} %")

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    result[target].hist(bins=50, ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title("Raw distribution", fontsize=11)
    axes[0].set_xlabel("Stone count", fontsize=11)
    axes[0].set_ylabel("Number of boreholes", fontsize=11)
    axes[0].grid(False)
    axes[0].tick_params(axis='both', labelsize=11)
    axes[0].set_xlim(0, 26)
    axes[0].set_xticks(np.arange(0, 26, 10))
    np.log1p(result[target]).hist(bins=50, ax=axes[1], color='steelblue', edgecolor='black')
    axes[1].set_title("Log1p transformed distribution", fontsize=11)
    axes[1].set_xlabel("log(1 + stone count)", fontsize=11)
    axes[1].set_ylabel("Number of boreholes", fontsize=11)
    axes[1].tick_params(axis='both', labelsize=11)
    axes[1].grid(False)
    axes[1].set_xlim(0, 5)
    axes[1].set_xticks(np.arange(0, 5, 1))
    fig.savefig("raw and log distribution of diamonds per borehole.png", dpi=400)
    plt.tight_layout()
    feature_cols = [col for col in result.columns if col not in [target, 'geometry', 'Sample_ID', 'WD_act', 'OS_Comp', 'PF_Comp', 'Potential', 'Comments', 'Status', 'Group_wt', 'Est_Size', 'Act_Size', 'Est_Grade', 'Act_Grade', 'Est_Brkdwn', 'Act_Brkdwn', 'Shape', 'Colour', 'Clarity', 'BrokenStones', 'PercBroken', 'index_right', 'Structure ID']]
    numeric_cols = result[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = result[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    results_encoded = pd.get_dummies(result, columns=cat_cols, drop_first=True)
    feature_cols_encoded = [col for col in results_encoded.columns if col not in [target, 'geometry', 'Sample_ID', 'WD_act', 'OS_Comp', 'PF_Comp', 'Potential', 'Comments', 'Status', 'Group_wt', 'Est_Size', 'Act_Size', 'Est_Grade', 'Act_Grade', 'Est_Brkdwn', 'Act_Brkdwn', 'Shape', 'Colour', 'Clarity', 'BrokenStones', 'PercBroken', 'index_right', 'Structure ID']]
    #drill_time = results_encoded['Minutes'].values #Only include when determining drill time data (Otherwise remove to prevent data leakage)
    #total_carats = results_encoded['Act_Carats'].values #Only include when determining drill time data (Otherwise remove to prevent data leakage)
    X = results_encoded[feature_cols_encoded].copy()
    y = results_encoded[target].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y = y.reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    y_binary = (y > 0).astype(int)
    y_binary = y_binary.reset_index(drop=True)
    y_nonzero = y[y > 0]
    X_nonzero = X_imputed[y > 0]
    print(f"features: {X.shape[1]}, Samples: {X.shape[0]}")
    print("Parameters:\n")
    print(X.columns.tolist())
    results_encoded['E_actual'] = results_encoded.geometry.x
    results_encoded['N_actual'] = results_encoded.geometry.y
    X_imputed['E_actual'] = results_encoded['E_actual'].values
    X_imputed['N_actual'] = results_encoded['N_actual'].values
    coords = list(zip(results_encoded['E_actual'], results_encoded['N_actual']))
    w = KNN.from_array(np.array(coords), k=5)
    w.transform = 'R'

    moran = Moran(y.values, w) #Moran's Index
    print(f"Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")
    y_lag = lag_spatial(w, y.values)
    X_imputed['spatial_lag_y'] = y_lag
    coords_full = np.array(list(zip(
        results_encoded['E_actual'].values,
        results_encoded['N_actual'].values
    )))
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    y_train_bin = (y_train > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)
    print(f"Train size: {len(X_train)}, test size: {len(X_test)}")
    #Local variance (nugget effect indicator)
    w_var   = KNN.from_array(coords_full, k=10)
    y_arr   = y.values
    local_var = np.zeros(len(y_arr))
    w_pct      = KNN.from_array(coords_full, k=20)
    local_pct  = np.zeros(len(y_arr))
    for i, neighbours in w_pct.neighbors.items():
        neighbour_vals = y_arr[list(neighbours)]
        local_pct[i]   = (y_arr[i] >= neighbour_vals).mean()
    X_imputed['local_grade_pct'] = local_pct
    print(f"Added local_grade_pct: "
        f"mean={local_pct.mean():.3f}")

    
    #Distance to nearest high-grade
    high_grade_threshold = y.quantile(0.90)
    hg_mask              = y.values >= high_grade_threshold
    high_grade_coords    = coords_full[hg_mask]
    print(f"High-grade threshold (90th pct): "
        f"{high_grade_threshold:.2f} stones")
    print(f"Number of high-grade boreholes: "
        f"{hg_mask.sum()}")
    nn_hg = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn_hg.fit(high_grade_coords)
    dist_hg, _ = nn_hg.kneighbors(coords_full)
    X_imputed['dist_to_high_grade'] = dist_hg.flatten()
    print(f"Added dist_to_high_grade: "
        f"mean={dist_hg.mean():.1f}m")
    print(f"\nFeature matrix after spatial engineering: "
        f"{X_imputed.shape}")
    print(f"New spatial features added: 3")
    print(f" dist_to_high_grade, local_grade_perc, spatal lag")

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    print("HYPERPARAMETER TUNING — SPATIAL CV")
    print("="*60)
    print("This may take a few minutes...")

    results_encoded['block_x'] = pd.cut(results_encoded['E_actual'], bins=5, labels=False)
    results_encoded['block_y'] = pd.cut(results_encoded['N_actual'], bins=5, labels=False)
    results_encoded['block_id'] = (results_encoded['block_x'].astype(str) + '_' +results_encoded['block_y'].astype(str))
    tune_groups  = results_encoded['block_id'].values
    spatial_cv_tuner = GroupKFold(n_splits=5)

    #Random Forest parameter search
    rf_param_dist = {
        'n_estimators':      [200, 300, 500, 750],
        'max_features':      ['sqrt', 'log2', 0.3, 0.4],
        'max_depth':         [None, 10, 15, 20],
        'min_samples_leaf':  [1, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'max_samples':       [0.6, 0.7, 0.8, None]
    }

    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(
            random_state=42, n_jobs=-1
        ),
        param_distributions=rf_param_dist,
        n_iter=30,
        cv=spatial_cv_tuner,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rf_search.fit(X_imputed,np.log1p(y), groups=tune_groups)

    best_rf_params = rf_search.best_params_
    best_rf_cv_rmse = -rf_search.best_score_
    print(f"\nBest RF parameters: {best_rf_params}")
    print(f"Best RF spatial CV RMSE: {best_rf_cv_rmse:.4f}")

    #XGBoost parameter search
    xgb_param_dist = {
        'n_estimators':     [300, 500, 750],
        'learning_rate':    stats.loguniform(0.01, 0.2),
        'max_depth':        [3, 4, 5, 6, 7],
        'subsample':        stats.uniform(0.6, 0.4),
        'colsample_bytree': stats.uniform(0.5, 0.5),
        'reg_alpha':        stats.loguniform(0.001, 1.0),
        'reg_lambda':       stats.loguniform(0.1, 10.0),
        'min_child_weight': [1, 3, 5, 7],
        'gamma':            stats.uniform(0, 0.3)
    }

    xgb_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'),
        param_distributions=xgb_param_dist,
        n_iter=30,
        cv=spatial_cv_tuner,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    xgb_search.fit(X_imputed, np.log1p(y), groups=tune_groups)
    best_xgb_params = xgb_search.best_params_
    best_xgb_cv_rmse = -xgb_search.best_score_
    print(f"\nBest XGBoost parameters: {best_xgb_params}")
    print(f"Best XGBoost spatial CV RMSE: {best_xgb_cv_rmse:.4f}")
    print(f"\nRF improvement from tuning: {3.892 - best_rf_cv_rmse:.4f} RMSE reduction")
    print(f"XGB improvement from tuning: {3.828 - best_xgb_cv_rmse:.4f} RMSE reduction")

    k_values = [3, 5, 10, 15, 20, 30]
    knn_results = []
    train_rmses = []
    test_rmses = []
    train_r2s = []
    test_r2s = []
    predictions_by_k = {}
    for k in k_values:
        knn = Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNR(
                n_neighbors=k,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            ))
        ])
        knn.fit(X_train, np.log1p(y_train))
        k_train_preds = np.expm1(knn.predict(X_train))
        k_train_rmse = np.sqrt(mean_squared_error(y_train, k_train_preds))
        k_train_r2 = r2_score(y_train, k_train_preds)
        k_test_preds = np.expm1(knn.predict(X_test))
        k_test_rmse = np.sqrt(mean_squared_error(y_test, k_test_preds))
        k_test_r2 = r2_score(y_test, k_test_preds)
        train_rmses.append(k_train_rmse)
        test_rmses.append(k_test_rmse)
        train_r2s.append(k_train_r2)
        test_r2s.append(k_test_r2)
        predictions_by_k[k] = k_test_preds
        knn_results.append({
            'k': k,
            'RMSE': k_test_rmse,
            'R2': k_test_r2,
            'train_rmse': k_train_rmse,
            'train_r2': k_train_r2
        })
        print(f"k = {k}\n Train RMSE: {k_train_rmse:.4f} R2: {k_train_r2:.4f} | "
              f"Test RMSE: {k_test_rmse:.4f} r2: {k_test_r2:4f}")

    knn_df = pd.DataFrame(knn_results).sort_values('RMSE')
    best_k_idx = np.argmin(test_rmses)
    best_k = k_values[best_k_idx]
    best_rmse = test_rmses[best_k_idx]
    best_r2 = test_r2s[best_k_idx]
    print(f"\nBest K: {best_k} | RMSE: {best_rmse:.4f} | r2: {best_r2:.4f}")
    print("Top 5 configs:")
    print(knn_df.head(5).to_string(index=False))
    print(knn_df.iloc[0])

    plot_ks = [
        k_values[0],
        best_k,
        k_values[len(k_values)//2],
        k_values[-1]
    ]
    seen = set()
    plot_ks = [k for k in plot_ks if not (k in seen or seen.add(k))]
    n_plots = len(plot_ks)

    rf = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    rf.fit(X_train, np.log1p(y_train))
    y_pred_log = rf.predict(X_test)
    y_pred_rf = np.expm1(y_pred_log)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"RF log1p: RMSE: {rmse_rf:.4f}, R-squared: {r2_rf:.4f}")
    plt.figure(figsize=(8, 8))
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top20 = importances.sort_values(ascending=False).head(20)
    clean_labels_top20 = [clean_feature_name(name) for name in top20.index]
    top20_asc = top20.sort_values(ascending=True)
    clean_labels_asc = [clean_feature_name(name) for name in top20_asc.index]
    top20_asc.plot(kind='barh', color='steelblue')
    plt.title("Random Forest feature importances", fontsize=12)
    plt.yticks(range(len(top20_asc)), clean_labels_asc, fontsize=12)
    plt.xlabel("Feature importance", fontsize=12)
    plt.rcParams['xtick.labelsize'] = 12
    plt.ylabel("Parameter", fontsize=12)
    plt.tight_layout()
    plt.savefig("feature importance Nov.png", dpi=400)


    geological_features = ['Tot_Pen', 'FW_Pen', 'SedT_Act', 'Sed_Tot', 'G_Units', 'G_Thickns', 'L', 'O']
    geo_encoded = [c for c in X_imputed.columns if any(x in c for x in ['FW_Reached', 'FW_Type', 'FW_Desc', 'Epoch'])]
    geological_features.extend(geo_encoded)
    X_cluster = X_imputed[geological_features].copy()
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cluster_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_cluster_scaled, labels, sample_size=2000)
        silhouette_scores.append(sil)
        print(f"k={k}: inertia={km.inertia_:.0f}, "
              f"silhouette={sil:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2)
    plt.xlabel("Number of clusters ($k$)", fontsize=12)
    plt.ylabel("Silhouette score", fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("K-means silhouette score.png", dpi=400)
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"\nOptimal k by silhouette: {optimal_k}")

    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_cluster_scaled)
    results_encoded['geological_cluster'] = cluster_labels
    X_imputed['geological_cluster'] = cluster_labels
    cluster_stats = results_encoded.groupby('geological_cluster')['Stones'].agg([
        'count', 'mean', 'median', 'std',
        lambda x: (x == 0).mean() * 100  # zero percentage
    ]).round(3)
    cluster_stats.columns = ['Count', 'Mean Stones', 'Median Stones', 'Std Dev', 'Zero %']
    cluster_stats = cluster_stats.sort_values('Mean Stones', ascending=False)
    cluster_gdf = gpd.GeoDataFrame(
        {'cluster': cluster_labels,
        'stones': y.values},
        geometry=gpd.points_from_xy(
            results_encoded['E_actual'],
            results_encoded['N_actual']
        ),
        crs='EPSG:32733'
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    cluster_gdf.plot(
        column='cluster',
        cmap='tab10',
        legend=True,
        markersize=5,
        ax=axes[0],
        categorical=True
    )
    axes[0].set_title("Spatial distribution of clusters", fontsize=11)
    axes[0].set_xlabel("Easting (m)", fontsize=10)
    axes[0].set_ylabel("Northing (m)", fontsize=10)
    axes[0].tick_params(axis='both', labelsize=10)
    axes[0].tick_params(axis='x', labelrotation=45)
    axes[0].grid(False)

    cluster_gdf.plot(
        column='stones',
        cmap='plasma',
        legend=True,
        markersize=5,
        ax=axes[1],
        vmin=0,
        vmax=y.quantile(0.95)
    )
    axes[1].set_title("Stone Recovery Distribution", fontsize=11)
    axes[1].set_xlabel("Easting (m)", fontsize=10)
    axes[1].set_ylabel("Northing (m)", fontsize=10)
    axes[1].tick_params(axis='both', labelsize=10)
    axes[1].grid(False)
    axes[1].tick_params(axis='x', labelrotation=45)
    plt.suptitle("K-Means clustering results", fontsize=10)
    plt.savefig('K-means maps Nov.png', dpi=400)
    plt.tight_layout()

    cluster_model_results = {}
    for cluster_id in sorted(results_encoded['geological_cluster'].unique()):
        cluster_mask = X_imputed['geological_cluster'] == cluster_id
        X_cl = X_imputed[cluster_mask].drop(columns=['geological_cluster'])
        y_cl = y[cluster_mask]
        n_cluster = len(X_cl)
        n_nonzero = (y_cl > 0).sum()
        print(f"\nCluster {cluster_id}: {n_cluster} boreholes, "
            f"{n_nonzero} non-zero ({n_nonzero/n_cluster*100:.1f}%)")
        # Skip if cluster is too small to model reliably
        if n_cluster < 100:
            print(f"  Skipping — too few boreholes for reliable model")
            continue

        # Split within cluster
        X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(
            X_cl, y_cl, test_size=0.2, random_state=42
        )

        rf_cl = RandomForestRegressor(
            n_estimators=200, max_features='sqrt',
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        rf_cl.fit(X_cl_train, np.log1p(y_cl_train))
        preds_cl = np.expm1(rf_cl.predict(X_cl_test))

        rmse_cl = np.sqrt(mean_squared_error(y_cl_test, preds_cl))
        r2_cl = r2_score(y_cl_test, preds_cl)
        cluster_model_results[cluster_id] = {
            'n': n_cluster,
            'RMSE': rmse_cl,
            'R2': r2_cl,
            'mean_stones': y_cl.mean()
        }
        print(f"  Cluster model — RMSE: {rmse_cl:.4f}, R²: {r2_cl:.4f}")
        print(f"  Global model comparison — RMSE: {rmse_rf:.4f}, "
            f"R²: {r2_rf:.4f}")
        print(f"  Improvement: "
            f"{'YES' if rmse_cl < rmse_rf else 'NO'}")
    print("\n=== CLUSTER MODEL SUMMARY ===")
    print(pd.DataFrame(cluster_model_results).T.to_string())

    X_profile = X_imputed.copy()
    X_profile['stones']        = y.values
    X_profile['stones_binary'] = (y.values > 0).astype(int)
    cluster_feature_means = X_profile.groupby(
        'geological_cluster'
    ).mean().round(3)

    profile_feature_cols = [c for c in X_profile.columns
                            if c not in ['geological_cluster',
                                        'stones',
                                        'stones_binary']]
    profile_overall_mean = X_profile[profile_feature_cols].mean()
    profile_overall_std  = X_profile[profile_feature_cols].std()
    cluster_zscore = (
        cluster_feature_means[profile_feature_cols]
        - profile_overall_mean
    ) / profile_overall_std

    n_clusters     = len(cluster_feature_means)
    profile_colours = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    print("\n" + "=" * 65)
    print("K-MEANS GEOLOGICAL CLUSTER INTERPRETATION SUMMARY")
    print("=" * 65)

    for cluster_id in cluster_stats.index:
        n_bh   = cluster_stats.loc[cluster_id, 'Count']
        mean_s = cluster_stats.loc[cluster_id, 'Mean Stones']
        zero_p = cluster_stats.loc[cluster_id, 'Zero %']
        max_s  = cluster_stats.loc[cluster_id, 'Std Dev']

        if mean_s >= cluster_stats['Mean Stones'].quantile(0.75):
            grade_class = "HIGH GRADE"
        elif mean_s >= cluster_stats['Mean Stones'].quantile(0.25):
            grade_class = "MODERATE GRADE"
        else:
            grade_class = "LOW GRADE / BARREN"

        top3 = cluster_zscore.loc[cluster_id].abs().sort_values(
            ascending=False
        ).head(3)

        print(f"\nCluster {cluster_id} — {grade_class}")
        print(f"  Boreholes:   {n_bh:.0f}")
        print(f"  Mean stones: {mean_s:.2f}")
        print(f"  Zero %:      {zero_p:.1f}%")
        print(f"  Defined by:")

        for feat, z_abs in top3.items():
            z_val     = cluster_zscore.loc[cluster_id, feat]
            direction = 'HIGH' if z_val > 0 else 'LOW'
            print(f"    - {clean_feature_name(feat):<35} "
                f"{direction} (z = {z_val:+.2f})")

    scale_pos = (y_train == 0).sum() / (y_train > 0).sum()

    xgb_model = xgb.XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1, eval_metric='rmse')
    xgb_model.fit(X_train, np.log1p(y_train), eval_set=[(X_test, np.log1p(y_test))], verbose=50)
    y_pred_xgb = np.expm1(xgb_model.predict(X_test))
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"XGBoost RMSE: {rmse_xgb:.4f}, r2: {r2_xgb:.4f}")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type='bar', color='steelblue', show=False)
    ax = plt.gca()
    current_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    clean_labels_shap = [clean_feature_name(label) for label in current_labels]
    ax.set_yticklabels(clean_labels_shap, fontsize=13)
    ax.set_title("SHAP feature importance on diamond distribution", fontsize=13)
    ax.set_xlabel("SHAP value (Stones)", fontsize=13)
    ax.set_ylabel("Parameter", fontsize=13)
    ax.tick_params(axis='both', colors='black', labelsize=13)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
    plt.tight_layout()
    plt.savefig("Shap plot Nov.png", dpi=400)


    top9 = importances.sort_values(ascending=False).head(9).index.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), constrained_layout=True)
    axes = axes.flatten()
    for i, feature in enumerate(top9):
        shap.dependence_plot(
            feature,
            shap_values,
            X_test,
            ax=axes[i],
            show=False
        )
        axes[i].set_title(clean_feature_name(feature), fontsize=10, pad=3)
        axes[i].set_xlabel(clean_feature_name(feature), fontsize=10, labelpad=3)
        axes[i].set_ylabel("SHAP value", fontsize=10, labelpad=3)
        axes[i].tick_params(axis='both', labelsize=10)
        axes[i].grid(False)
    main_axes_set = set(id(ax) for ax in axes)
    for ax_extra in fig.axes:
        if id(ax_extra) not in main_axes_set:
            ax_extra.tick_params(labelsize=10)
            raw_label = ax_extra.get_ylabel()
            if raw_label and raw_label.strip():
                ax_extra.set_ylabel(
                    clean_feature_name(raw_label.strip()), fontsize=10, labelpad=2
                )

    #Shap coefficient table
    mean_shap = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0)  # positive = increases concentration
    }).sort_values('mean_abs_shap', ascending=False)
    print("\n=== SHAP FEATURE IMPORTANCE TABLE ===")
    print("(Equivalent to statistical model coefficient table)")
    print(f"{'Feature':<30} {'Mean |SHAP|':>12} {'Direction':>12}")
    print("-" * 56)
    for _, row in mean_shap.head(20).iterrows():
        direction = "increases conc." if row['mean_shap'] > 0 else "decreases conc."
        print(f"{row['feature']:<30} {row['mean_abs_shap']:>12.4f} {direction:>16}")

    clf_stage1 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
    clf_stage1.fit(X_train, y_train_bin)
    p_presense = clf_stage1.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_bin, p_presense)
    print(f"Stage 1 AUC: {auc:.4f}")
    fpr, tpr, thresholds = roc_curve(y_test_bin, p_presense)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Classifier (AUC = 0.500)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("ROC_Hurdle.png", dpi=400)


    clf_stage1_cal = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        method='isotonic',
        cv=5
    )

    clf_stage1_cal.fit(X_train, y_train_bin)
    p_presense = clf_stage1_cal.predict_proba(X_test)[:, 1] # p_presense = calibrated probabilities
    print(f"\np_presense is now calibrated.")
    print(f"All Hurdle predictions will use calibrated Stage 1 probabilities.")

    mask_train = y_train > 0
    X_train_nz = X_train[mask_train]
    y_train_nz = np.log1p(y_train[mask_train])
    reg_stage2 = GradientBoostingRegressor(n_estimators=300, random_state=42)
    reg_stage2.fit(X_train_nz, y_train_nz)
    abundance_pred = np.expm1(reg_stage2.predict(X_test))
    y_pred_hurdle = p_presense * abundance_pred
    rmse_hurdle = np.sqrt(mean_squared_error(y_test, y_pred_hurdle))
    r2_hurdle = r2_score(y_test, y_pred_hurdle)
    print(f"Hurdle RMSE:{rmse_hurdle:.4f}, R2: {r2_hurdle:.4f}")

    scaler_stats = StandardScaler()
    X_train_scaled = scaler_stats.fit_transform(X_train)
    X_test_scaled = scaler_stats.transform(X_test)
    X_train_sm = sm.add_constant(X_train_scaled)
    X_test_sm = sm.add_constant(X_test_scaled)
    importances = pd.Series(
        rf.feature_importances_, 
        index=X_train.columns
    )
    top_features = importances.sort_values(ascending=False).head(15).index.tolist()
    print(f"Top 15 features for statistical model:\n{top_features}")
    X_train_sm_top = sm.add_constant(X_train_scaled[:, [X_train.columns.get_loc(f) for f in top_features]])
    X_test_sm_top = sm.add_constant(X_test_scaled[:, [X_train.columns.get_loc(f) for f in top_features]])
    y_train_counts = y_train.round().astype(int)
    y_test_counts = y_test.round().astype(int)

    print(f"Target summary:")
    print(f"  Min: {y_train_counts.min()}")
    print(f"  Max: {y_train_counts.max()}")  
    print(f"  Mean: {y_train_counts.mean():.2f}")
    print(f"  Variance: {y_train_counts.var():.2f}")
    print(f"  Overdispersion ratio (var/mean): {y_train_counts.var()/y_train_counts.mean():.2f}")
    # If overdispersion ratio >> 1, ZINB is appropriate
    # If max value is very large (>100), consider log-scaling or Tweedie instead

    # If max stones per borehole is very large, cap or transform
    if y_train_counts.max() > 100:
        print(f"\nWARNING: Max count of {y_train_counts.max()} is very large")
        print("Consider using Tweedie regression instead of ZIP/ZINB")

    try:
        zip_model = ZeroInflatedPoisson(y_train_counts, X_train_sm_top, inflation='logit')
        zip_result = zip_model.fit(method='nm', maxiter=2000, disp=True, skip_hessian=False)
        print(zip_result.summary())
    except Exception as e:
        print(f"ZIP failed: {e}")

    try:
        zinb_model = ZeroInflatedNegativeBinomialP(y_train_counts, X_train_sm_top, inflation='logit')
        zinb_result = zinb_model.fit(method='nm', maxiter=2000, disp=True)
        print(zinb_result.summary())

        X_test_sm = sm.add_constant(X_test.values)
        y_pred_zinb = zinb_result.predict(X_test_sm)
        rmse_zinb = np.sqrt(mean_squared_error(y_test, y_pred_zinb))
        print(f"ZINB RMSE: {rmse_zinb:.4f}")
    except Exception as e:
        print(f"ZINB failed: {e}")

    tweedie = Pipeline([('scaler', StandardScaler()), ('model', TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000))])
    tweedie.fit(X_train, y_train)
    y_pred_tweedie = tweedie.predict(X_test)
    rmse_tw = np.sqrt(mean_squared_error(y_test, y_pred_tweedie))
    r2_tw = r2_score(y_test, y_pred_tweedie)
    print(f"Tweedie RMSE: {rmse_tw:.4f}, r2; {r2_tw:.4f}")
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': tweedie.named_steps['model'].coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    print(coef_df.head(20))

    n_gwr_features = 5
    top_gwr_features = importances.sort_values(ascending=False).head(n_gwr_features).index.tolist()
    print(f"Fts selected for GWR: {top_gwr_features}")
    coords_array = np.array(list(zip(results_encoded['E_actual'], results_encoded['N_actual'])))
    coords_train = coords_array[X_train.index]
    coords_test = coords_array[X_test.index]
    scaler_gwr = StandardScaler()
    X_train_gwr = scaler_gwr.fit_transform(X_train[top_gwr_features])
    X_test_gwr = scaler_gwr.transform(X_test[top_gwr_features])
    y_train_gwr = np.log1p(y_train.values).reshape(-1, 1)
    y_test_gwr = np.log1p(y_test.values).reshape(-1, 1)
    print(f"GWR training array shape: {X_train_gwr.shape}")
    print(f"Coords shape: {coords_train.shape}")
    print(f"Samples per ft ratio: {X_train_gwr.shape[0] / X_train_gwr.shape[1]:.0f}:1")

    try:
        bw_selector = Sel_BW(coords_train, y_train_gwr, X_train_gwr, fixed=False, kernel='bisquare', spherical=False)
        bw = bw_selector.search(criterion='AICc', verbose=True)
        print(f"Optimal GWR bandwidth: {bw} neighbours")
        if bw < n_gwr_features * 3:
            print(f"WARNING: bw {bw} is very small relative to "
                  f"{n_gwr_features} fts - singular matrices likely")
            print("Force a minimum bw:")
            bw = max(bw, n_gwr_features * 10)
            print(f"BW forced to: {bw}")
    except np.linalg.LinAlgError as e:
        print(f"BW search failed: {e}")
        print("Falling back to fixed bw of 100 neighbours")
        bw = 100
    
    print(f"Fitting gwr with bw = {bw} ...")
    try:
        gwr_model = GWR(coords_train, y_train_gwr, X_train_gwr, bw=bw, fixed=False, kernel='bisquare', spherical=False)
        gwr_results = gwr_model.fit()
        print(gwr_results.summary())
        gwr_success = True
    except np.linalg.LinAlgError as e:
        print(f"GWR fit failed: {e}")
        print("try increasing n_gwr_features reduction or BW")
        gwr_success = False
        top3 = importances.sort_values(ascending=False).head(3).index.tolist()
        X_train_gwr3 = scaler_gwr.fit_transform(X_train[top3])
        try:
            bw3_selector = Sel_BW(coords_train, y_train_gwr, X_train_gwr3, fixed=False, kernel='bisquare', spherical=False)
            bw3 = bw3_selector.search(criterion='AICc', verbose=True)
            gwr_model = GWR(coords_train, y_train_gwr, X_train_gwr3, bw=bw3, fixed=False, kernel='bisquare', spherical=False)
            gwr_results = gwr_model.fit()
            print(gwr_results.summary())
            top_gwr_features = top3
            X_train_gwr = X_train_gwr3
            gwr_success = True
            print(f"GWR succeeded with 3 ft fallback")
        except Exception as e2:
            print(f"3 ft fallback also failed: {e2}")
            print("Skipping GWR - proceed to spatial residual analysis")
    if gwr_success:
        train_geometry = results_encoded.iloc[X_train.index].geometry.values
        coef_col_names = ['coef_intercept'] + [
            f'coef_{f}' for f in top_gwr_features
        ]
        gwr_result_gdf = gpd.GeoDataFrame(
            {
                'local_R2': gwr_results.localR2.flatten(),
                'actual': y_train.values,
                'predicted': np.expm1(gwr_results.predy.flatten()),
                'residual': (y_train.values - np.expm1(gwr_results.predy.flatten())),
            },
            geometry=train_geometry,
            crs=results_encoded.crs
        )

        for i, col_name in enumerate(coef_col_names):
            gwr_result_gdf[col_name] = gwr_results.params[:, i]
        print(f"GWR results GDF: {gwr_result_gdf.shape}")
        print(f"local r2 - min: {gwr_result_gdf['local_R2'].min():.3f}, "
                f"Mean: {gwr_result_gdf['local_R2'].mean():.3f}, "
                f"Max: {gwr_result_gdf['local_R2'].max():.3f}")
        good_fit = (gwr_result_gdf['local_R2'] > 0.3).sum()
        poor_fit = (gwr_result_gdf['local_R2'] < 0.1).sum()
        print(f"Bhs with local r2 > 0.3 (good fit): {good_fit} "
              f"({good_fit/len(gwr_result_gdf)*100:.1f}%)")
        print(f"Bhs with local r2 < 0.1 (poor fit): {poor_fit} "
              f"({poor_fit/len(gwr_result_gdf)*100:.1f}%)")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        gwr_result_gdf.plot(column='local_R2', cmap='RdYlGn', legend=True, markersize=6, ax=axes[0], vmin=0, vmax=1,
                            legend_kwds={
                                'shrink': 0.5,
                                'pad': 0.02,
                                'aspect': 20
                            }
        )
        axes[0].set_title("GWR local R\u00b2", fontsize=11)
        axes[0].set_xlabel('Easting (m)', fontsize=10)
        axes[0].set_ylabel('Northing (m)', fontsize=10)
        axes[0].tick_params(axis='x', labelsize=10)
        axes[0].tick_params(axis='y', labelsize=10)
        axes[0].tick_params(axis='x', labelrotation=45)
        gwr_result_gdf.plot(column='residual', cmap='RdBu', legend=True, markersize=6, ax=axes[1], vmin=-gwr_result_gdf['residual'].std() * 2, vmax=gwr_result_gdf['residual'].std() * 2,
                            legend_kwds={
                                'shrink': 0.5,
                                'pad': 0.02,
                                'aspect': 20
                            }
        )
        axes[1].set_title("GWR residual", fontsize=11)
        axes[1].set_xlabel('Easting (m)', fontsize=10)
        axes[1].set_ylabel('Northing (m)', fontsize=10)
        axes[1].tick_params(axis='x', labelsize=10)
        axes[1].tick_params(axis='y', labelsize=10)
        axes[1].tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        plt.savefig("GWR Maps Nov.png", dpi=400)

    print("\n=== SPATIAL NON-STATIONARITY SUMMARY ===")
    print(f"{'Feature':<25} {'Global coef':>12} {'Local mean':>12} "
          f"{'Local STD':>12} {'Min':>10} {'Max':>10} {'Non-stationary?':>16}")
    print("-" * 100)

    gwr_train_pred    = np.expm1(gwr_results.predy.flatten())
    gwr_train_actual  = y_train.values
    gwr_train_rmse    = np.sqrt(mean_squared_error(
        gwr_train_actual, gwr_train_pred
    ))
    gwr_global_r2     = gwr_results.R2
    gwr_adj_r2        = gwr_results.adj_R2 if hasattr(
        gwr_results, 'adj_R2'
    ) else gwr_results.R2
    print(f"\n=== GWR PERFORMANCE METRICS ===")
    print(f"GWR Global R²:          {gwr_global_r2:.4f}")
    print(f"GWR Adjusted R²:        {gwr_adj_r2:.4f}")
    print(f"GWR Training RMSE:      {gwr_train_rmse:.4f}")
    print(f"GWR Mean Local R²:      "
          f"{gwr_result_gdf['local_R2'].mean():.4f}")
    print(f"GWR Min Local R²:       "
          f"{gwr_result_gdf['local_R2'].min():.4f}")
    print(f"GWR Max Local R²:       "
          f"{gwr_result_gdf['local_R2'].max():.4f}")
    print(f"NOTE: GWR metrics are in-sample (training data only).")
    print(f"Direct comparison with test-set ML metrics is not valid.")
    try:
        gwr_test_results = gwr_model.predict(
            coords_test,
            X_test_gwr
        )
        gwr_test_pred   = np.expm1(
            gwr_test_results.predictions.flatten()
        )
        gwr_test_rmse   = np.sqrt(mean_squared_error(
            y_test.values, gwr_test_pred
        ))
        gwr_test_r2     = r2_score(y_test.values, gwr_test_pred)
        print(f"\nGWR Approximate Test RMSE:  {gwr_test_rmse:.4f}")
        print(f"GWR Approximate Test R²:    {gwr_test_r2:.4f}")
        print(f"NOTE: GWR test prediction is interpolative and uses")
        print(f"only {len(top_gwr_features)} features vs "
              f"{X_train.shape[1]} for ML models.")
        gwr_comparable = True
    except Exception as e:
        print(f"GWR test prediction failed: {e}")
        print("Using training metrics only for GWR.")
        gwr_test_rmse = gwr_train_rmse
        gwr_test_r2   = gwr_global_r2
        gwr_comparable = False
    for i, feature in enumerate(top_gwr_features):
        col_name   = f'coef_{feature}'
        local_mean = gwr_result_gdf[col_name].mean()
        local_std  = gwr_result_gdf[col_name].std()
        local_min  = gwr_result_gdf[col_name].min()
        local_max  = gwr_result_gdf[col_name].max()
        # A feature is non-stationary if its local STD is large
        # relative to its mean, or if it changes sign across the study area
        changes_sign = (local_min < 0) and (local_max > 0)
        high_variance = local_std > abs(local_mean)
        non_stationary = changes_sign or high_variance
        flag = "YES — spatially varying" if non_stationary else "stable globally"
        print(f"{feature:<25} {'N/A':>12} {local_mean:>12.3f} "
              f"{local_std:>12.3f} {local_min:>10.3f} "
              f"{local_max:>10.3f} {flag:>16}")

    results_encoded['block_x'] = pd.cut(results_encoded['E_actual'], bins=5, labels=False)
    results_encoded['block_y'] = pd.cut(results_encoded['N_actual'], bins=5, labels=False)
    results_encoded['block_id'] = results_encoded['block_x'].astype(str) + '_' + results_encoded['block_y'].astype(str)

    gkf = GroupKFold(n_splits=5)
    groups = results_encoded['block_id'].values
    fold_assignments = np.zeros(len(X_imputed), dtype=int)
    cv_rf_scores = []
    cv_xgb_scores = []
    cv_hurdle_scores = []
    cv_knn_scores = []
    print("\n=== SPATIAL CROSS-VALIDATION — ALL MODELS ===")
    print(f"{'Fold':<8} {'RF':>10} {'XGBoost':>10} "
        f"{'Hurdle':>10} {'KNN':>10}")
    print("-" * 50)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X_imputed, y, groups)
    ):
        fold_assignments[test_idx] = fold + 1
        X_cv_train = X_imputed.iloc[train_idx]
        X_cv_test  = X_imputed.iloc[test_idx]
        y_cv_train = y.iloc[train_idx]
        y_cv_test  = y.iloc[test_idx]
        #Random Forest
        rf_cv = RandomForestRegressor(
            n_estimators=200, max_features='sqrt',
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        rf_cv.fit(X_cv_train, np.log1p(y_cv_train))
        rf_cv_pred  = np.expm1(rf_cv.predict(X_cv_test))
        rf_cv_score = np.sqrt(mean_squared_error(y_cv_test, rf_cv_pred))
        cv_rf_scores.append(rf_cv_score)

        #XGBoost
        xgb_cv = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        xgb_cv.fit(
            X_cv_train, np.log1p(y_cv_train),
            verbose=False
        )
        xgb_cv_pred  = np.expm1(xgb_cv.predict(X_cv_test))
        xgb_cv_score = np.sqrt(mean_squared_error(
            y_cv_test, xgb_cv_pred
        ))
        cv_xgb_scores.append(xgb_cv_score)

        #Hurdle Model
        y_cv_bin = (y_cv_train > 0).astype(int)
        clf_cv = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        clf_cv.fit(X_cv_train, y_cv_bin)
        p_cv = clf_cv.predict_proba(X_cv_test)[:, 1]
        mask_cv     = y_cv_train > 0
        X_cv_nz     = X_cv_train[mask_cv]
        y_cv_nz     = np.log1p(y_cv_train[mask_cv])
        if len(X_cv_nz) > 10:
            reg_cv = GradientBoostingRegressor(
                n_estimators=200, random_state=42
            )
            reg_cv.fit(X_cv_nz, y_cv_nz)
            abund_cv        = np.expm1(reg_cv.predict(X_cv_test))
            hurdle_cv_pred  = p_cv * abund_cv
        else:
            hurdle_cv_pred = p_cv * y_cv_train[mask_cv].mean()
        hurdle_cv_score = np.sqrt(mean_squared_error(
            y_cv_test, hurdle_cv_pred
        ))
        cv_hurdle_scores.append(hurdle_cv_score)

        #KNN
        knn_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNR(
                n_neighbors=best_k,
                weights='distance',
                n_jobs=-1
            ))
        ])
        knn_cv.fit(X_cv_train, np.log1p(y_cv_train))
        knn_cv_pred  = np.expm1(knn_cv.predict(X_cv_test))
        knn_cv_score = np.sqrt(mean_squared_error(
            y_cv_test, knn_cv_pred
        ))
        cv_knn_scores.append(knn_cv_score)

        print(f"Fold {fold+1:<3} "
            f"{rf_cv_score:>10.4f} "
            f"{xgb_cv_score:>10.4f} "
            f"{hurdle_cv_score:>10.4f} "
            f"{knn_cv_score:>10.4f}")
    cv_rf_mean      = np.mean(cv_rf_scores)
    cv_rf_std       = np.std(cv_rf_scores)
    cv_xgb_mean     = np.mean(cv_xgb_scores)
    cv_xgb_std      = np.std(cv_xgb_scores)
    cv_hurdle_mean  = np.mean(cv_hurdle_scores)
    cv_hurdle_std   = np.std(cv_hurdle_scores)
    cv_knn_mean     = np.mean(cv_knn_scores)
    cv_knn_std      = np.std(cv_knn_scores)

    print("-" * 50)
    print(f"{'Mean':<8} "
        f"{cv_rf_mean:>10.4f} "
        f"{cv_xgb_mean:>10.4f} "
        f"{cv_hurdle_mean:>10.4f} "
        f"{cv_knn_mean:>10.4f}")
    print(f"{'Std':<8} "
        f"{cv_rf_std:>10.4f} "
        f"{cv_xgb_std:>10.4f} "
        f"{cv_hurdle_std:>10.4f} "
        f"{cv_knn_std:>10.4f}")
    print(f"\nMean spatial CV RMSE: {cv_rf_mean:.4f} "
        f"+- {cv_rf_std:.4f}")

    results_encoded['cv_fold'] = fold_assignments
    fold_rmse = {
        1: cv_rf_scores[0],
        2: cv_rf_scores[1],
        3: cv_rf_scores[2],
        4: cv_rf_scores[3],
        5: cv_rf_scores[4]
    }
    results_encoded['fold_rmse'] = results_encoded['cv_fold'].map(fold_rmse)
    fold_gdf = gpd.GeoDataFrame(
        results_encoded[['cv_fold', 'fold_rmse', 'Stones']],
        geometry=gpd.points_from_xy(
            results_encoded['E_actual'],
            results_encoded['N_actual']
        ),
        crs='EPSG:32736'
    )
    print(f"fold_gdf created: {fold_gdf.shape}")
    print(f"Fold counts:\n{fold_gdf['cv_fold'].value_counts().sort_index()}")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fold_gdf.plot(column='cv_fold', cmap='tab10', legend=True, markersize=5, ax=axes[0], categorical=True)
    axes[0].set_title("Spatial cross validation", fontsize=11)
    axes[0].set_xlabel("Easting (m)", fontsize=10)
    axes[0].set_ylabel("Northing (m)", fontsize=10)
    axes[0].tick_params(axis='both', labelsize=10)
    axes[0].grid(False)
    axes[0].tick_params(axis='x', labelrotation=45)
    offsets = {1: 0, 2: 2000, 3: -2000, 4: 2000, 5: -2000}
    for fold_id in range(1, 6):
        fold_points = fold_gdf[fold_gdf['cv_fold'] == fold_id]
        if len(fold_points) > 0:
            cx = fold_points.geometry.x.median()
            cy = fold_points.geometry.y.median() + offsets.get(fold_id, 0)
            axes[0].annotate(
                f"Fold {fold_id}\nRMSE={fold_rmse[fold_id]:.2f}",
                xy=(cx, cy),
                fontsize=7,
                fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.2',
                        facecolor='white', alpha=0.8,
                        edgecolor='grey', linewidth=0.5)
            )
    fold_gdf.plot(column='fold_rmse', cmap='RdYlGn_r', legend=True, markersize=5, ax=axes[1])
    axes[1].set_title("Spatical CV predictive performance by location", fontsize=11)
    axes[1].set_xlabel("Easting (m)", fontsize=10)
    axes[1].set_ylabel("Northing (m)", fontsize=10)
    axes[1].tick_params(axis='both', labelsize=10)
    axes[1].tick_params(axis='x', labelrotation=45)
    axes[1].grid(False)
    plt.suptitle(f"Spatial Block Cross-Validation Results", fontsize=11)

    plt.tight_layout()
    plt.savefig('Spatial cv map Nov.png', dpi=400)

    results = {
        'Random Forest': {'RMSE': rmse_rf, 'R2': r2_rf},
        'XGBoost': {'RMSE': rmse_xgb, 'R2': r2_xgb},
        'Hurdle Model': {'RMSE': rmse_hurdle, 'R2': r2_hurdle},
        'Tweedie GLM': {'RMSE': rmse_tw, 'R2': r2_tw},
        f'KNN (k = {best_k})': {'RMSE': best_rmse, 'R2': best_r2}
    }
    results_df = pd.DataFrame(results).T.sort_values('RMSE')
    print("MODEL COMPARISON")
    print(results_df)
    holdout_results = {
        'Hurdle Model':    {'RMSE': rmse_hurdle,  'R2': r2_hurdle},
        'XGBoost':         {'RMSE': rmse_xgb,     'R2': r2_xgb},
        f'KNN (k={best_k})': {'RMSE': best_rmse,  'R2': best_r2},
        'Random Forest':   {'RMSE': rmse_rf,      'R2': r2_rf},
        'Tweedie GLM':     {'RMSE': rmse_tw,      'R2': r2_tw},
    }
    if gwr_success and gwr_comparable:
        holdout_results['GWR (5 features)'] = {
            'RMSE': gwr_test_rmse,
            'R2':   gwr_test_r2
        }
    holdout_df = pd.DataFrame(holdout_results).T.sort_values('RMSE')

    #Framework 2: Spatial CV
    spatial_cv_results = {
        'Random Forest':    {
            'CV_RMSE': cv_rf_mean,
            'CV_STD':  cv_rf_std
        },
        'XGBoost':          {
            'CV_RMSE': cv_xgb_mean,
            'CV_STD':  cv_xgb_std
        },
        'Hurdle Model':     {
            'CV_RMSE': cv_hurdle_mean,
            'CV_STD':  cv_hurdle_std
        },
        f'KNN (k={best_k})': {
            'CV_RMSE': cv_knn_mean,
            'CV_STD':  cv_knn_std
        },
        'Tweedie GLM':      {
            'CV_RMSE': None,
            'CV_STD':  None
        },
    }
    if gwr_success:
        spatial_cv_results['GWR (5 features)'] = {
            'CV_RMSE': None,   # GWR not run through spatial CV
            'CV_STD':  None
        }

    spatial_cv_df = pd.DataFrame(spatial_cv_results).T
    print("\n=== HOLD-OUT TEST SET RESULTS ===")
    print(holdout_df.to_string())
    print("\n=== SPATIAL CV RESULTS ===")
    print(spatial_cv_df.to_string())



    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)

    # --- Panel 1: Hold-out RMSE ---
    model_names_ho = holdout_df.index.tolist()
    rmse_vals_ho   = holdout_df['RMSE'].tolist()
    rmse_cols_ho   = ['gold' if v == min(rmse_vals_ho)
                    else 'steelblue' for v in rmse_vals_ho]

    bars1 = axes[0, 0].barh(
        model_names_ho, rmse_vals_ho,
        color=rmse_cols_ho, edgecolor='white'
    )
    axes[0, 0].set_xlabel("RMSE", fontsize=13)
    axes[0, 0].set_title(
        "RMSE per model", fontsize=13)
    axes[0, 0].tick_params(axis='both', labelsize=13)
    axes[0, 0].grid(False)
    for bar, val in zip(bars1, rmse_vals_ho):
        axes[0, 0].text(
            bar.get_width() + 0.055,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}',
            va='center', fontsize=12
        )
    axes[0, 0].set_xlim(0, max(rmse_vals_ho) * 1.10)


    # --- Panel 2: Hold-out R² ---
    r2_vals_ho = holdout_df['R2'].tolist()
    r2_cols_ho = ['gold' if v == max(r2_vals_ho)
                else 'steelblue' for v in r2_vals_ho]

    bars2 = axes[0, 1].barh(
        model_names_ho, r2_vals_ho,
        color=r2_cols_ho, edgecolor='white'
    )
    axes[0, 1].axvline(
        x=0, color='red', linestyle='--',
        linewidth=1, alpha=0.7,
        label='R² = 0'
    )
    axes[0, 1].set_xlabel(
        "R²", fontsize=13
    )
    axes[0, 1].set_title("R² per model", fontsize=13)
    axes[0, 1].tick_params(axis='both', labelsize=13)
    axes[0, 1].legend(fontsize=13)
    axes[0, 1].grid(False)
    for bar, val in zip(bars2, r2_vals_ho):
        if val >= 0:
            x = -0.03
            ha = 'right'
        else:
            x = 0.03
            ha = 'left'
        axes[0, 1].text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}',
            va='center', ha=ha, fontsize=12
        )

    # --- Panel 3: Spatial CV RMSE with error bars ---
    cv_models  = ['Random Forest', 'XGBoost',
                'Hurdle Model', f'KNN (k={best_k})']
    cv_means   = [cv_rf_mean, cv_xgb_mean,
                cv_hurdle_mean, cv_knn_mean]
    cv_stds    = [cv_rf_std, cv_xgb_std,
                cv_hurdle_std, cv_knn_std]
    cv_colours = ['gold' if v == min(cv_means) else 'steelblue'
                for v in cv_means]

    # Sort by CV mean RMSE
    cv_order  = np.argsort(cv_means)
    cv_models_sorted  = [cv_models[i] for i in cv_order]
    cv_means_sorted   = [cv_means[i] for i in cv_order]
    cv_stds_sorted    = [cv_stds[i] for i in cv_order]
    cv_colours_sorted = ['gold' if v == min(cv_means_sorted)
                        else 'steelblue' for v in cv_means_sorted]

    bars3 = axes[1, 0].barh(
        cv_models_sorted, cv_means_sorted,
        color=cv_colours_sorted, edgecolor='white',
    )
    axes[1, 0].set_xlabel(
        "Spatial CV RMSE ± Std Dev\n",
        fontsize=10
    )
    axes[1, 0].set_title(
        "Spatial Block Cross-Validation RMSE\n"
        "Gold = best",
        fontsize=10
    )
    axes[1, 0].tick_params(axis='both', labelsize=10)
    axes[1, 0].grid(False)
    for bar, mean_val, std_val in zip(
        bars3, cv_means_sorted, cv_stds_sorted
    ):
        axes[1, 0].text(
            bar.get_width() + std_val + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f'{mean_val:.3f} ± {std_val:.3f}',
            va='center', fontsize=8
        )
    #Panel 4: Per-fold RMSE heatmap
    fold_data = np.array([
        cv_rf_scores,
        cv_xgb_scores,
        cv_hurdle_scores,
        cv_knn_scores
    ])
    im = axes[1, 1].imshow(
        fold_data,
        cmap='RdYlGn_r',
        aspect='auto',
        vmin=0,
        vmax=max(max(cv_rf_scores), max(cv_xgb_scores),
                max(cv_hurdle_scores), max(cv_knn_scores)) * 1.05
    )
    plt.colorbar(
        im, ax=axes[1, 1],
        label='RMSE (stones per borehole)',
        shrink=0.8
    )

    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_xticklabels(
        [f'Fold {i+1}' for i in range(5)], fontsize=10
    )
    axes[1, 1].set_yticks(range(4))
    axes[1, 1].set_yticklabels(
        ['Random Forest', 'XGBoost',
        'Hurdle Model', f'KNN (k={best_k})'],
        fontsize=10
    )
    for i in range(4):
        for j in range(5):
            val = fold_data[i, j]
            text_col = 'white' if val > fold_data.max() * 0.7 \
                    else 'black'
            axes[1, 1].text(
                j, i, f'{val:.2f}',
                ha='center', va='center',
                fontsize=8, color=text_col,
                fontweight='bold'
            )
    axes[1, 1].set_xlabel("Spatial Fold", fontsize=10)
    plt.savefig("Model comparison Nov.png", dpi=400)


    print("\n" + "=" * 70)
    print("FORMAL MODEL SELECTION FRAMEWORK")
    print("=" * 70)
    print("\n--- Criterion 1: Hold-out test set RMSE ---")
    print("(Best single estimate of absolute prediction accuracy)")
    for rank, (model, row) in enumerate(holdout_df.iterrows(), 1):
        print(f"  {rank}. {model:<25} RMSE = {row['RMSE']:.4f}  "
            f"R² = {row['R2']:.4f}")
    print(f"\n--- Criterion 2: Spatial CV mean RMSE ---")
    print("(Most honest estimate — tests true generalisation to new areas)")
    cv_ranking = sorted(
        zip(cv_models, cv_means, cv_stds),
        key=lambda x: x[1]
    )
    for rank, (model, mean_val, std_val) in enumerate(cv_ranking, 1):
        print(f"  {rank}. {model:<25} "
            f"CV RMSE = {mean_val:.4f} ± {std_val:.4f}")
    print(f"\n--- Criterion 3: Spatial CV stability (std dev) ---")
    print("(Low std dev = model performs consistently across all areas)")
    stability_ranking = sorted(
        zip(cv_models, cv_means, cv_stds),
        key=lambda x: x[2]
    )
    score_lookup = {
        "Random Forest": cv_rf_scores,
        "XGBoost": cv_xgb_scores,
        "Hurdle Model": cv_hurdle_scores,
        f"KNN (k={best_k})": cv_knn_scores,
    }
    for rank, (model, mean_val, std_val) in enumerate(stability_ranking, 1):
        scores = score_lookup[model]
        print(
            f"  {rank}. {model:<25} "
            f"Std = {std_val:.4f}  "
            f"(range: {min(scores):.2f} - {max(scores):2f})"
        )
    # Simpler stability print
    print(f"\n--- Criterion 3 (simplified): Spatial CV stability ---")
    stability_data = [
        ('Random Forest',        cv_rf_std,     min(cv_rf_scores),     max(cv_rf_scores)),
        ('XGBoost',              cv_xgb_std,    min(cv_xgb_scores),    max(cv_xgb_scores)),
        ('Hurdle Model',         cv_hurdle_std, min(cv_hurdle_scores), max(cv_hurdle_scores)),
        (f'KNN (k={best_k})',    cv_knn_std,    min(cv_knn_scores),    max(cv_knn_scores)),
    ]
    stability_data.sort(key=lambda x: x[1])
    for rank, (model, std_val, mn, mx) in enumerate(
        stability_data, 1
    ):
        print(f"  {rank}. {model:<25} "
            f"Std = {std_val:.4f}  "
            f"Range: {mn:.2f} — {mx:.2f}")
    if gwr_success:
        print(f"\n--- GWR Spatial R² (supplementary — not directly comparable) ---")
        print(f"  GWR Global R²:       {gwr_global_r2:.4f}")
        print(f"  GWR Adjusted R²:     {gwr_adj_r2:.4f}")
        print(f"  GWR Mean Local R²:   "
            f"{gwr_result_gdf['local_R2'].mean():.4f}")
        print(f"  GWR uses only {len(top_gwr_features)} features "
            f"vs {X_train.shape[1]} for ML models.")
        print(f"  GWR improvement over global OLS: "
            f"R² {gwr_global_r2:.3f} vs 0.181 "
            f"({(gwr_global_r2 - 0.181)/0.181*100:.1f}% improvement)")
        print(f"  GWR purpose: spatial heterogeneity analysis,")
        print(f"  not grade prediction competition.")

    # WEIGHTED SCORING — OVERALL MODEL RECOMMENDATION
    print(f"\n{'='*70}")
    print("WEIGHTED SCORING — OVERALL MODEL RECOMMENDATION")
    print(f"{'='*70}")
    print("Weights: Spatial CV RMSE 40% | Hold-out RMSE 30% |"
        " CV Stability 20% | R² 10%")
    print()
    scoring_models = {
        'Random Forest': {
            'cv_rmse':    cv_rf_mean,
            'cv_std':     cv_rf_std,
            'ho_rmse':    rmse_rf,
            'ho_r2':      r2_rf
        },
        'XGBoost': {
            'cv_rmse':    cv_xgb_mean,
            'cv_std':     cv_xgb_std,
            'ho_rmse':    rmse_xgb,
            'ho_r2':      r2_xgb
        },
        'Hurdle Model': {
            'cv_rmse':    cv_hurdle_mean,
            'cv_std':     cv_hurdle_std,
            'ho_rmse':    rmse_hurdle,
            'ho_r2':      r2_hurdle
        },
        f'KNN (k={best_k})': {
            'cv_rmse':    cv_knn_mean,
            'cv_std':     cv_knn_std,
            'ho_rmse':    best_rmse,
            'ho_r2':      best_r2
        },
    }
    all_cv_rmse  = [v['cv_rmse'] for v in scoring_models.values()]
    all_cv_std   = [v['cv_std']  for v in scoring_models.values()]
    all_ho_rmse  = [v['ho_rmse'] for v in scoring_models.values()]
    all_ho_r2    = [v['ho_r2']   for v in scoring_models.values()]
    def normalise_lower_better(values):
        mn, mx = min(values), max(values)
        if mx == mn:
            return [1.0] * len(values)
        return [(mx - v) / (mx - mn) for v in values]
    def normalise_higher_better(values):
        mn, mx = min(values), max(values)
        if mx == mn:
            return [1.0] * len(values)
        return [(v - mn) / (mx - mn) for v in values]
    norm_cv_rmse = normalise_lower_better(all_cv_rmse)
    norm_cv_std  = normalise_lower_better(all_cv_std)
    norm_ho_rmse = normalise_lower_better(all_ho_rmse)
    norm_ho_r2   = normalise_higher_better(all_ho_r2)
    weights = {
        'cv_rmse': 0.40,
        'ho_rmse': 0.30,
        'cv_std':  0.20,
        'ho_r2':   0.10
    }
    scores = {}
    print(f"{'Model':<25} {'CV RMSE':>10} {'HO RMSE':>10} "
        f"{'CV Std':>10} {'R²':>10} {'Score':>10}")
    print("-" * 77)
    for idx, (model, metrics) in enumerate(scoring_models.items()):
        weighted_score = (
            weights['cv_rmse'] * norm_cv_rmse[idx] +
            weights['ho_rmse'] * norm_ho_rmse[idx] +
            weights['cv_std']  * norm_cv_std[idx]  +
            weights['ho_r2']   * norm_ho_r2[idx]
        )
        scores[model] = weighted_score
        print(f"{model:<25} "
            f"{metrics['cv_rmse']:>10.4f} "
            f"{metrics['ho_rmse']:>10.4f} "
            f"{metrics['cv_std']:>10.4f} "
            f"{metrics['ho_r2']:>10.4f} "
            f"{weighted_score:>10.4f}")
    best_model = max(scores, key=scores.get)
    print(f"\n{'='*70}")
    print(f"RECOMMENDED MODEL: {best_model}")
    print(f"Weighted Score: {scores[best_model]:.4f}")
    print(f"{'='*70}")

    print(f"\nRECOMMENDATIONS:")
    print(f"  Grade estimation (resource):    "
        f"{min(scoring_models, key=lambda m: scoring_models[m]['ho_rmse'])}"
        f"  (lowest hold-out RMSE)")
    print(f"  New area targeting:             "
        f"{min(scoring_models, key=lambda m: scoring_models[m]['cv_rmse'])}"
        f"  (lowest spatial CV RMSE)")
    print(f"  Consistent deployment:          "
        f"{min(scoring_models, key=lambda m: scoring_models[m]['cv_std'])}"
        f"  (lowest CV std dev)")
    print(f"  Spatial heterogeneity analysis: GWR "
        f"(spatially varying coefficients)")
    print(
        f"  Diamond occurrence targeting:   Hurdle Stage 1 "
        f"(AUC = {roc_auc_score(y_test_bin, p_presense):.3f})"
    )
    x_min = results_encoded['E_actual'].min()
    x_max = results_encoded['E_actual'].max()
    y_min = results_encoded['N_actual'].min()
    y_max = results_encoded['N_actual'].max()

    buffer     = 500
    grid_res   = 50  # 50m grid resolution — adjust to your deposit scale
    x_grid = np.arange(x_min - buffer, x_max + buffer, grid_res)
    y_grid = np.arange(y_min - buffer, y_max + buffer, grid_res)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    print(f"Grid dimensions: {xx.shape}")
    print(f"Total grid points: {len(grid_points)}")
    # Get your feature columns excluding coordinates and spatial lag
    grid_feature_cols = [c for c in X_imputed.columns
                        if c not in ['E_actual', 'N_actual',
                                    'spatial_lag_y',
                                    'geological_cluster']]
    print("Interpolating features to grid...") #Prospectivity map
    X_grid_features = {}
    for feat in grid_feature_cols:
        interp = KNR(n_neighbors=3,
                                    weights='distance')
        interp.fit(
            np.c_[results_encoded['E_actual'].values,
                results_encoded['N_actual'].values],
            X_imputed[feat].values
        )
        X_grid_features[feat] = interp.predict(grid_points)
    X_grid = pd.DataFrame(X_grid_features)
    X_grid['E_actual'] = grid_points[:, 0]
    X_grid['N_actual'] = grid_points[:, 1]
    # Compute spatial lag for grid points using borehole values
    w_grid = KNN.from_array(
        np.c_[results_encoded['E_actual'].values,
            results_encoded['N_actual'].values],
        k=5
    )
    lag_interp = KNR(n_neighbors=5, weights='distance')
    lag_interp.fit(
        np.c_[results_encoded['E_actual'].values,
            results_encoded['N_actual'].values],
        y.values
    )
    X_grid['spatial_lag_y'] = lag_interp.predict(grid_points)
    cluster_interp = KNR(
        n_neighbors=1  # hard assignment to nearest cluster
    )
    cluster_interp.fit(
        np.c_[results_encoded['E_actual'].values,
            results_encoded['N_actual'].values],
        cluster_labels
    )
    X_grid['geological_cluster'] = cluster_interp.predict(
        grid_points
    ).round().astype(int)
    X_grid = X_grid[X_imputed.columns]
    X_grid = X_grid[X_train.columns].copy()
    grid_pred_rf = np.expm1(rf.predict(X_grid))
    # Hurdle Stage 1 — probability of diamond occurrence
    grid_prob_presence = clf_stage1.predict_proba(X_grid)[:, 1]
    # Reshape to grid
    pred_grid_rf       = grid_pred_rf.reshape(xx.shape)
    pred_grid_presence = grid_prob_presence.reshape(xx.shape)
    plt.figure(figsize=(10, 10))
    pred_display = pred_grid_rf.copy()
    pred_display[pred_display < 0.5] = np.nan
    im2 = plt.contourf(
        xx, yy, pred_grid_presence,
        levels=20,
        cmap='RdYlGn',
        alpha=0.85,
        vmin=0,
        vmax=1
    )
    # Overlay boreholes coloured by actual presence/absence
    zero_mask = (y.values == 0)
    nonzero_mask = (y.values > 0)
    plt.scatter(
        results_encoded['E_actual'].values[zero_mask],
        results_encoded['N_actual'].values[zero_mask],
        c='red', s=5, alpha=0.4,
        label='Zero recovery', zorder=5
    )
    plt.scatter(
        results_encoded['E_actual'].values[nonzero_mask],
        results_encoded['N_actual'].values[nonzero_mask],
        c='green', s=5, alpha=0.6,
        label='Diamond recovery', zorder=5
    )
    cbar2 = plt.colorbar(im2, shrink=0.8)
    cbar2.set_label('P(Diamond Occurrence)', fontsize=9)
    plt.title("Probability of Diamond Occurrence", fontsize=10)
    plt.xlabel("Easting (m)", fontsize=10)
    plt.ylabel("Northing (m)", fontsize=10, labelpad=-1)
    plt.tick_params(axis='both', labelsize=9)
    plt.legend(fontsize=9, markerscale=2)
    plt.grid(False)
    plt.tight_layout()

    residuals    = y_test.values - y_pred_rf
    abs_residuals = np.abs(residuals)
    pct_error    = np.where(
        y_test.values > 0,
        np.abs(residuals) / y_test.values * 100,
        np.nan
    )
    residual_df = pd.DataFrame({
        'actual':       y_test.values,
        'predicted':    y_pred_rf,
        'residual':     residuals,
        'abs_residual': abs_residuals,
        'pct_error':    pct_error,
        'E_actual':     X_test['E_actual'].values,
        'N_actual':     X_test['N_actual'].values
    })
    actual_max = residual_df['actual'].max()
    if actual_max > 100:
        bins   = [0, 1, 5, 10, 25, 100, actual_max + 1]
        labels = ['Zero', '1-5', '5-10', '10-25', '25-100', '>100']
    else:
        bins   = [0, 1, 5, 10, 25, actual_max + 1]
        labels = ['Zero', '1-5', '5-10', '10-25', f'25-{actual_max:.0f}']
    print(f"Actual max in test set: {actual_max}")
    print(f"Bins used: {bins}")

    grade_bands = pd.cut(
        residual_df['actual'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    residual_df['grade_band'] = grade_bands
    band_stats = residual_df.groupby('grade_band').agg(
        count=('actual', 'count'),
        mean_actual=('actual', 'mean'),
        mean_abs_error=('abs_residual', 'mean'),
        median_abs_error=('abs_residual', 'median'),
        mean_pct_error=('pct_error', 'mean')
    ).round(3)

    print("=== RESIDUAL ANALYSIS BY GRADE BAND ===")
    print(band_stats.to_string())

    y_pred_class_default = (p_presense >= 0.5).astype(int)
    thresholds  = np.arange(0.1, 0.9, 0.01)
    f1_scores   = []
    precisions  = []
    recalls     = []
    for thresh in thresholds:
        y_pred_t = (p_presense >= thresh).astype(int)
        tp = ((y_pred_t == 1) & (y_test_bin == 1)).sum()
        fp = ((y_pred_t == 1) & (y_test_bin == 0)).sum()
        fn = ((y_pred_t == 0) & (y_test_bin == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall /
                    (precision + recall)
                    if (precision + recall) > 0 else 0)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold  = thresholds[best_thresh_idx]
    best_f1         = f1_scores[best_thresh_idx]
    print(f"Default threshold (0.5):")
    print(f"  F1 = {f1_scores[list(thresholds).index(min(thresholds, key=lambda x: abs(x-0.5)))]:.4f}")
    print(f"\nOptimal threshold: {best_threshold:.2f}")
    print(f"  F1 = {best_f1:.4f}")
    print(f"  Precision = {precisions[best_thresh_idx]:.4f}")
    print(f"  Recall = {recalls[best_thresh_idx]:.4f}")
    y_pred_class_optimal = (p_presense >= best_threshold).astype(int)

    #Confusion matrices
    cm_default = confusion_matrix(y_test_bin, y_pred_class_default)
    cm_optimal = confusion_matrix(y_test_bin, y_pred_class_optimal)

    def get_cm_stats(cm):
        tn, fp, fn, tp = cm.ravel()
        total    = tn + fp + fn + tp
        accuracy = (tp + tn) / total * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall   = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        f1 = (2 * precision * recall /
            (precision + recall)
            if (precision + recall) > 0 else 0)
        return {
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'accuracy':    accuracy,
            'precision':   precision,
            'recall':      recall,
            'specificity': specificity,
            'f1':          f1
        }
    stats_default = get_cm_stats(cm_default)
    stats_optimal = get_cm_stats(cm_optimal)
    print("\n=== DEFAULT THRESHOLD (0.5) ===")
    for k, v in stats_default.items():
        print(f"  {k:<12}: {v:.1f}" if isinstance(v, float)
            else f"  {k:<12}: {v}")
    print(f"\n=== OPTIMAL THRESHOLD ({best_threshold:.2f}) ===")
    for k, v in stats_optimal.items():
        print(f"  {k:<12}: {v:.1f}" if isinstance(v, float)
            else f"  {k:<12}: {v}")

    #Confusion matrices
    fig = plt.figure(figsize=(12, 12))
    gs  = gridspec.GridSpec(1, 2, figure=fig, hspace=0.6, wspace=0.6)
    #Plot 1: Confusion matrix — default threshold
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(cm_default, interpolation='nearest',
                    cmap='Blues')
    plt.colorbar(im1, ax=ax1, shrink=0.5)
    labels = ['Absent\n(Zero recovery)',
            'Present\n(Diamond recovery)']
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel("Predicted", fontsize=11)
    ax1.set_ylabel("Actual", fontsize=11, labelpad=-5)

    # Annotate cells
    total = cm_default.sum()
    for i in range(2):
        for j in range(2):
            count = cm_default[i, j]
            pct   = count / total * 100
            colour = 'white' if count > cm_default.max() * 0.5 \
                    else 'black'
            # Label what each cell means
            cell_labels = {
                (0, 0): 'True Negative\n(Correctly predicted\nbarren)',
                (0, 1): 'False Positive\n(Incorrectly predicted\ndiamond-bearing)',
                (1, 0): 'False Negative\n(Missed\ndiamond zone)',
                (1, 1): 'True Positive\n(Correctly predicted\ndiamond-bearing)'
            }
            ax1.text(j, i,
                    f"{cell_labels[(i,j)]}\n\nn={count}\n({pct:.1f}%)",
                    ha='center', va='center',
                    fontsize=9, color=colour)
    ax1.set_title(f"Confusion Matrix at defualt threshold (0.50)", fontsize=10)

    #Plot 2: Confusion matrix — optimal threshold
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(cm_optimal, interpolation='nearest',
                    cmap='Greens')
    plt.colorbar(im2, ax=ax2, shrink=0.5)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel("Predicted", fontsize=11)
    ax2.set_ylabel("Actual", fontsize=11, labelpad=-5)
    total_opt = cm_optimal.sum()
    for i in range(2):
        for j in range(2):
            count = cm_optimal[i, j]
            pct   = count / total_opt * 100
            colour = 'white' if count > cm_optimal.max() * 0.5 \
                    else 'black'
            ax2.text(j, i,
                    f"{cell_labels[(i,j)]}\n\nn={count}\n({pct:.1f}%)",
                    ha='center', va='center',
                    fontsize=9, color=colour)
    ax2.set_title(f"Confusion Matrix at optimal threshold (0.59)", fontsize=10)
    plt.suptitle(
        "Hurdle Model Stage 1 — Binary Classification Performance\n"
        "Diamond Occurrence Prediction: "
        "Confusion Matrix and Diagnostic Curves",
        fontsize=10
    )
    plt.savefig("Confusion matrices_Nov.png", dpi=400)

    tn = stats_optimal['TN']
    fp = stats_optimal['FP']
    fn = stats_optimal['FN']
    tp = stats_optimal['TP']
    total_test = tn + fp + fn + tp
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX — GEOLOGICAL INTERPRETATION")
    print(f"Optimal threshold: {best_threshold:.2f}")
    print("=" * 60)
    print(f"\nOf {total_test} test boreholes:")
    print(f"\n  Correctly identified barren ground (TN): {tn}")
    print(f"  ({tn/total_test*100:.1f}% of all test boreholes)")
    print(f"  These locations would correctly be de-prioritised")
    print(f"  in future drilling programmes")
    print(f"\n  Correctly identified diamond zones (TP): {tp}")
    print(f"  ({tp/total_test*100:.1f}% of all test boreholes)")
    print(f"  These locations would correctly be prioritised")
    print(f"\n  False alarms — barren ground flagged as "
        f"diamond-bearing (FP): {fp}")
    print(f"  ({fp/total_test*100:.1f}% of all test boreholes)")
    print(f"  Each represents a wasted drill hole if acted upon")
    print(f"\n  Missed diamond zones — diamond-bearing ground "
        f"flagged as barren (FN): {fn}")
    print(f"  ({fn/total_test*100:.1f}% of all test boreholes)")
    print(f"  Each represents a missed resource discovery")
    print(f"\n  Of all predicted diamond-bearing locations,")
    print(f"  {stats_optimal['precision']:.1f}% were genuinely "
        f"diamond-bearing (Precision)")
    print(f"\n  Of all actual diamond zones,")
    print(f"  {stats_optimal['recall']:.1f}% were correctly "
        f"identified (Recall)")


    #In order to get the following workflow to run:
    # 1. Remove the """ at lines 2689 and 2777
    # 2. Comment (#) out line 1049
    # 3. Un-comment lines 1084 and 1085 - Commented out to prevent data leakage
    """
    print("=== DRILL TIME SUMMARY ===")
    print(f"Mean drill time:    {drill_time.mean():.2f} mins")
    print(f"Median drill time:  {np.median(drill_time):.2f} mins")
    print(f"Min drill time:     {drill_time.min():.2f} mins")
    print(f"Max drill time:     {drill_time.max():.2f} mins")
    print(f"Total drill hours:  {drill_time.sum():.0f} mins")

    print("\n=== TOTAL CARATS SUMMARY ===")
    print(f"Mean carats/borehole:   {total_carats.mean():.4f} ct")
    print(f"Median carats/borehole: {np.median(total_carats):.4f} ct")
    print(f"Max carats/borehole:    {total_carats.max():.4f} ct")
    print(f"Zero-carat boreholes:   "
        f"{(total_carats == 0).sum()} "
        f"({(total_carats == 0).mean()*100:.1f}%)")
    print(f"Total carats recovered: "
        f"{total_carats.sum():.2f} ct")

    # Average stone weight (carats per stone)
    nonzero_stones  = results_encoded['Stones'].values > 0
    avg_stone_weight = (
        total_carats[nonzero_stones].sum() /
        results_encoded['Stones'].values[nonzero_stones].sum()
    )
    print(f"\nAverage stone weight: {avg_stone_weight:.4f} ct/stone")
    # Split drill time and carats aligned to your existing split
    # Using the same random_state=42 ensures identical split
    (drill_time_train, drill_time_test,
    carats_train, carats_test) = (
        drill_time[X_train.index],
        drill_time[X_test.index],
        total_carats[X_train.index],
        total_carats[X_test.index]
    )

    print(f"\nTrain set drill time: {drill_time_train.sum():.0f} total mins")
    print(f"Test set drill time: {drill_time_test.sum():.0f} total mins")
    print(f"Train set carats: {carats_train.sum():.2f} total carats")
    print(f"Test set carats: {carats_test.sum():.2f} total carats")

    print("\n=== FITTING DRILL TIME MODEL (XGBoost) ===")
    xgb_drilltime = xgb.XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1, eval_metric='rmse')
    xgb_drilltime.fit(X_train, np.log1p(drill_time_train), eval_set=[(X_test, np.log1p(drill_time_test))], verbose=False)
    drill_time_pred = np.clip(np.expm1(xgb_drilltime.predict(X_test)), 0, None)

    rmse_dt = np.sqrt(mean_squared_error(drill_time_test, drill_time_pred))
    r2_dt   = r2_score(drill_time_test, drill_time_pred)
    print(f"XGBoost Drill time RMSE: {rmse_dt:.4f} hours")
    print(f"XGBoost Drill time R²:   {r2_dt:.4f}")

    # Feature importance for drill time
    dt_importances = pd.Series(xgb_drilltime.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)

    print(f"\nTop 10 predictors of drill time:")
    for feat, imp in dt_importances.items():
        print(f"{clean_feature_name(feat):<35} {imp:.4f}")

    print("\n=== FITTING CARAT HURDLE MODEL ===")
    print("Stage 1 for carats: reusing existing calibrated "
        "classifier")
    print(f"  AUC = {roc_auc_score(y_test_bin, p_presense):.4f}")
    print(f"  This is valid because carats=0 exactly when "
        f"stones=0")
    print(f"  Same presence/absence structure applies")

    # Verify carat-stone relationship
    carats_zero_mask   = carats_train == 0
    stones_zero_mask   = y_train.values == 0
    agreement          = (carats_zero_mask == stones_zero_mask).mean()
    print(f"  Zero agreement between carats and stones: "
        f"{agreement*100:.1f}%")
    if agreement < 0.99:
        print(f"  WARNING: {(1-agreement)*100:.1f}% of boreholes have disagreement between stone and carat zeros.")
        print(f"  Investigating...")
        disagree_mask = carats_zero_mask != stones_zero_mask
        print(f"  Stones>0 but carats=0: "
            f"{((y_train.values > 0) & carats_zero_mask).sum()}")
        print(f"  Stones=0 but carats>0: "
            f"{((y_train.values == 0) & ~carats_zero_mask).sum()}")
    """

def main():
    orientation_info()
    corr_value = correlations()
    #correlations()
    fig, ax_s1, ax_s2, ax_s3, ax_s4, ax_r1, ax_r2, ax_r3, ax_r4 = orientation_length_graph(corr_value)
    faults()
    offset_drag()
    fault_groups()
    fault_clusters()
    secondary_clusters()
    rose_diag_sec(ax_r2)
    rose_diag_prim(ax_r1)
    bh_data() #KEEP
    normalised()
    models()
    plt.show()

main()